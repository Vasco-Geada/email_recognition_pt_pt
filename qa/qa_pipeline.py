"""
qa_pipeline.py
==============

Pipeline integrado de Question Answering.

Orquestra:
1. Carregamento de emails
2. Inferência QA
3. Pós-processamento
4. Integração com pipeline NLP existente
5. Exportação de resultados

Características:
- Interface simples e intuitiva
- Integração com gold annotations
- Suporte a batch processing
- Caching
- Logging detalhado
- Exportação de resultados

Project: Email Recognition PT-PT
Version: 1.0
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime

from qa_questions import QAQuestions, QuestionCategory
from qa_inference import QAInferenceEngine, QAResultsCache
from qa_utils import (
    QAResult,
    AnswerPostProcessor,
    TextNormalizer,
    setup_logging,
)

try:
    from preprocessing.temporal_normalization import TemporalNormalizer
except Exception:  # pragma: no cover - optional when QA is used standalone
    TemporalNormalizer = None


logger = logging.getLogger(__name__)


@dataclass
class EmailQAResult:
    """Resultado QA para um email."""
    
    email_id: Optional[int]
    email_text: str
    subject: Optional[str]
    qa_results: Dict[str, Dict[str, Any]]  # {category: {answer, confidence, question}}
    processed_at: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário."""
        return {
            'email_id': self.email_id,
            'email_text': self.email_text,
            'subject': self.subject,
            'qa_results': self.qa_results,
            'processed_at': self.processed_at,
            'metadata': self.metadata,
        }
    
    def get_answers_only(self) -> Dict[str, str]:
        """Extrai apenas respostas."""
        return {
            category: result['answer']
            for category, result in self.qa_results.items()
            if result and 'answer' in result
        }


class QAContextEnricher:
    """Adds explicit metadata and deterministic fallbacks around extractive QA."""

    REMOTE_DEFAULT = "remoto - zoom"
    NON_MEETING_LABELS = {"nao_reuniao", "nao reuniao", "não_reuniao", "não reuniao"}
    SENDER_KEYS = (
        "sender",
        "sender_name",
        "sender_email",
        "from",
        "from_email",
        "email_from",
        "remetente",
        "author",
        "name",
        "nome",
    )
    LOCATION_METADATA_KEYS = (
        "location",
        "locations",
        "meeting_location",
        "local",
        "localizacao",
        "localização",
    )
    LOCATION_PATTERN = re.compile(
        r"\b("
        r"zoom|teams|microsoft\s+teams|google\s+meet|meet|skype|discord|"
        r"online|remot[ao]|presencial|"
        r"sala(?:\s+de\s+reuni(?:ao|oes|ões))?\s+[A-Za-z0-9]+(?:[.\-][A-Za-z0-9]+)?|"
        r"gabinete(?:\s+(?:da\s+direcao|de\s+\w+|\d+(?:[.\-]\d+)?))?|"
        r"secretaria(?!\s+academica)|"
        r"biblioteca|bar\s+da\s+faculdade|"
        r"laborat(?:orio|ório)|lab\b|audit(?:orio|ório)(?:\s+\w+)?|"
        r"campus(?:\s+\w+)?|departamento|reitoria|universidade|faculdade"
        r")\b",
        re.IGNORECASE,
    )
    REMOTE_PATTERN = re.compile(
        r"\b(zoom|teams|microsoft\s+teams|google\s+meet|meet|online|remot[ao])\b",
        re.IGNORECASE,
    )
    ROOM_DEPARTMENT_PATTERN = re.compile(
        r"\bsala\s+de\s+reuni(?:ao|oes|ões|Ãµes)\s+do\s+departamento\b",
        re.IGNORECASE,
    )
    ROOM_PATTERN = re.compile(
        r"\bsala(?:\s+de\s+reuni(?:ao|oes|Ãµes))?(?:\s+[A-Za-z0-9]+(?:[.\-][A-Za-z0-9]+)?)?(?:\s+do\s+departamento)?\b",
        re.IGNORECASE,
    )
    OFFICE_PATTERN = re.compile(
        r"\bgabinete(?:\s+(?:da\s+direcao|da\s+direÃ§Ã£o|de\s+\w+|\d+(?:[.\-]\d+)?))?\b",
        re.IGNORECASE,
    )
    PRESENTIAL_PATTERN = re.compile(
        r"\b(secretaria(?!\s+academica)|biblioteca|laborat(?:orio|Ã³rio)(?:\s+de\s+\w+)?|audit(?:orio|Ã³rio)(?:\s+\w+)?|campus(?:\s+\w+)?)\b",
        re.IGNORECASE,
    )
    TEMPORAL_PATTERNS = [
        re.compile(
            r"\b(hoje|amanh[ãa]|depois de amanh[ãa]|mais logo|"
            r"pr[óo]xima semana|na pr[óo]xima semana|"
            r"segunda|ter[çc]a|quarta|quinta|sexta|s[áa]bado|domingo)"
            r"(?:\s+(?:de|à|a|pela|ao)\s+(?:manh[ãa]|tarde|noite))?"
            r"(?:\s+(?:às|as|a)\s+\d{1,2}(?:h|:\d{2})?)?",
            re.IGNORECASE,
        ),
        re.compile(r"\b\d{1,2}(?:h|:\d{2})(?:\d{2})?\b", re.IGNORECASE),
        re.compile(r"\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b", re.IGNORECASE),
        re.compile(r"\bdaqui\s+a\s+\d+\s+(?:dias|semanas|horas)\b", re.IGNORECASE),
    ]

    def __init__(self, reference_datetime: Optional[datetime] = None) -> None:
        self.temporal_normalizer = TemporalNormalizer() if TemporalNormalizer else None
        self.reference_datetime = reference_datetime

    def build_context(
        self,
        email_text: str,
        subject: Optional[str],
        metadata: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        metadata = metadata or {}
        reference_datetime = self.extract_reference_datetime(metadata)
        sender = self.extract_sender(metadata, email_text)
        entity_values = self.extract_entity_values(metadata, sender)
        temporal_hints = self.extract_temporal_hints(email_text, reference_datetime)
        explicit_location = self.extract_explicit_location(email_text)
        if explicit_location and self.matches_entity_value(explicit_location, entity_values):
            explicit_location = None
        metadata_location = self.extract_metadata_location(metadata)
        meeting_related = self.is_meeting_related(email_text, subject, metadata)

        hints = []
        if subject:
            hints.append(f"Assunto: {subject}")
        if sender:
            hints.append(f"Remetente: {sender}")
            hints.append(f"Participante por defeito quando nao ha outro participante explicito: {sender}")
        if temporal_hints:
            hints.append("Expressoes temporais normalizadas:")
            for hint in temporal_hints:
                normalized = hint.get("normalized_datetime") or hint.get("normalized_date") or ""
                hints.append(f"- {hint['text']} -> {normalized}".rstrip())

        enriched_context = email_text
        if hints:
            enriched_context = f"{email_text}\n\nInformacao auxiliar:\n" + "\n".join(hints)

        return {
            "context": enriched_context,
            "subject": subject,
            "sender": sender,
            "metadata": metadata,
            "metadata_location": metadata_location,
            "entity_values": entity_values,
            "temporal_hints": temporal_hints,
            "reference_datetime": reference_datetime,
            "explicit_location": explicit_location,
            "meeting_related": meeting_related,
            "hints": hints,
        }

    def apply_fallbacks(
        self,
        qa_results: Dict[str, Dict[str, Any]],
        email_text: str,
        enrichment: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        results = {category: dict(result) for category, result in qa_results.items()}

        participant = results.get("participants", {})
        if self.is_bad_answer(participant, email_text) and enrichment.get("sender"):
            results["participants"] = self.fallback_result(
                participant,
                enrichment["sender"],
                "fallback_sender",
                confidence=0.7,
            )

        time_result = results.get("time", {})
        temporal_hints = enrichment.get("temporal_hints", [])
        if self.is_bad_answer(time_result, email_text) and temporal_hints:
            first_hint = temporal_hints[0]
            results["time"] = self.fallback_result(
                time_result,
                first_hint["text"],
                "fallback_temporal_normalization",
                confidence=0.75,
                extra={"normalized": first_hint},
            )

        location = results.get("location", {})
        if self.is_non_meeting(enrichment.get("metadata", {})):
            results["location"] = self.empty_result(location, "non_meeting_no_location")
        elif not self.is_bad_location_answer(location, email_text, enrichment):
            normalized_location = self.normalize_location_answer(str(location.get("answer") or ""))
            if normalized_location != location.get("answer"):
                results["location"] = self.fallback_result(
                    location,
                    normalized_location,
                    "normalized_location",
                    confidence=float(location.get("confidence") or 0.0),
                )
        else:
            metadata_location = enrichment.get("metadata_location")
            explicit_location = enrichment.get("explicit_location")
            if metadata_location:
                results["location"] = self.fallback_result(
                    location,
                    metadata_location,
                    "fallback_metadata_location",
                    confidence=0.9,
                )
            elif explicit_location:
                results["location"] = self.fallback_result(
                    location,
                    explicit_location,
                    "fallback_explicit_location",
                    confidence=0.8,
                )
            elif enrichment.get("meeting_related"):
                results["location"] = self.fallback_result(
                    location,
                    self.REMOTE_DEFAULT,
                    "default_remote_location",
                    confidence=0.6,
                )

        topic = results.get("topic", {})
        if self.is_bad_answer(topic, email_text) and enrichment.get("subject"):
            results["topic"] = self.fallback_result(
                topic,
                enrichment["subject"],
                "fallback_subject",
                confidence=0.65,
            )

        return results

    @classmethod
    def empty_result(cls, original: Dict[str, Any], source: str) -> Dict[str, Any]:
        result = dict(original)
        result.update(
            {
                "answer": None,
                "confidence": 0.0,
                "valid": False,
                "fallback_source": source,
            }
        )
        return result

    @classmethod
    def fallback_result(
        cls,
        original: Dict[str, Any],
        answer: str,
        source: str,
        confidence: float,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        result = dict(original)
        result.update(
            {
                "answer": answer,
                "confidence": confidence,
                "valid": True,
                "fallback_source": source,
            }
        )
        if extra:
            result.update(extra)
        return result

    @staticmethod
    def is_bad_answer(result: Dict[str, Any], email_text: str) -> bool:
        answer = (result or {}).get("answer")
        if not answer:
            return True
        answer_text = str(answer).strip()
        if not answer_text:
            return True
        confidence = float((result or {}).get("confidence") or 0.0)
        if confidence < 0.35:
            return True
        normalized_answer = TextNormalizer.clean_text(answer_text, lowercase=True)
        normalized_email = TextNormalizer.clean_text(email_text, lowercase=True)
        if normalized_answer == normalized_email:
            return True
        if QAContextEnricher.is_auxiliary_answer(answer_text):
            return True
        if len(answer_text.split()) > 8:
            return True
        return False

    @staticmethod
    def is_bad_location_answer(
        result: Dict[str, Any],
        email_text: str,
        enrichment: Dict[str, Any],
    ) -> bool:
        if QAContextEnricher.is_bad_answer(result, email_text):
            return True

        answer = str((result or {}).get("answer") or "").strip()
        if not answer:
            return True

        normalized_answer = TextNormalizer.clean_text(answer, lowercase=True)
        if normalized_answer in {"em", "no", "na", "por", "via", "onde"}:
            return True

        for value in enrichment.get("entity_values", []):
            if QAContextEnricher.matches_entity_value(normalized_answer, [str(value)]):
                return True

        subject = enrichment.get("subject")
        if subject and normalized_answer == TextNormalizer.clean_text(str(subject), lowercase=True):
            return True

        if QAContextEnricher.is_temporal_like(answer, enrichment):
            return True

        if QAContextEnricher.LOCATION_PATTERN.search(answer):
            return False

        return True

    @classmethod
    def normalize_location_answer(cls, answer: str) -> str:
        answer = re.sub(r"^\s*(?:em|no|na|por|via)\s+", "", answer.strip(), flags=re.IGNORECASE)
        remote_match = cls.REMOTE_PATTERN.search(answer)
        if remote_match:
            return f"remoto - {cls.normalize_remote_platform(remote_match.group(0))}"
        room_department_match = cls.ROOM_DEPARTMENT_PATTERN.search(answer)
        if room_department_match:
            return f"presencial - {room_department_match.group(0).strip()}"
        room_match = cls.ROOM_PATTERN.search(answer)
        if room_match:
            return f"presencial - {room_match.group(0).strip()}"
        office_match = cls.OFFICE_PATTERN.search(answer)
        if office_match:
            return f"presencial - {office_match.group(0).strip()}"
        presential_match = cls.PRESENTIAL_PATTERN.search(answer)
        if presential_match:
            return f"presencial - {presential_match.group(0).strip()}"
        return answer

    @staticmethod
    def normalize_remote_platform(platform: str) -> str:
        text = str(platform or "").strip().lower()
        if "team" in text:
            return "teams"
        if "meet" in text:
            return "google meet"
        if "zoom" in text:
            return "zoom"
        return "zoom"

    @staticmethod
    def matches_entity_value(answer: str, entity_values: List[str]) -> bool:
        normalized_answer = TextNormalizer.clean_text(str(answer), lowercase=True)
        if not normalized_answer:
            return False
        for value in entity_values:
            normalized_value = TextNormalizer.clean_text(str(value), lowercase=True)
            if normalized_value and (
                normalized_answer == normalized_value
                or normalized_value in normalized_answer
                or normalized_answer in normalized_value
            ):
                return True
        return False

    @classmethod
    def is_temporal_like(cls, answer: str, enrichment: Dict[str, Any]) -> bool:
        answer_norm = TextNormalizer.clean_text(answer, lowercase=True)
        if not answer_norm:
            return False
        for hint in enrichment.get("temporal_hints", []):
            hint_text = TextNormalizer.clean_text(str(hint.get("text") or ""), lowercase=True)
            if hint_text and (answer_norm == hint_text or answer_norm in hint_text or hint_text in answer_norm):
                return True
        return any(pattern.search(answer) for pattern in cls.TEMPORAL_PATTERNS)

    @staticmethod
    def is_auxiliary_answer(answer: str) -> bool:
        answer = TextNormalizer.clean_text(answer, lowercase=True)
        auxiliary_markers = [
            "informacao auxiliar",
            "assunto",
            "remetente",
            "participante por defeito",
            "localizacao por defeito",
            "expressoes temporais normalizadas",
        ]
        return any(marker in answer for marker in auxiliary_markers)

    @classmethod
    def extract_sender(cls, metadata: Dict[str, Any], email_text: str) -> Optional[str]:
        for key in cls.SENDER_KEYS:
            value = metadata.get(key)
            if value:
                return cls.clean_sender(str(value))

        from_header = re.search(r"\b(?:from|de):\s*([^\n<]+)(?:<([^>]+)>)?", email_text, re.IGNORECASE)
        if from_header:
            name = from_header.group(1).strip()
            email = from_header.group(2).strip() if from_header.group(2) else ""
            return cls.clean_sender(f"{name} {email}".strip())

        email_match = re.search(r"\b[\w.\-+%]+@[\w.\-]+\.[A-Za-z]{2,}\b", email_text)
        if email_match:
            return email_match.group(0)

        return None

    @classmethod
    def extract_metadata_location(cls, metadata: Dict[str, Any]) -> Optional[str]:
        for key in cls.LOCATION_METADATA_KEYS:
            value = metadata.get(key)
            if not value:
                continue
            if isinstance(value, list):
                value = next((item for item in value if item), None)
            if not value:
                continue
            text = cls.normalize_location_answer(str(value))
            return text or None
        return None

    @classmethod
    def extract_entity_values(cls, metadata: Dict[str, Any], sender: Optional[str]) -> List[str]:
        values: List[str] = []
        for key in (
            "sender",
            "sender_name",
            "sender_email",
            "recipient",
            "recipient_name",
            "recipient_email",
            "from",
            "to",
            "author",
            "participants",
        ):
            value = metadata.get(key)
            if isinstance(value, list):
                values.extend(str(item) for item in value if item)
            elif value:
                values.append(str(value))
        if sender:
            values.append(sender)
        return list(dict.fromkeys(value.strip() for value in values if value and value.strip()))

    @staticmethod
    def clean_sender(value: str) -> str:
        value = re.sub(r"\s+", " ", value).strip()
        value = value.strip("<>")
        return value or ""

    def extract_temporal_hints_with_reference(
        self,
        email_text: str,
        reference_datetime: Optional[datetime],
    ) -> List[Dict[str, Any]]:
        seen = set()
        hints = []
        for pattern in self.TEMPORAL_PATTERNS:
            for match in pattern.finditer(email_text):
                text = match.group(0).strip()
                key = text.lower()
                if not text or key in seen:
                    continue
                seen.add(key)
                hint = {"text": text}
                if self.temporal_normalizer:
                    normalized = self.temporal_normalizer.normalize(
                        text,
                        reference_datetime=reference_datetime,
                    )
                    hint.update(normalized.to_dict())
                hints.append(hint)
        return hints

    def extract_temporal_hints(
        self,
        email_text: str,
        reference_datetime: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        return self.extract_temporal_hints_with_reference(
            email_text,
            reference_datetime or self.reference_datetime,
        )

    def extract_reference_datetime(self, metadata: Dict[str, Any]) -> Optional[datetime]:
        for key in ("sent_datetime", "sent_at", "email_date", "date", "created_at"):
            value = metadata.get(key)
            if not value:
                continue
            try:
                return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            except ValueError:
                continue
        return self.reference_datetime

    @staticmethod
    def attach_temporal_metadata(
        qa_results: Dict[str, Dict[str, Any]],
        temporal_hints: List[Dict[str, Any]],
    ) -> None:
        time_result = qa_results.get("time")
        if not time_result or not temporal_hints:
            return

        answer = str(time_result.get("answer") or "").lower()
        if not answer:
            return

        for hint in temporal_hints:
            hint_text = str(hint.get("text") or "").lower()
            if hint_text and (hint_text in answer or answer in hint_text):
                time_result.setdefault("normalized", hint)
                return

    @classmethod
    def extract_explicit_location(cls, email_text: str) -> Optional[str]:
        for match in cls.LOCATION_PATTERN.finditer(email_text or ""):
            prefix = (email_text or "")[max(0, match.start() - 10):match.start()]
            if re.search(r"(?:em|no|na|por|via)\s+$", prefix, re.IGNORECASE):
                return cls.normalize_location_answer(match.group(0).strip())
        return None

    @classmethod
    def is_non_meeting(cls, metadata: Dict[str, Any]) -> bool:
        intent = metadata.get("intent") or metadata.get("label") or metadata.get("predicted_intent")
        return str(intent or "").strip().lower() in cls.NON_MEETING_LABELS

    @staticmethod
    def is_meeting_related(
        email_text: str,
        subject: Optional[str],
        metadata: Dict[str, Any],
    ) -> bool:
        intent = metadata.get("intent") or metadata.get("label") or metadata.get("predicted_intent")
        if intent and str(intent).strip().lower() not in QAContextEnricher.NON_MEETING_LABELS:
            return True
        combined = f"{subject or ''} {email_text or ''}".lower()
        return any(term in combined for term in ["reuni", "call", "disponibilidade", "combinar", "encontro"])


class QAPipeline:
    """
    Pipeline integrado de Question Answering.
    
    Orquestra todo o fluxo:
    email -> perguntas -> respostas -> estrutura
    """
    
    def __init__(
        self,
        model_name: str = 'bertimbau-pt',
        device: str = 'cuda',
        confidence_threshold: float = 0.5,
        use_cache: bool = True,
        cache_size: int = 1000,
        verbose: bool = True,
        reference_datetime: Optional[datetime] = None,
    ):
        """
        Inicializa pipeline QA.
        
        Args:
            model_name: Nome do modelo
            device: Dispositivo ('cuda' ou 'cpu')
            confidence_threshold: Threshold de confiança
            use_cache: Se True, usa cache
            cache_size: Tamanho do cache
            verbose: Se True, log detalhado
            reference_datetime: Referencia para normalizacao temporal
        """
        self.model_name = model_name
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.verbose = verbose
        self.context_enricher = QAContextEnricher(reference_datetime=reference_datetime)
        
        # Setup logging
        self.logger = setup_logging(
            __name__,
            level=logging.INFO if verbose else logging.WARNING,
        )
        
        # Inicializar engine de QA
        try:
            self.logger.info(f"Inicializando QA Pipeline com modelo: {model_name}")
            self.qa_engine = QAInferenceEngine(
                model_name=model_name,
                device=device,
                confidence_threshold=confidence_threshold,
            )
            self.engine_available = True
        except Exception as e:
            self.logger.error(f"Erro ao inicializar engine: {str(e)}")
            self.engine_available = False
            self.qa_engine = None
        
        # Cache
        self.cache = QAResultsCache(max_size=cache_size) if use_cache else None
        self.logger.info("Pipeline QA inicializado")
    
    def process_email(
        self,
        email_text: str,
        email_id: Optional[int] = None,
        subject: Optional[str] = None,
        use_cache: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[EmailQAResult]:
        """
        Processa email e responde perguntas.
        
        Args:
            email_text: Texto do email
            email_id: ID do email (opcional)
            subject: Assunto do email (opcional)
            use_cache: Se True, usa cache
            metadata: Metadados do email usados para contexto auxiliar
            
        Returns:
            EmailQAResult com respostas estruturadas
        """
        if not self.engine_available:
            self.logger.warning("QA Engine não disponível")
            return None
        
        if not email_text or len(email_text.strip()) == 0:
            self.logger.warning("Email vazio")
            return None
        
        try:
            # Limpar texto
            email_text = TextNormalizer.normalize_whitespace(email_text)
            enrichment = self.context_enricher.build_context(
                email_text=email_text,
                subject=subject,
                metadata=metadata,
            )
            
            # Responder todas as perguntas
            qa_results = self.qa_engine.answer_all_questions(
                context=enrichment["context"],
                use_variations=False,
            )
            
            # Processar resultados
            processed_results = {}
            
            for category in QuestionCategory:
                result = qa_results.get(category.value)
                
                if result is None:
                    processed_results[category.value] = {
                        'answer': None,
                        'confidence': 0.0,
                        'question': QAQuestions.get_primary_question(category),
                        'valid': False,
                    }
                else:
                    processed_results[category.value] = {
                        'answer': result.answer,
                        'confidence': result.confidence,
                        'question': result.question,
                        'valid': not AnswerPostProcessor.is_empty_answer(result.answer),
                    }
            processed_results = self.context_enricher.apply_fallbacks(
                processed_results,
                email_text=email_text,
                enrichment=enrichment,
            )
            self.context_enricher.attach_temporal_metadata(
                processed_results,
                enrichment.get("temporal_hints", []),
            )

            # Criar resultado final
            email_result = EmailQAResult(
                email_id=email_id,
                email_text=email_text,
                subject=subject,
                qa_results=processed_results,
                processed_at=datetime.now().isoformat(),
                metadata={
                    'model': self.model_name,
                    'confidence_threshold': self.confidence_threshold,
                    'context_enrichment': {
                        'sender': enrichment.get('sender'),
                        'metadata_location': enrichment.get('metadata_location'),
                        'temporal_hints': enrichment.get('temporal_hints', []),
                        'reference_datetime': (
                            enrichment.get('reference_datetime').isoformat()
                            if enrichment.get('reference_datetime')
                            else None
                        ),
                        'explicit_location': enrichment.get('explicit_location'),
                        'meeting_related': enrichment.get('meeting_related'),
                        'hints': enrichment.get('hints', []),
                    },
                },
            )
            
            if self.verbose:
                self.logger.info(
                    f"Email {email_id} processado com sucesso "
                    f"({len([r for r in processed_results.values() if r['valid']])} respostas válidas)"
                )
            
            return email_result
        
        except Exception as e:
            self.logger.error(f"Erro ao processar email {email_id}: {str(e)}")
            return None
    
    def process_batch(
        self,
        emails: List[Dict[str, Any]],
        show_progress: bool = True,
    ) -> List[EmailQAResult]:
        """
        Processa batch de emails.
        
        Args:
            emails: Lista de emails
                    Formato: [{'id': int, 'text': str, 'subject': str}, ...]
            show_progress: Se True, mostra progresso
            
        Returns:
            Lista de EmailQAResult
        """
        results = []
        
        for i, email in enumerate(emails):
            if show_progress and (i + 1) % 10 == 0:
                self.logger.info(f"Processados {i + 1}/{len(emails)} emails")
            
            result = self.process_email(
                email_text=email.get('text', ''),
                email_id=email.get('id'),
                subject=email.get('subject'),
                metadata={
                    key: value
                    for key, value in email.items()
                    if key not in {'text', 'id', 'subject'}
                },
            )
            
            if result:
                results.append(result)
        
        if show_progress:
            self.logger.info(f"Batch completo: {len(results)}/{len(emails)} emails processados")
        
        return results
    
    def load_gold_annotations(
        self,
        filepath: str,
    ) -> List[Dict[str, Any]]:
        """
        Carrega gold annotations.
        
        Args:
            filepath: Path para ficheiro JSON
            
        Returns:
            Lista de anotações carregadas
        """
        self.logger.info(f"Carregando gold annotations de {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        
        self.logger.info(f"Carregadas {len(annotations)} anotações")
        return annotations
    
    def save_results(
        self,
        results: List[EmailQAResult],
        output_dir: str,
        formats: List[str] = None,
    ) -> None:
        """
        Salva resultados de processamento.
        
        Args:
            results: Lista de resultados
            output_dir: Diretório de output
            formats: Formatos para salvar ('json', 'jsonl', 'csv')
        """
        if formats is None:
            formats = ['json']
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if 'json' in formats:
            output_file = output_dir / 'qa_results.json'
            with open(output_file, 'w', encoding='utf-8') as f:
                data = [result.to_dict() for result in results]
                json.dump(data, f, ensure_ascii=False, indent=2)
            self.logger.info(f"Resultados salvos em {output_file}")
        
        if 'jsonl' in formats:
            output_file = output_dir / 'qa_results.jsonl'
            with open(output_file, 'w', encoding='utf-8') as f:
                for result in results:
                    json.dump(result.to_dict(), f, ensure_ascii=False)
                    f.write('\n')
            self.logger.info(f"Resultados (JSONL) salvos em {output_file}")
        
        if 'csv' in formats:
            import csv
            output_file = output_dir / 'qa_results.csv'
            
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                fieldnames = [
                    'email_id', 'subject', 'participants', 'time',
                    'location', 'topic', 'processed_at'
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for result in results:
                    answers = result.get_answers_only()
                    writer.writerow({
                        'email_id': result.email_id,
                        'subject': result.subject,
                        'participants': answers.get('participants', ''),
                        'time': answers.get('time', ''),
                        'location': answers.get('location', ''),
                        'topic': answers.get('topic', ''),
                        'processed_at': result.processed_at,
                    })
            
            self.logger.info(f"Resultados (CSV) salvos em {output_file}")
    
    def integrate_with_gold_annotations(
        self,
        gold_annotations: List[Dict[str, Any]],
        output_file: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Integra resultados QA com gold annotations.
        
        Args:
            gold_annotations: Lista de gold annotations
            output_file: Ficheiro de output (opcional)
            
        Returns:
            Dicionário com anotações estendidas
        """
        self.logger.info("Integrando resultados QA com gold annotations")
        
        # Processar emails
        qa_results = self.process_batch(gold_annotations)
        
        # Mapear por ID
        qa_map = {r.email_id: r for r in qa_results if r.email_id is not None}
        
        # Integrar
        integrated = []
        for annotation in gold_annotations:
            email_id = annotation.get('id')
            
            integrated_item = annotation.copy()
            
            if email_id in qa_map:
                qa_result = qa_map[email_id]
                answers = qa_result.get_answers_only()
                
                # Adicionar respostas QA
                integrated_item['qa_answers'] = answers
                integrated_item['qa_results'] = qa_result.qa_results
            
            integrated.append(integrated_item)
        
        # Salvar se especificado
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(integrated, f, ensure_ascii=False, indent=2)
            self.logger.info(f"Integração salva em {output_file}")
        
        return {'data': integrated, 'count': len(integrated)}
    
    def print_sample(
        self,
        result: EmailQAResult,
    ) -> None:
        """Imprime amostra formatada de resultado."""
        print("\n" + "="*70)
        print("QA PIPELINE RESULT")
        print("="*70)
        
        if result.subject:
            print(f"\nSubject: {result.subject}")
        
        print(f"\nEmail ({result.email_id}):")
        email_preview = result.email_text[:150] + "..." if len(result.email_text) > 150 else result.email_text
        print(f"  {email_preview}")
        
        print(f"\nAnswers:")
        for category, qa_result in result.qa_results.items():
            answer = qa_result.get('answer')
            confidence = qa_result.get('confidence', 0.0)
            valid = qa_result.get('valid', False)
            
            status = "✓" if valid else "✗"
            print(f"  {status} {category.upper()}:")
            print(f"      Answer: {answer}")
            print(f"      Confidence: {confidence:.2%}")
        
        print("\n" + "="*70)


class QuickQA:
    """Interface rápida para QA (atalho)."""
    
    _pipeline = None
    
    @classmethod
    def init(cls, model_name: str = 'bertimbau-pt', **kwargs):
        """Inicializa pipeline global."""
        cls._pipeline = QAPipeline(model_name=model_name, **kwargs)
    
    @classmethod
    def answer(cls, text: str) -> Dict[str, Optional[str]]:
        """Responde rapidamente todas as perguntas."""
        if cls._pipeline is None:
            cls.init()
        
        result = cls._pipeline.process_email(text)
        return result.get_answers_only() if result else {}


def main():
    """Exemplo de uso do pipeline."""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Inicializar pipeline
    print("Inicializando QA Pipeline...")
    pipeline = QAPipeline(
        model_name='bertimbau-pt',
        device='cpu',
        confidence_threshold=0.3,
        verbose=True,
    )
    
    # Exemplos de emails
    emails = [
        {
            'id': 1,
            'subject': 'Reunião sexta',
            'text': 'Boas Ana, podemos reunir sexta às 15h no Teams para discutir o dataset?',
        },
        {
            'id': 2,
            'subject': 'Adiamento',
            'text': 'Olá, desculpa mas tenho de adiar a reunião de segunda. Terça funciona?',
        },
    ]
    
    # Processar
    print("\nProcessando emails...")
    results = pipeline.process_batch(emails)
    
    # Mostrar resultados
    for result in results:
        pipeline.print_sample(result)
    
    # Salvar
    print("\nSalvando resultados...")
    pipeline.save_results(results, 'qa/output', formats=['json'])
    
    print("\n✓ Pipeline completo!")


if __name__ == "__main__":
    main()
