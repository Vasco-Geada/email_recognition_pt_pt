"""
heuristic_extractors.py
========================

Módulo de extração heurística de informações de emails para gold annotations.

Contém extractores baseados em regex e heurísticas para:
- Trigger (verbos relacionados com reunião)
- Participantes (nomes próprios, menções)
- Expressões temporais (datas, horas, expressões relativas)
- Localização (plataformas, salas, espaços)
- Tópicos académicos

Author: Generated for Email Recognition PT-PT Project
"""

import re
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ExtractionResult:
    """Resultado de uma extração com confidence score"""
    values: List[str] = field(default_factory=list)
    confidence: float = 1.0
    metadata: Dict = field(default_factory=dict)


class TriggerExtractor:
    """
    Extrai o verbo/trigger principal relacionado com reunião.
    
    Triggers suportados: reunir, marcar, cancelar, confirmar, combinar, 
    agendar, aparecer, falar, encontrar, conversar, discutir
    """
    
    # Mapa de triggers com variações e expressões regulares
    TRIGGER_PATTERNS = {
        'reunir': [
            r'\breunir(?:mos|em|ão|ia)?(?:\s+(?:com|a))?',
            r'\breunião\b',
            r'\b(?:uma|uma\s+)?reunião\b',
        ],
        'marcar': [
            r'\bmarcar(?:\s+(?:uma|a|o))?(?:\s+reunião)?',
            r'\bmarquei\b',
            r'\bmarca(?:do|da)?\b',
        ],
        'cancelar': [
            r'\bcancelar(?:\s+(?:a|o|uma))?(?:\s+reunião)?',
            r'\bcancelado\b',
            r'\bcancelamento\b',
            r'\bfaltar(?:\s+à|ao)?(?:\s+reunião)?',
        ],
        'confirmar': [
            r'\bconfirmar(?:\s+(?:a|o|que|se))?',
            r'\bconfirmado\b',
            r'\bconfirmação\b',
        ],
        'agendar': [
            r'\bagendar',
            r'\bagendado\b',
        ],
        'combinar': [
            r'\bcombinar',
            r'\bcombinado\b',
        ],
        'encontrar': [
            r'\bencontrar(?:\s+(?:com|a|se))?',
            r'\bencontro\b',
        ],
        'falar': [
            r'\bfalar(?:\s+(?:com|sobre))?',
            r'\bconversa(?:r|ção)?\b',
        ],
        'discutir': [
            r'\bdiscutir',
            r'\bdiscussão\b',
        ],
    }
    
    def extract(self, text: str) -> ExtractionResult:
        """
        Extrai o trigger principal do texto.
        
        Args:
            text: Texto do email
            
        Returns:
            ExtractionResult com o trigger encontrado
        """
        text_lower = text.lower()
        found_triggers = []
        
        for trigger, patterns in self.TRIGGER_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    found_triggers.append(trigger)
                    break
        
        # Remove duplicados mantendo ordem
        unique_triggers = []
        for t in found_triggers:
            if t not in unique_triggers:
                unique_triggers.append(t)
        
        confidence = 0.9 if unique_triggers else 0.0
        
        return ExtractionResult(
            values=unique_triggers[:1] if unique_triggers else [],
            confidence=confidence,
            metadata={'all_triggers': unique_triggers}
        )


class ParticipantExtractor:
    """
    Extrai participantes mencionados no email.
    
    Detecta:
    - Nomes próprios em maiúscula (Ana, Silva)
    - Títulos (Professor, Dr., Eng.)
    - Pronomes/menções diretas
    """
    
    # Padrões para salutações e menções
    SALUTATION_PATTERNS = [
        r'(?:Olá|Oi|Caro|Cara|Prezado|Prezada|Exmo|Exma|Boas)\s+([A-Z][a-zá-ú]+(?:\s+[A-Z][a-zá-ú]+)?)',
        r'(?:com|a|para|pelo|pela)\s+(?:Professor|Professora|Dr\.|Dra\.|Eng\.|Enga\.|Prof\.)\s+([A-Z][a-zá-ú]+(?:\s+[A-Z][a-zá-ú]+)?)',
        r'@([a-zA-Z][a-zA-Z0-9]*)',  # Menções @nome
    ]
    
    # Títulos que precedem nomes
    TITLES = [
        'Professor', 'Professora', 'Prof.', 'Dr.', 'Dra.',
        'Eng.', 'Engenheiro', 'Engenheira', 'Exmo', 'Exma',
        'Sr.', 'Sra.', 'Senhor', 'Senhora'
    ]
    
    def extract(self, text: str) -> ExtractionResult:
        """
        Extrai participantes do texto.
        
        Args:
            text: Texto do email
            
        Returns:
            ExtractionResult com participantes encontrados
        """
        participants = set()
        
        # Buscar por salutações e menções
        for pattern in self.SALUTATION_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                name = match.group(1).strip()
                if name and len(name) > 1:  # Filtra nomes muito curtos
                    participants.add(name)
        
        # Buscar nomes próprios em contextos de menção (precedidos de verbos comuns)
        name_patterns = [
            r'(?:com|a|para|junto de)\s+([A-Z][a-zá-ú]+(?:\s+[A-Z][a-zá-ú]+)?)',
            r'(?:você|tu|ele|ela|eles|elas|nós|a|o)\s+([A-Z][a-zá-ú]+)',
        ]
        
        for pattern in name_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                name = match.group(1).strip()
                if name and len(name) > 2:
                    participants.add(name)
        
        # Remover títulos isolados
        participants = {p for p in participants if not p.split()[0] in self.TITLES}
        
        confidence = 0.75 if participants else 0.0
        
        return ExtractionResult(
            values=sorted(list(participants)),
            confidence=confidence,
            metadata={'extraction_method': 'salutation_and_proper_nouns'}
        )


class TemporalExtractor:
    """
    Extrai expressões temporais do email.
    
    Detecta:
    - Dias relativos (amanhã, hoje, sexta)
    - Horas (15h, às 15:00)
    - Períodos (morning, afternoon, depois de almoço)
    - Datas relativas (próxima semana, semana que vem)
    """
    
    # Padrões temporais estruturados
    TEMPORAL_PATTERNS = {
        'dias_relativos': [
            r'\b(?:amanhã|hoje|ontem|agora)\b',
            r'\b(?:segunda|terça|quarta|quinta|sexta|sábado|domingo)(?:-feira)?\b',
            r'\b(?:próximo|próxima|seguinte)?\s*(?:segunda|terça|quarta|quinta|sexta|sábado|domingo)(?:-feira)?\b',
        ],
        'semanas': [
            r'\b(?:esta|próxima|semana\s+que\s+vem)\s+semana\b',
            r'\b(?:na|para\s+a)\s+(?:próxima\s+)?semana\b',
        ],
        'horas': [
            r'\b(?:às|ao|a(?:s)?)\s*(\d{1,2})(?:\s*[:h]\s*)?(\d{2})?\s*(?:h|horas)?\b',
            r'(\d{1,2}):(\d{2})',
            r'\b(?:meio-dia|meia-noite)\b',
        ],
        'periodos': [
            r'\b(?:de\s+)?(?:manhã|manha)\b',
            r'\b(?:à\s+)?(?:tarde|à tarde)\b',
            r'\b(?:à\s+)?(?:noite|à noite)\b',
            r'\b(?:depois\s+de\s+)?almoço\b',
            r'\b(?:depois\s+de\s+)?intervalo\b',
        ],
        'expressoes_relativas': [
            r'\b(?:mais\s+)?logo\b',
            r'\b(?:em\s+breve|brevemente)\b',
            r'\b(?:ao\s+fim\s+da\s+tarde|final\s+da\s+tarde)\b',
        ],
    }
    
    def extract(self, text: str) -> ExtractionResult:
        """
        Extrai expressões temporais do texto.
        
        Args:
            text: Texto do email
            
        Returns:
            ExtractionResult com expressões temporais encontradas
        """
        temporal_expressions = []
        
        # Buscar por cada tipo de padrão temporal
        for category, patterns in self.TEMPORAL_PATTERNS.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    temporal_expressions.append(match.group(0).strip())
        
        # Remove duplicados mantendo ordem
        temporal_expressions = list(dict.fromkeys(temporal_expressions))
        
        confidence = 0.8 if temporal_expressions else 0.0
        
        return ExtractionResult(
            values=temporal_expressions,
            confidence=confidence,
            metadata={'temporal_categories': list(self.TEMPORAL_PATTERNS.keys())}
        )


class LocationExtractor:
    """
    Extrai localizações mencionadas no email.
    
    Detecta:
    - Plataformas online (Teams, Zoom, Discord)
    - Salas físicas (sala 2.3, auditório)
    - Edifícios e espaços (biblioteca, laboratório)
    """
    
    # Padrões de localização
    LOCATION_PATTERNS = {
        'plataformas_online': [
            r'\b(?:MS\s+)?Teams\b',
            r'\bZoom\b',
            r'\bDiscord\b',
            r'\bSkype\b',
            r'\bGoogle\s+Meet\b',
            r'\bMicrosoft\s+Teams\b',
        ],
        'salas_fisicas': [
            r'\bsala\s+(?:\d+[.-]?\d*|[A-Z]\d+)',
            r'\b(?:auditório|anfiteatro|sala\s+de\s+aula)\b',
            r'\baula\s+\d+',
        ],
        'espacos': [
            r'\bbiblioteca\b',
            r'\blaboratório\b',
            r'\blab\b',
            r'\boficina\b',
            r'\bcafetaria\b',
            r'\bcantina\b',
        ],
        'edificios': [
            r'\b(?:edifício|bloco)\s+[A-Z0-9]+',
            r'\bDepartamento\b',
        ],
    }
    
    def extract(self, text: str) -> ExtractionResult:
        """
        Extrai localizações do texto.
        
        Args:
            text: Texto do email
            
        Returns:
            ExtractionResult com localizações encontradas
        """
        locations = []
        
        for category, patterns in self.LOCATION_PATTERNS.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    location = match.group(0).strip()
                    locations.append(location)
        
        # Remove duplicados mantendo ordem
        locations = list(dict.fromkeys(locations))
        
        confidence = 0.85 if locations else 0.0
        
        return ExtractionResult(
            values=locations,
            confidence=confidence,
            metadata={'location_categories': list(self.LOCATION_PATTERNS.keys())}
        )


class TopicExtractor:
    """
    Extrai tópicos académicos mencionados no email.
    
    Detecta:
    - Trabalhos académicos (dissertação, tese, relatório)
    - Conceitos técnicos (BERT, pipeline, métricas)
    - Disciplinas e assuntos
    """
    
    # Padrões de tópicos académicos
    TOPIC_PATTERNS = {
        'trabalhos_academicos': [
            r'\bdissertação\b',
            r'\btese\b',
            r'\brelatório\b',
            r'\b(?:trabalho|projeto|assignment)\b',
            r'\bartigo\b',
            r'\bpaper\b',
        ],
        'conceitos_tecn': [
            r'\b(?:pipeline|NLP|IA|AI|machine\s+learning|deep\s+learning)\b',
            r'\b(?:BERT|GPT|RNN|LSTM|CNN)\b',
            r'\b(?:dataset|corpus|embeddings?)\b',
            r'\b(?:precisão|recall|F1|accuracy|métrica)\b',
            r'\b(?:classificação|segmentação|extração|análise)\b',
        ],
        'disciplinas': [
            r'\b(?:linguística|processamento|computacional)\b',
            r'\b(?:inteligência\s+artificial|artificial\s+intelligence)\b',
        ],
    }
    
    def extract(self, text: str) -> ExtractionResult:
        """
        Extrai tópicos académicos do texto.
        
        Args:
            text: Texto do email
            
        Returns:
            ExtractionResult com tópicos encontrados
        """
        topics = []
        
        for category, patterns in self.TOPIC_PATTERNS.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    topic = match.group(0).strip()
                    topics.append(topic)
        
        # Remove duplicados mantendo ordem
        topics = list(dict.fromkeys(topics))
        
        confidence = 0.7 if topics else 0.0
        
        return ExtractionResult(
            values=topics,
            confidence=confidence,
            metadata={'topic_categories': list(self.TOPIC_PATTERNS.keys())}
        )


class HeuristicAnnotationExtractor:
    """
    Extractor principal que orquestra todos os extractores.
    
    Combina resultados de diferentes extractores para gerar
    uma anotação completa.
    """
    
    def __init__(self):
        """Inicializa todos os extractores"""
        self.trigger_extractor = TriggerExtractor()
        self.participant_extractor = ParticipantExtractor()
        self.temporal_extractor = TemporalExtractor()
        self.location_extractor = LocationExtractor()
        self.topic_extractor = TopicExtractor()
    
    def extract_all(self, text: str, subject: str = "") -> Dict:
        """
        Extrai todas as informações do email.
        
        Args:
            text: Corpo do email
            subject: Assunto do email (opcional, para contexto adicional)
            
        Returns:
            Dict com todas as informações extraídas
        """
        full_text = f"{subject}\n{text}".strip()
        
        return {
            'trigger': self.trigger_extractor.extract(full_text),
            'participants': self.participant_extractor.extract(full_text),
            'temporal': self.temporal_extractor.extract(full_text),
            'location': self.location_extractor.extract(full_text),
            'topic': self.topic_extractor.extract(full_text),
        }
