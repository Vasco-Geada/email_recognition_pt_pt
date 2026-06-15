"""
qa_utils.py
===========

Funções utilitárias para o módulo de Question Answering.

Inclui:
- Limpeza e pós-processamento de respostas
- Normalização de texto
- Manipulação de confiança
- Validação de formato
- Logging e debugging

Project: Email Recognition PT-PT
Version: 1.0
"""

import re
import logging
import unicodedata
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass
class QAResult:
    """Estrutura para resultado de QA."""
    
    question: str
    answer: str
    confidence: float
    context: str = ""
    start_logit: float = 0.0
    end_logit: float = 0.0
    
    def is_valid(self, confidence_threshold: float = 0.0) -> bool:
        """Verifica se resultado atende critérios mínimos."""
        return (
            self.confidence >= confidence_threshold
            and len(self.answer.strip()) > 0
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário."""
        return {
            'question': self.question,
            'answer': self.answer,
            'confidence': self.confidence,
            'context': self.context,
            'start_logit': self.start_logit,
            'end_logit': self.end_logit,
        }


class TextNormalizer:
    """Normaliza texto para melhor processamento."""
    
    # Expressões regulares para padrões específicos
    WHITESPACE_RE = re.compile(r'\s+')
    PUNCT_RE = re.compile(r'[^\w\s]', re.UNICODE)
    MULTIPLE_SPACES_RE = re.compile(r' {2,}')
    
    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """Remove whitespace excessivo."""
        return TextNormalizer.WHITESPACE_RE.sub(' ', text).strip()
    
    @staticmethod
    def normalize_unicode(text: str) -> str:
        """Normaliza caracteres Unicode (NFD)."""
        return unicodedata.normalize('NFD', text)
    
    @staticmethod
    def remove_accents(text: str) -> str:
        """Remove acentuação do texto."""
        nfd = unicodedata.normalize('NFD', text)
        return ''.join(c for c in nfd if unicodedata.category(c) != 'Mn')
    
    @staticmethod
    def clean_text(
        text: str,
        lowercase: bool = False,
        remove_accents: bool = False,
        remove_punct: bool = False,
    ) -> str:
        """
        Limpeza completa de texto.
        
        Args:
            text: Texto a limpar
            lowercase: Se True, converte para minúsculas
            remove_accents: Se True, remove acentuação
            remove_punct: Se True, remove pontuação
            
        Returns:
            Texto limpo
        """
        if not text:
            return ""
        
        # Normalizar whitespace
        text = TextNormalizer.normalize_whitespace(text)
        
        # Remover acentos se solicitado
        if remove_accents:
            text = TextNormalizer.remove_accents(text)
        
        # Remover pontuação se solicitado
        if remove_punct:
            text = TextNormalizer.PUNCT_RE.sub('', text)
            text = TextNormalizer.MULTIPLE_SPACES_RE.sub(' ', text)
        
        # Converter para minúsculas se solicitado
        if lowercase:
            text = text.lower()
        
        return TextNormalizer.normalize_whitespace(text)


class AnswerPostProcessor:
    """Pós-processamento de respostas do modelo QA."""
    
    # Palavras que indicam resposta vazia em PT
    EMPTY_INDICATORS = {
        'não', 'nenhum', 'nenhuma', 'nada', 'vazio',
        'sem', 'ausente', 'falta', 'não há', 'não tem',
        'não existe', 'não há nada', 'não tem nada',
    }
    
    @staticmethod
    def clean_answer(answer: str) -> str:
        """
        Limpa resposta removendo pontuação extra e espaços.
        
        Args:
            answer: Resposta bruta
            
        Returns:
            Resposta limpa
        """
        if not answer:
            return ""
        
        answer = answer.strip()
        
        # Remover pontuação trailing
        answer = re.sub(r'[,;:.!?]+$', '', answer)
        
        # Remover espaços múltiplos
        answer = re.sub(r'\s+', ' ', answer)
        
        return answer.strip()
    
    @staticmethod
    def is_empty_answer(answer: str) -> bool:
        """
        Detecta se resposta indica ausência de informação.
        
        Args:
            answer: Resposta a verificar
            
        Returns:
            True se resposta é vazia ou negativa
        """
        if not answer:
            return True
        
        answer_lower = answer.lower().strip()
        
        # Verificar padrões comuns de respostas vazias
        for indicator in AnswerPostProcessor.EMPTY_INDICATORS:
            if answer_lower == indicator or answer_lower.startswith(indicator):
                return True
        
        return False
    
    @staticmethod
    def filter_common_artifacts(answer: str) -> str:
        """
        Remove artefatos comuns em respostas (emojis, assinaturas, etc).
        
        Args:
            answer: Resposta bruta
            
        Returns:
            Resposta filtrada
        """
        # Remover emojis
        answer = re.sub(r'[\U0001F300-\U0001F9FF]+', '', answer)
        
        # Remover menção de "disclaimer" ou similares
        answer = re.sub(r'DISCLAIMER.*', '', answer, flags=re.IGNORECASE)
        answer = re.sub(r'Enviado.*', '', answer, flags=re.IGNORECASE)
        answer = re.sub(r'Sent from.*', '', answer, flags=re.IGNORECASE)
        
        return AnswerPostProcessor.clean_answer(answer)
    
    @staticmethod
    def extract_first_sentence(answer: str) -> str:
        """
        Extrai primeira frase da resposta.
        
        Args:
            answer: Resposta potencialmente multi-sentença
            
        Returns:
            Primeira frase
        """
        if not answer:
            return ""
        
        # Dividir por pontos
        sentences = re.split(r'[.!?]+', answer)
        if sentences:
            first = sentences[0].strip()
            if first:
                return first
        
        return answer.strip()


class ConfidenceScaler:
    """Manipulação de scores de confiança."""
    
    @staticmethod
    def sigmoid(x: float) -> float:
        """Aplica função sigmoid para converter logits em probabilidades."""
        import math
        try:
            return 1.0 / (1.0 + math.exp(-x))
        except OverflowError:
            return 0.0 if x < 0 else 1.0
    
    @staticmethod
    def scale_confidence(
        start_logit: float,
        end_logit: float,
        temperature: float = 1.0,
    ) -> float:
        """
        Calcula confiança combinada a partir de logits.
        
        Args:
            start_logit: Logit de início (do BERT)
            end_logit: Logit de fim (do BERT)
            temperature: Parâmetro de temperatura (default=1.0)
            
        Returns:
            Confiança entre 0 e 1
        """
        # Escalar por temperatura
        scaled_start = start_logit / temperature
        scaled_end = end_logit / temperature
        
        # Aplicar sigmoid
        conf_start = ConfidenceScaler.sigmoid(scaled_start)
        conf_end = ConfidenceScaler.sigmoid(scaled_end)
        
        # Combinar (média geométrica)
        combined = (conf_start * conf_end) ** 0.5
        
        # Clamp a [0, 1]
        return max(0.0, min(1.0, combined))
    
    @staticmethod
    def apply_threshold(
        confidence: float,
        threshold: float,
    ) -> Optional[float]:
        """
        Aplica threshold a confiança.
        
        Args:
            confidence: Confiança calculada
            threshold: Threshold mínimo
            
        Returns:
            Confiança se >= threshold, senão None
        """
        if confidence >= threshold:
            return confidence
        return None


class MetricsCalculator:
    """Cálculo de métricas de avaliação."""
    
    @staticmethod
    def exact_match(predicted: str, reference: str) -> bool:
        """
        Calcula Exact Match (EM).
        
        Args:
            predicted: Resposta predita
            reference: Resposta de referência
            
        Returns:
            True se exato match
        """
        pred_clean = TextNormalizer.clean_text(predicted, lowercase=True)
        ref_clean = TextNormalizer.clean_text(reference, lowercase=True)
        return pred_clean == ref_clean
    
    @staticmethod
    def token_overlap_f1(predicted: str, reference: str) -> float:
        """
        Calcula F1 baseado em overlap de tokens.
        
        Args:
            predicted: Resposta predita
            reference: Resposta de referência
            
        Returns:
            F1 score entre 0 e 1
        """
        pred_tokens = set(predicted.lower().split())
        ref_tokens = set(reference.lower().split())
        
        if len(pred_tokens) == 0 and len(ref_tokens) == 0:
            return 1.0
        
        if len(pred_tokens) == 0 or len(ref_tokens) == 0:
            return 0.0
        
        overlap = len(pred_tokens & ref_tokens)
        
        if overlap == 0:
            return 0.0
        
        precision = overlap / len(pred_tokens)
        recall = overlap / len(ref_tokens)
        f1 = 2 * (precision * recall) / (precision + recall)
        
        return f1

    @staticmethod
    def token_overlap_scores(predicted: str, reference: str) -> Dict[str, float]:
        """
        Calcula precision, recall e F1 por overlap de tokens.

        Casos vazios:
        - ambos vazios: score perfeito
        - apenas um vazio: score zero
        """
        pred_tokens = set(predicted.lower().split()) if predicted else set()
        ref_tokens = set(reference.lower().split()) if reference else set()

        if not pred_tokens and not ref_tokens:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
        if not pred_tokens or not ref_tokens:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        overlap = len(pred_tokens & ref_tokens)
        precision = overlap / len(pred_tokens)
        recall = overlap / len(ref_tokens)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    
    @staticmethod
    def compute_metrics(
        predicted: str,
        reference: str,
    ) -> Dict[str, float]:
        """
        Computa múltiplas métricas de comparação.
        
        Args:
            predicted: Resposta predita
            reference: Resposta de referência
            
        Returns:
            Dicionário com scores
        """
        overlap = MetricsCalculator.token_overlap_scores(predicted, reference)
        return {
            'exact_match': 1.0 if MetricsCalculator.exact_match(predicted, reference) else 0.0,
            'precision': overlap['precision'],
            'recall': overlap['recall'],
            'f1': overlap['f1'],
        }


def setup_logging(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Configura logging para módulo.
    
    Args:
        name: Nome do logger
        level: Nível de logging
        log_file: Ficheiro de output (opcional)
        
    Returns:
        Logger configurado
    """
    logger_obj = logging.getLogger(name)
    logger_obj.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    logger_obj.addHandler(console_handler)
    
    # File handler (se especificado)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger_obj.addHandler(file_handler)
    
    return logger_obj


if __name__ == "__main__":
    """Testes de utilidades."""
    
    # Teste de limpeza
    messy = "  Olá   ana,  vamos  reunir  sexta?  "
    clean = TextNormalizer.clean_text(messy)
    print(f"Messy: '{messy}'")
    print(f"Clean: '{clean}'")
    
    # Teste de resposta vazia
    print("\n--- Teste de Respostas Vazias ---")
    empty_answers = ["", "não", "nada", "não há", "Ana"]
    for ans in empty_answers:
        is_empty = AnswerPostProcessor.is_empty_answer(ans)
        print(f"'{ans}' -> empty: {is_empty}")
    
    # Teste de métricas
    print("\n--- Teste de Métricas ---")
    pred = "sexta às 15h"
    ref = "sexta à tarde"
    metrics = MetricsCalculator.compute_metrics(pred, ref)
    print(f"Predicted: '{pred}'")
    print(f"Reference: '{ref}'")
    print(f"Metrics: {metrics}")
