"""
qa_inference.py
===============

Módulo de inferência de Question Answering com HuggingFace Transformers.

Usa modelos BERTimbau (neuralmind/bert-base-portuguese-cased) ou outros
modelos QA pré-treinados em português.

Características:
- Pipeline de QA com transformers
- Suporte a múltiplos modelos
- Caching de modelos
- Confidence thresholding
- Pós-processamento de respostas
- Logging detalhado

Project: Email Recognition PT-PT
Version: 1.0
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json

import torch
from transformers import (
    pipeline,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    Pipeline,
)

from qa_questions import QAQuestions, QuestionCategory
from qa_utils import (
    QAResult,
    AnswerPostProcessor,
    ConfidenceScaler,
    TextNormalizer,
)


logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class QAModelLoader:
    """Carregamento e caching de modelos QA."""
    
    # Modelos recomendados para português
    RECOMMENDED_MODELS = {
        'bertimbau-pt': 'neuralmind/bert-base-portuguese-cased',
        'bertimbau-pt-large': 'neuralmind/bert-large-portuguese-cased',
        'multilingual': 'bert-base-multilingual-cased',
    }
    
    _model_cache: Dict[str, Any] = {}
    
    @classmethod
    def get_model(
        cls,
        model_name: str,
        use_cache: bool = True,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ) -> Tuple[Any, Any]:
        """
        Carrega modelo e tokenizer com suporte a cache.
        
        Args:
            model_name: Nome do modelo (HuggingFace hub ou recomendado)
            use_cache: Se True, usa cache
            device: Dispositivo ('cuda' ou 'cpu')
            
        Returns:
            (model, tokenizer)
            
        Raises:
            ValueError: Se modelo não encontrado
        """
        # Resolver nome do modelo
        full_model_name = cls.RECOMMENDED_MODELS.get(model_name, model_name)
        
        # Verificar cache
        cache_key = f"{full_model_name}_{device}"
        if use_cache and cache_key in cls._model_cache:
            logger.info(f"Usando modelo em cache: {cache_key}")
            return cls._model_cache[cache_key]
        
        # Carregar modelo e tokenizer
        logger.info(f"Carregando modelo: {full_model_name} no dispositivo {device}")
        
        try:
            model = AutoModelForQuestionAnswering.from_pretrained(full_model_name)
            tokenizer = AutoTokenizer.from_pretrained(full_model_name)
            
            model.to(device)
            model.eval()
            
            # Guardar em cache
            if use_cache:
                cls._model_cache[cache_key] = (model, tokenizer)
            
            logger.info(f"Modelo carregado com sucesso: {full_model_name}")
            return model, tokenizer
        
        except Exception as e:
            logger.error(f"Erro ao carregar modelo {full_model_name}: {str(e)}")
            raise ValueError(f"Modelo não encontrado: {full_model_name}")
    
    @classmethod
    def clear_cache(cls) -> None:
        """Limpa cache de modelos."""
        cls._model_cache.clear()
        logger.info("Cache de modelos limpo")


class QAInferenceEngine:
    """
    Motor de inferência para Question Answering.
    
    Características:
    - Usa HuggingFace transformers pipeline
    - Suporta múltiplos modelos
    - Pós-processamento de respostas
    - Aplicação de confiança
    - Logging e debug
    """
    
    def __init__(
        self,
        model_name: str = 'bertimbau-pt',
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        confidence_threshold: float = 0.5,
        temperature: float = 1.0,
        top_k: int = 1,
        max_answer_len: int = 128,
        use_cache: bool = True,
    ):
        """
        Inicializa engine de QA.
        
        Args:
            model_name: Nome do modelo (ou recomendado)
            device: Dispositivo ('cuda' ou 'cpu')
            confidence_threshold: Threshold mínimo de confiança (0-1)
            temperature: Parâmetro de temperatura para softmax
            top_k: Número de top respostas a retornar
            max_answer_len: Comprimento máximo da resposta
            use_cache: Se True, usa cache de modelos
        """
        self.model_name = model_name
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.temperature = temperature
        self.top_k = top_k
        self.max_answer_len = max_answer_len
        
        logger.info(f"Inicializando QA Engine com modelo: {model_name}")
        
        # Carregar modelo
        try:
            model, tokenizer = QAModelLoader.get_model(
                model_name,
                use_cache=use_cache,
                device=device,
            )
            
            # Criar pipeline
            self.pipeline = pipeline(
                'question-answering',
                model=model,
                tokenizer=tokenizer,
                device=0 if device == 'cuda' else -1,
            )
            
            logger.info("QA Engine inicializado com sucesso")
        
        except Exception as e:
            logger.error(f"Erro ao inicializar QA Engine: {str(e)}")
            raise
    
    def answer_question(
        self,
        question: str,
        context: str,
        apply_postprocessing: bool = True,
    ) -> Optional[QAResult]:
        """
        Responde pergunta sobre contexto.
        
        Args:
            question: Pergunta em português
            context: Contexto (email body)
            apply_postprocessing: Se True, aplica pós-processamento
            
        Returns:
            QAResult com resposta e confiança, ou None se threshold não atingido
        """
        if not question or not context:
            logger.warning("Pergunta ou contexto vazios")
            return None
        
        try:
            # Inferência
            result = self.pipeline(
                question=question,
                context=context,
                top_k=self.top_k,
                max_answer_len=self.max_answer_len,
            )
            
            # Se resultado é lista, pegar primeiro
            if isinstance(result, list):
                result = result[0]
            
            # Extrair componentes
            answer = result.get('answer', '')
            score = result.get('score', 0.0)
            
            # Calcular confiança
            confidence = ConfidenceScaler.scale_confidence(
                start_logit=result.get('start', 0.0),
                end_logit=result.get('end', 0.0),
                temperature=self.temperature,
            )
            
            # Usar score do pipeline se disponível
            if 'score' in result:
                confidence = result['score']
            
            # Pós-processamento
            if apply_postprocessing:
                answer = AnswerPostProcessor.filter_common_artifacts(answer)
                answer = AnswerPostProcessor.clean_answer(answer)
            
            # Criar resultado
            qa_result = QAResult(
                question=question,
                answer=answer,
                confidence=confidence,
                context=context[:100],  # Guardar contexto truncado
                start_logit=result.get('start', 0.0),
                end_logit=result.get('end', 0.0),
            )
            
            # Verificar threshold
            if confidence < self.confidence_threshold:
                logger.debug(
                    f"Confiança baixa ({confidence:.2f}) para pergunta: {question}"
                )
                return None
            
            return qa_result
        
        except Exception as e:
            logger.error(f"Erro na inferência QA: {str(e)}")
            return None
    
    def answer_all_questions(
        self,
        context: str,
        use_variations: bool = False,
    ) -> Dict[str, Optional[QAResult]]:
        """
        Responde todas as perguntas estruturadas sobre um contexto.
        
        Args:
            context: Contexto (email)
            use_variations: Se True, tenta variações de perguntas
            
        Returns:
            Dicionário mapeando categoria -> QAResult
        """
        results = {}
        
        for category in QuestionCategory:
            if use_variations:
                # Tentar pergunta primária e variações
                question = QAQuestions.get_primary_question(category)
                result = self.answer_question(question, context)
                
                # Se não conseguir resposta, tentar variações
                if result is None or AnswerPostProcessor.is_empty_answer(result.answer):
                    question = QAQuestions.get_random_variation(category)
                    result = self.answer_question(question, context)
            else:
                # Apenas pergunta primária
                question = QAQuestions.get_primary_question(category)
                result = self.answer_question(question, context)
            
            results[category.value] = result
        
        return results
    
    def batch_answer_questions(
        self,
        contexts: List[str],
        category: Optional[QuestionCategory] = None,
    ) -> List[Dict[str, Optional[QAResult]]]:
        """
        Processa batch de contextos.
        
        Args:
            contexts: Lista de emails/contextos
            category: Se especificado, apenas responder essa categoria
            
        Returns:
            Lista de dicionários com resultados
        """
        all_results = []
        
        for i, context in enumerate(contexts):
            if (i + 1) % 10 == 0:
                logger.info(f"Processados {i + 1}/{len(contexts)} contextos")
            
            if category:
                question = QAQuestions.get_primary_question(category)
                result = self.answer_question(question, context)
                all_results.append({category.value: result})
            else:
                results = self.answer_all_questions(context)
                all_results.append(results)
        
        logger.info(f"Batch processing completo: {len(contexts)} contextos")
        return all_results


class MultilingualQAFallback:
    """
    Fallback para QA multilíngue se modelo português não disponível.
    
    Usa bert-base-multilingual-cased como fallback.
    """
    
    def __init__(self):
        """Inicializa fallback."""
        self.engine = None
        self.available = False
        
        try:
            self.engine = QAInferenceEngine(
                model_name='multilingual',
                confidence_threshold=0.4,  # Threshold mais baixo
            )
            self.available = True
            logger.warning("Usando modelo multilíngue como fallback para QA português")
        except Exception as e:
            logger.error(f"Erro ao inicializar fallback multilíngue: {str(e)}")
    
    def answer_question(
        self,
        question: str,
        context: str,
    ) -> Optional[QAResult]:
        """Responde usando modelo multilíngue."""
        if not self.available:
            return None
        
        return self.engine.answer_question(question, context)


class QAResultsCache:
    """Cache simples para resultados de QA."""
    
    def __init__(self, max_size: int = 1000):
        """
        Inicializa cache.
        
        Args:
            max_size: Tamanho máximo de cache
        """
        self.cache: Dict[str, QAResult] = {}
        self.max_size = max_size
    
    def _make_key(self, question: str, context: str) -> str:
        """Cria chave de cache."""
        import hashlib
        key = f"{question}|{context[:100]}"
        return hashlib.md5(key.encode()).hexdigest()
    
    def get(self, question: str, context: str) -> Optional[QAResult]:
        """Obtém resultado do cache."""
        key = self._make_key(question, context)
        return self.cache.get(key)
    
    def set(self, question: str, context: str, result: QAResult) -> None:
        """Adiciona resultado ao cache."""
        if len(self.cache) >= self.max_size:
            # Remover elemento aleatório para fazer espaço
            import random
            random_key = random.choice(list(self.cache.keys()))
            del self.cache[random_key]
        
        key = self._make_key(question, context)
        self.cache[key] = result
    
    def clear(self) -> None:
        """Limpa cache."""
        self.cache.clear()


def main():
    """Exemplo de uso do motor de QA."""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Inicializar engine
    print("Inicializando QA Engine...")
    try:
        engine = QAInferenceEngine(
            model_name='bertimbau-pt',
            confidence_threshold=0.5,
        )
    except Exception as e:
        print(f"Erro: {e}")
        print("Tentando com modelo multilíngue...")
        engine = QAInferenceEngine(
            model_name='multilingual',
            confidence_threshold=0.4,
        )
    
    # Exemplo de email
    context = """
    Boas Ana, podemos reunir sexta às 15h no Teams para discutir o dataset?
    """
    
    print(f"\nContexto: {context.strip()}")
    print("\n" + "="*60)
    print("Respondendo perguntas...")
    print("="*60)
    
    # Responder a todas as perguntas
    results = engine.answer_all_questions(context.strip())
    
    for category_name, result in results.items():
        print(f"\n[{category_name.upper()}]")
        if result:
            print(f"  Pergunta: {result.question}")
            print(f"  Resposta: {result.answer}")
            print(f"  Confiança: {result.confidence:.2%}")
        else:
            print(f"  Sem resposta (confiança baixa)")


if __name__ == "__main__":
    main()
