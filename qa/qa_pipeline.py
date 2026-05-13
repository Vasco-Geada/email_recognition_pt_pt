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
        """
        self.model_name = model_name
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.verbose = verbose
        
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
    ) -> Optional[EmailQAResult]:
        """
        Processa email e responde perguntas.
        
        Args:
            email_text: Texto do email
            email_id: ID do email (opcional)
            subject: Assunto do email (opcional)
            use_cache: Se True, usa cache
            
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
            
            # Responder todas as perguntas
            qa_results = self.qa_engine.answer_all_questions(
                context=email_text,
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
        model_name='multilingual',  # Usar multilíngue para teste rápido
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
