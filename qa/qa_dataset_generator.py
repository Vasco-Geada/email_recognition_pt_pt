"""
qa_dataset_generator.py
=======================

Gerador de dataset para Question Answering a partir de gold annotations.

Converte gold annotations (format: argumentos estruturados) para formato
compatível com modelos de QA (SQuAD-style):
    context -> email completo
    question -> pergunta estruturada
    answers -> resposta + span character positions

Suporta:
- Carregamento de gold annotations
- Conversão para formato QA
- Geração de variações de perguntas
- Validação de dataset
- Export para diferentes formatos (JSON, SQuAD, HuggingFace)

Project: Email Recognition PT-PT
Version: 1.0
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict, field
from datetime import datetime

from qa_questions import QAQuestions, QuestionCategory


logger = logging.getLogger(__name__)


@dataclass
class QAExample:
    """Exemplo de QA no formato SQuAD."""
    
    context: str                  # Email body (texto completo)
    question: str                 # Pergunta
    question_category: str        # Categoria (participants, time, location, topic)
    answers: List[str]           # Lista de respostas possíveis
    answer_spans: List[Dict] = field(default_factory=list)  # {start, end}
    id: Optional[str] = None
    email_id: Optional[int] = None
    subject: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário."""
        return asdict(self)
    
    def to_squad_format(self) -> Dict[str, Any]:
        """
        Converte para formato SQuAD.
        
        Returns:
            {
                'id': str,
                'question': str,
                'context': str,
                'answers': {
                    'text': [str],
                    'answer_start': [int]
                }
            }
        """
        # Calcular posições no contexto
        answer_starts = []
        for answer in self.answers:
            start = self.context.find(answer)
            if start != -1:
                answer_starts.append(start)
            else:
                # Se não encontrar, usar -1 como sentinel
                answer_starts.append(-1)
        
        return {
            'id': self.id or str(hash(self.question + self.context)),
            'question': self.question,
            'context': self.context,
            'answers': {
                'text': self.answers,
                'answer_start': answer_starts
            }
        }


@dataclass
class QADataset:
    """Dataset completo de QA."""
    
    examples: List[QAExample] = field(default_factory=list)
    version: str = "1.0"
    created_at: str = ""
    source: str = "gold_annotations"
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
    
    def add_example(self, example: QAExample) -> None:
        """Adiciona exemplo ao dataset."""
        self.examples.append(example)
    
    def filter_by_category(self, category: str) -> 'QADataset':
        """Filtra exemplos por categoria."""
        filtered = QADataset(
            version=self.version,
            created_at=self.created_at,
            source=self.source,
        )
        filtered.examples = [
            ex for ex in self.examples
            if ex.question_category == category
        ]
        return filtered
    
    def split(self, train_ratio: float = 0.8) -> Tuple['QADataset', 'QADataset']:
        """
        Divide dataset em treino/teste.
        
        Args:
            train_ratio: Proporção para treino (default=0.8)
            
        Returns:
            (train_dataset, test_dataset)
        """
        n_train = int(len(self.examples) * train_ratio)
        
        train = QADataset(
            examples=self.examples[:n_train],
            version=self.version,
            source=self.source,
        )
        test = QADataset(
            examples=self.examples[n_train:],
            version=self.version,
            source=self.source,
        )
        
        return train, test
    
    def to_squad_format(self) -> Dict[str, Any]:
        """
        Converte dataset para formato SQuAD.
        
        Returns:
            {
                'version': str,
                'data': [
                    {
                        'paragraphs': [
                            {
                                'context': str,
                                'qas': [...]
                            }
                        ]
                    }
                ]
            }
        """
        # Agrupar exemplos por contexto
        contexts_map: Dict[str, List[QAExample]] = {}
        for example in self.examples:
            if example.context not in contexts_map:
                contexts_map[example.context] = []
            contexts_map[example.context].append(example)
        
        # Converter para formato SQuAD
        squad_data = []
        for context, examples in contexts_map.items():
            qas = [ex.to_squad_format() for ex in examples]
            article = {
                'paragraphs': [
                    {
                        'context': context,
                        'qas': [
                            {
                                'id': ex.id or str(hash(ex.question)),
                                'question': ex.question,
                                'answers': {
                                    'text': ex.answers,
                                    'answer_start': [
                                        context.find(ans) if context.find(ans) != -1 else -1
                                        for ans in ex.answers
                                    ]
                                }
                            }
                            for ex in examples
                        ]
                    }
                ]
            }
            squad_data.append(article)
        
        return {
            'version': self.version,
            'data': squad_data
        }
    
    def save_json(self, filepath: str) -> None:
        """Salva dataset em JSON."""
        data = {
            'version': self.version,
            'created_at': self.created_at,
            'source': self.source,
            'examples': [ex.to_dict() for ex in self.examples]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Dataset salvo em {filepath}")
    
    def save_squad_format(self, filepath: str) -> None:
        """Salva dataset em formato SQuAD."""
        squad_data = self.to_squad_format()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(squad_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Dataset (formato SQuAD) salvo em {filepath}")
    
    @staticmethod
        def load_json(filepath: str) -> 'QADataset':
        """Carrega dataset de JSON."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        dataset = QADataset(
            version=data.get('version', '1.0'),
            created_at=data.get('created_at', ''),
            source=data.get('source', 'gold_annotations'),
        )
        
        for ex_data in data.get('examples', []):
            example = QAExample(
                context=ex_data['context'],
                question=ex_data['question'],
                question_category=ex_data['question_category'],
                answers=ex_data['answers'],
                id=ex_data.get('id'),
                email_id=ex_data.get('email_id'),
                subject=ex_data.get('subject'),
                metadata=ex_data.get('metadata', {}),
            )
            dataset.add_example(example)
        
        logger.info(f"Dataset carregado de {filepath} ({len(dataset.examples)} exemplos)")
        return dataset


class QADatasetGenerator:
    """
    Gerador de dataset QA a partir de gold annotations.
    
    Pipeline:
    1. Carregar gold annotations
    2. Para cada email:
       - Para cada categoria (participants, time, location, topic):
         - Se há respostas nessa categoria:
           - Gerar exemplos QA
           - Incluir variações de perguntas
    3. Validar dataset
    4. Salvar em formatos múltiplos
    """
    
    def __init__(self, include_question_variations: bool = True):
        """
        Inicializa gerador.
        
        Args:
            include_question_variations: Se True, cria exemplos com variações
        """
        self.include_question_variations = include_question_variations
        self.dataset = QADataset()
        self.errors = []
    
    def load_gold_annotations(self, filepath: str) -> None:
        """
        Carrega gold annotations em JSON.
        
        Espera formato:
        [
            {
                "id": int,
                "text": str,
                "subject": str (opcional),
                "arguments": {
                    "participants": [str],
                    "time": [str],
                    "location": [str],
                    "topic": [str]
                }
            }
        ]
        
        Args:
            filepath: Path para ficheiro JSON
        """
        logger.info(f"Carregando gold annotations de {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        
        logger.info(f"Loaded {len(annotations)} annotations")
        
        for annotation in annotations:
            self._process_annotation(annotation)
    
    def _process_annotation(self, annotation: Dict[str, Any]) -> None:
        """Processa uma anotação individual."""
        email_id = annotation.get('id')
        email_text = annotation.get('text', '')
        email_subject = annotation.get('subject', '')
        arguments = annotation.get('arguments', {})
        
        if not email_text:
            logger.warning(f"Email {email_id} sem texto")
            return
        
        # Processar cada categoria
        for category in QuestionCategory:
            category_name = category.value
            answers = arguments.get(category_name, [])
            
            if not answers or len(answers) == 0:
                continue
            
            # Criar exemplo QA com pergunta primária
            self._create_qa_example(
                context=email_text,
                email_id=email_id,
                subject=email_subject,
                category=category,
                answers=answers,
            )
            
            # Criar exemplos com variações de perguntas (se habilitado)
            if self.include_question_variations:
                question_obj = QAQuestions.get_question(category)
                for variation in question_obj.variations[:2]:  # Limitar a 2 variações
                    self._create_qa_example(
                        context=email_text,
                        email_id=email_id,
                        subject=email_subject,
                        category=category,
                        answers=answers,
                        question_override=variation,
                    )
    
    def _create_qa_example(
        self,
        context: str,
        email_id: Optional[int],
        subject: Optional[str],
        category: QuestionCategory,
        answers: List[str],
        question_override: Optional[str] = None,
    ) -> None:
        """Cria exemplo QA individual."""
        # Obter pergunta
        if question_override:
            question = question_override
        else:
            question = QAQuestions.get_primary_question(category)
        
        # Filtrar respostas vazias
        valid_answers = [ans for ans in answers if ans and len(ans.strip()) > 0]
        
        if not valid_answers:
            return
        
        # Criar exemplo
        example = QAExample(
            context=context,
            question=question,
            question_category=category.value,
            answers=valid_answers,
            email_id=email_id,
            subject=subject,
            id=f"{email_id}_{category.value}_{len(self.dataset.examples)}",
            metadata={
                'source': 'gold_annotations',
                'email_id': email_id,
                'subject': subject,
            }
        )
        
        self.dataset.add_example(example)
    
    def validate_dataset(self) -> Dict[str, Any]:
        """
        Valida dataset gerado.
        
        Returns:
            Estatísticas de validação
        """
        stats = {
            'total_examples': len(self.dataset.examples),
            'by_category': {},
            'invalid_examples': [],
        }
        
        # Contar por categoria
        for category in QuestionCategory:
            count = len([
                ex for ex in self.dataset.examples
                if ex.question_category == category.value
            ])
            stats['by_category'][category.value] = count
        
        # Detectar exemplos inválidos
        for i, example in enumerate(self.dataset.examples):
            if not example.context or not example.question:
                stats['invalid_examples'].append(i)
            if not example.answers or len(example.answers) == 0:
                stats['invalid_examples'].append(i)
        
        logger.info(f"Dataset validation: {stats}")
        return stats
    
    def save_dataset(
        self,
        output_dir: str,
        formats: List[str] = None,
    ) -> None:
        """
        Salva dataset em múltiplos formatos.
        
        Args:
            output_dir: Diretório de output
            formats: Lista de formatos ('json', 'squad')
        """
        if formats is None:
            formats = ['json', 'squad']
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if 'json' in formats:
            self.dataset.save_json(str(output_dir / 'qa_dataset.json'))
        
        if 'squad' in formats:
            self.dataset.save_squad_format(str(output_dir / 'qa_dataset_squad.json'))
        
        logger.info(f"Dataset salvo em {output_dir}")
    
    def print_statistics(self) -> None:
        """Imprime estatísticas do dataset."""
        print("\n" + "="*60)
        print("QA DATASET STATISTICS")
        print("="*60)
        
        stats = self.validate_dataset()
        
        print(f"\nTotal Examples: {stats['total_examples']}")
        print("\nExamples by Category:")
        for category, count in stats['by_category'].items():
            print(f"  {category}: {count}")
        
        print(f"\nInvalid Examples: {len(stats['invalid_examples'])}")
        
        if self.dataset.examples:
            print("\nExample (first):")
            ex = self.dataset.examples[0]
            print(f"  Context: {ex.context[:100]}...")
            print(f"  Question: {ex.question}")
            print(f"  Answers: {ex.answers}")


def main():
    """Exemplo de uso do gerador."""
    import logging
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Criar gerador
    generator = QADatasetGenerator(include_question_variations=True)
    
    # Carregar gold annotations
    gold_path = "gold_annotations/output/gold.json"
    try:
        generator.load_gold_annotations(gold_path)
    except FileNotFoundError:
        logger.error(f"Gold annotations não encontradas em {gold_path}")
        return
    
    # Validar e imprimir estatísticas
    generator.print_statistics()
    
    # Salvar dataset
    output_dir = "qa/output"
    generator.save_dataset(output_dir, formats=['json', 'squad'])
    
    print(f"\n✓ Dataset gerado com sucesso!")


if __name__ == "__main__":
    main()
