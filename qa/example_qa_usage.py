"""
example_qa_usage.py
===================

Exemplos completos de uso do módulo QA.

Demonstra:
1. Inicialização do pipeline
2. Processamento de emails individuais
3. Batch processing
4. Integração com gold annotations
5. Avaliação de resultados
6. Exportação de resultados


Project: Email Recognition PT-PT
Version: 1.0
"""

import json
from pathlib import Path
from typing import List, Dict

# Imports do módulo QA
from qa_pipeline import QAPipeline, QuickQA
from qa_dataset_generator import QADatasetGenerator
from qa_evaluator import QAEvaluator
from qa_questions import QAQuestions


def example_1_quick_usage():
    """Exemplo 1: Uso rápido e simples."""
    print("\n" + "="*70)
    print("EXEMPLO 1: Quick Usage")
    print("="*70)
    
    # Interface rápida
    QuickQA.init(model_name='multilingual', device='cpu')
    
    email = "Boas Ana, podemos reunir sexta às 15h no Teams?"
    answers = QuickQA.answer(email)
    
    print(f"\nEmail: {email}")
    print(f"\nAnswers:")
    for category, answer in answers.items():
        print(f"  {category}: {answer}")


def example_2_single_email():
    """Exemplo 2: Processar email individual."""
    print("\n" + "="*70)
    print("EXEMPLO 2: Single Email Processing")
    print("="*70)
    
    # Inicializar pipeline
    pipeline = QAPipeline(
        model_name='multilingual',
        device='cpu',
        confidence_threshold=0.5,
        verbose=True,
    )
    
    # Email de teste
    email = {
        'id': 1,
        'subject': 'Reunião Project',
        'text': """
        Olá,
        
        Consegues reunir na próxima quinta às 14h para discutir o dataset?
        Acho que o Teams é o melhor para isso.
        
        Obrigado
        """
    }
    
    # Processar
    result = pipeline.process_email(
        email_text=email['text'],
        email_id=email['id'],
        subject=email['subject'],
    )
    
    # Mostrar resultado
    if result:
        pipeline.print_sample(result)


def example_3_batch_processing():
    """Exemplo 3: Processar batch de emails."""
    print("\n" + "="*70)
    print("EXEMPLO 3: Batch Processing")
    print("="*70)
    
    # Inicializar pipeline
    pipeline = QAPipeline(
        model_name='multilingual',
        device='cpu',
        confidence_threshold=0.4,
    )
    
    # Batch de emails
    emails = [
        {
            'id': 1,
            'subject': 'Call agenda',
            'text': 'Consegues reunir depois da aula?',
        },
        {
            'id': 2,
            'subject': 'Mudança planos',
            'text': 'Hoje não vai dar 😅',
        },
        {
            'id': 3,
            'subject': 'Meeting',
            'text': 'Bora fazer uma reunião rápida quinta de manhã?',
        },
    ]
    
    # Processar batch
    results = pipeline.process_batch(emails, show_progress=True)
    
    # Estatísticas
    valid_results = [r for r in results if any(
        qa['valid'] for qa in r.qa_results.values()
    )]
    
    print(f"\nProcessados: {len(results)}/{len(emails)}")
    print(f"Com respostas válidas: {len(valid_results)}")


def example_4_load_and_integrate():
    """Exemplo 4: Carregar gold annotations e integrar resultados QA."""
    print("\n" + "="*70)
    print("EXEMPLO 4: Load & Integrate with Gold Annotations")
    print("="*70)
    
    # Inicializar pipeline
    pipeline = QAPipeline(
        model_name='multilingual',
        device='cpu',
        confidence_threshold=0.4,
    )
    
    # Criar dados de teste (substituir pelo caminho real)
    test_annotations = [
        {
            'id': 1,
            'text': 'Podemos reunir sexta às 10h no Teams?',
            'subject': 'Reunião',
            'arguments': {
                'participants': ['eu'],
                'time': ['sexta às 10h'],
                'location': ['Teams'],
                'topic': []
            }
        },
        {
            'id': 2,
            'text': 'Desculpa, tenho de adiar a reunião de terça.',
            'subject': 'Adiamento',
            'arguments': {
                'participants': [],
                'time': ['terça'],
                'location': [],
                'topic': []
            }
        }
    ]
    
    # Integrar
    integrated = pipeline.integrate_with_gold_annotations(
        test_annotations,
        output_file='qa/output/integrated_results.json'
    )
    
    print(f"\nIntegrados: {integrated['count']} items")
    
    # Mostrar amostra
    if integrated['data']:
        first = integrated['data'][0]
        print(f"\nAmostra:")
        print(f"  Email ID: {first['id']}")
        print(f"  QA Answers: {first.get('qa_answers', {})}")


def example_5_evaluation():
    """Exemplo 5: Avaliação de resultados."""
    print("\n" + "="*70)
    print("EXEMPLO 5: Evaluation")
    print("="*70)
    
    # Criar evaluador
    evaluator = QAEvaluator()
    
    # Dados de teste
    test_cases = [
        {
            'id': '1',
            'question': 'Quem participa?',
            'predicted': 'Ana',
            'reference': 'Ana',
            'confidence': 0.95,
            'category': 'participants',
        },
        {
            'id': '2',
            'question': 'Quando?',
            'predicted': 'sexta',
            'reference': 'sexta às 15h',
            'confidence': 0.80,
            'category': 'time',
        },
        {
            'id': '3',
            'question': 'Onde?',
            'predicted': 'Teams',
            'reference': 'Teams',
            'confidence': 0.85,
            'category': 'location',
        },
        {
            'id': '4',
            'question': 'Tópico?',
            'predicted': 'reunião',
            'reference': 'dataset',
            'confidence': 0.40,
            'category': 'topic',
        },
    ]
    
    # Avaliar
    for case in test_cases:
        evaluator.evaluate_example(
            example_id=case['id'],
            question=case['question'],
            predicted=case['predicted'],
            reference=case['reference'],
            confidence=case['confidence'],
            category=case['category'],
        )
    
    # Imprimir relatório
    evaluator.print_report()
    evaluator.print_error_analysis(top_n=2)
    
    # Salvar
    evaluator.save_results('qa/output/evaluation_results.json')


def example_6_dataset_generation():
    """Exemplo 6: Gerar dataset QA a partir de gold annotations."""
    print("\n" + "="*70)
    print("EXEMPLO 6: Dataset Generation")
    print("="*70)
    
    # Criar gerador
    generator = QADatasetGenerator(include_question_variations=True)
    
    # Dados de teste
    test_annotations = [
        {
            'id': 1,
            'text': 'Consegues reunir depois da aula?',
            'subject': 'Meeting',
            'arguments': {
                'participants': [],
                'time': ['depois da aula'],
                'location': [],
                'topic': []
            }
        },
        {
            'id': 2,
            'text': 'Bora fazer uma reunião rápida quinta?',
            'subject': 'Quick call',
            'arguments': {
                'participants': [],
                'time': ['quinta'],
                'location': [],
                'topic': ['reunião']
            }
        }
    ]
    
    # Processar anotações
    for annotation in test_annotations:
        generator._process_annotation(annotation)
    
    # Estatísticas
    generator.print_statistics()
    
    # Salvar
    generator.save_dataset(
        output_dir='qa/output',
        formats=['json', 'squad']
    )


def example_7_questions_overview():
    """Exemplo 7: Visualizar todas as perguntas."""
    print("\n" + "="*70)
    print("EXEMPLO 7: Questions Overview")
    print("="*70)
    
    # Imprimir perguntas
    QAQuestions.print_all_questions()
    
    # Obter perguntas primárias
    print("\n" + "="*70)
    print("Perguntas Primárias:")
    print("="*70)
    for cat_name, question in QAQuestions.get_all_primary_questions().items():
        print(f"\n{cat_name.upper()}:")
        print(f"  {question}")


def run_all_examples():
    """Executa todos os exemplos."""
    import logging
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("\n" + "="*70)
    print("QA MODULE - COMPREHENSIVE EXAMPLES")
    print("="*70)
    
    # Criar diretório de output
    Path('qa/output').mkdir(parents=True, exist_ok=True)
    
    try:
        # Executar exemplos
        example_7_questions_overview()
        example_1_quick_usage()
        example_2_single_email()
        example_3_batch_processing()
        example_5_evaluation()
        example_6_dataset_generation()
        
        # Este requer gold annotations reais
        # example_4_load_and_integrate()
        
        print("\n" + "="*70)
        print("✓ TODOS OS EXEMPLOS COMPLETOS")
        print("="*70)
    
    except Exception as e:
        print(f"\n✗ Erro: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_examples()
