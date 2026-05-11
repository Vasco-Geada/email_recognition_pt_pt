"""
example_usage.py
================

Exemplo de uso do sistema de gold annotations.

Demonstra:
1. Carregamento de dataset
2. Geração de gold annotations
3. Validação
4. Salvar output
5. Gerar relatório

Author: Generated for Email Recognition PT-PT Project
"""

import json
import sys
from pathlib import Path

# Importar módulos do sistema
from gold_annotations_generator import GoldAnnotationsGenerator
from evaluate_annotations import AnnotationEvaluator


def example_1_basic_generation():
    """Exemplo básico: gerar gold annotations"""
    print("\n" + "="*70)
    print("EXEMPLO 1: Geração Básica de Gold Annotations")
    print("="*70 + "\n")
    
    # Caminho para o dataset
    input_path = "../dataset/realistic_emails_v2.json"
    output_path = "./output/gold_annotations_v1.json"
    
    # Criar gerador
    generator = GoldAnnotationsGenerator(verbose=True)
    
    # Executar pipeline
    success = generator.run(input_path, output_path)
    
    if success:
        print(f"\n[SUCCESS] Gold annotations geradas com sucesso em: {output_path}")
        return output_path
    else:
        print("\n[ERROR] Erro ao gerar gold annotations")
        return None


def example_2_load_and_inspect():
    """Exemplo: carregar e inspecionar anotações"""
    print("\n" + "="*70)
    print("EXEMPLO 2: Carregar e Inspecionar Anotações")
    print("="*70 + "\n")
    
    filepath = "./output/gold_annotations_v1.json"
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        
        print(f"Total de anotações: {len(annotations)}\n")
        
        # Mostrar primeiras 3 anotações
        for ann in annotations[:3]:
            print(f"ID: {ann.get('id')}")
            print(f"  Intent: {ann.get('intent')}")
            print(f"  Trigger: {ann.get('trigger')}")
            print(f"  Participantes: {ann.get('arguments', {}).get('participants', [])}")
            print(f"  Horário: {ann.get('arguments', {}).get('time', [])}")
            print(f"  Localização: {ann.get('arguments', {}).get('location', [])}")
            print(f"  Tópico: {ann.get('arguments', {}).get('topic', [])}")
            print(f"  Confiança: {ann.get('confidence', {})}")
            print()
        
        return annotations
    
    except Exception as e:
        print(f"Erro: {str(e)}")
        return None


def example_3_manual_review():
    """Exemplo: revisar e editar manualmente"""
    print("\n" + "="*70)
    print("EXEMPLO 3: Revisão Manual (Simular)")
    print("="*70 + "\n")
    
    filepath = "./output/gold_annotations_v1.json"
    review_path = "./output/gold_annotations_reviewed.json"
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        
        print(f"Carregadas {len(annotations)} anotações para revisão\n")
        
        # Simular revisão: corrigir algumas anotações
        for i, ann in enumerate(annotations[:5]):  # Revisar primeiras 5
            print(f"Revisão da anotação {i + 1}:")
            print(f"  Texto: {ann.get('text')[:50]}...")
            print(f"  Intent atual: {ann.get('intent')}")
            print(f"  Trigger atual: {ann.get('trigger')}")
            
            # Simular aceitação (em uso real, seria interativo)
            print("  [OK] Aceito")
            print()
        
        # Salvar anotações revisadas
        Path(review_path).parent.mkdir(parents=True, exist_ok=True)
        with open(review_path, 'w', encoding='utf-8') as f:
            json.dump(annotations, f, indent=2, ensure_ascii=False)
        
        print(f"[SUCCESS] Anotacoes revisadas salvas em: {review_path}")
        return review_path
    
    except Exception as e:
        print(f"Erro: {str(e)}")
        return None


def example_4_validation():
    """Exemplo: validação de anotações"""
    print("\n" + "="*70)
    print("EXEMPLO 4: Validação de Anotações")
    print("="*70 + "\n")
    
    from validators import AnnotationValidator
    
    filepath = "./output/gold_annotations_v1.json"
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        
        validator = AnnotationValidator()
        result = validator.validate_batch(annotations)
        
        print(f"Validação: {'[OK] Sucesso' if result.is_valid else '[FAIL] Falha'}")
        print(f"Erros encontrados: {len(result.errors)}")
        print(f"Avisos encontrados: {len(result.warnings)}")
        
        if result.errors:
            print("\nPrimeiros erros:")
            for error in result.errors[:3]:
                print(f"  - {error.error_type}: {error.message}")
        
        if result.warnings:
            print("\nPrimeiros avisos:")
            for warning in result.warnings[:3]:
                print(f"  - {warning}")
        
        return True
    
    except Exception as e:
        print(f"Erro: {str(e)}")
        return False


def example_5_evaluation():
    """Exemplo: avaliar gold annotations"""
    print("\n" + "="*70)
    print("EXEMPLO 5: Avaliação (Simular Predições)")
    print("="*70 + "\n")
    
    gold_path = "./output/gold_annotations_v1.json"
    pred_path = "./output/predictions_v1.json"
    report_path = "./output/evaluation_report.json"
    
    try:
        # Carregar gold annotations
        with open(gold_path, 'r', encoding='utf-8') as f:
            gold = json.load(f)
        
        # Simular predições (usando os mesmos dados com pequenas variações)
        predictions = []
        for ann in gold[:50]:  # Usar apenas 50 para teste
            pred = ann.copy()
            # Simular alguns erros em predições
            if ann.get('id', 0) % 5 == 0:
                pred['trigger'] = 'wrong_trigger'  # Simular erro
            predictions.append(pred)
        
        # Salvar predições
        Path(pred_path).parent.mkdir(parents=True, exist_ok=True)
        with open(pred_path, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)
        
        # Avaliar
        evaluator = AnnotationEvaluator()
        result = evaluator.evaluate_batch(gold[:50], predictions)
        
        evaluator.print_metrics(result)
        evaluator.save_report(result, report_path)
        
        print(f"\n[OK] Relatório salvo em: {report_path}")
        return True
    
    except Exception as e:
        print(f"Erro: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def create_sample_dataset():
    """Cria um dataset de amostra para teste"""
    print("\n[INFO] Criando dataset de amostra para teste...\n")
    
    sample_emails = [
        {
            "subject": "Reunião amanhã",
            "body": "Boas Ana, podemos reunir amanhã às 15h no Teams para discutir o dataset?",
            "label": "agendamento_reuniao"
        },
        {
            "subject": "Cancelar reunião de sexta",
            "body": "Infelizmente não consigo aparecer sexta de manhã à reunião da sala 2.3. Marca para próxima semana?",
            "label": "cancelamento_reuniao"
        },
        {
            "subject": "Re: Reunião confirmada",
            "body": "Confirmed! Vejo te sexta às 14h na biblioteca para falarmos sobre o pipeline NLP.",
            "label": "reuniao_confirmada"
        },
        {
            "subject": "Combinar horário",
            "body": "Professor Silva, podemos marcar uma reunião para discutir a dissertação? Estou disponível depois de almoço.",
            "label": "agendamento_reuniao"
        },
        {
            "subject": "Reunião com o grupo",
            "body": "Oi a todos, combinámos para amanhã às 16h via Zoom para falarmos dos resultados das métricas F1?",
            "label": "agendamento_reuniao"
        },
    ]
    
    sample_path = "./output/sample_emails.json"
    Path(sample_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(sample_path, 'w', encoding='utf-8') as f:
        json.dump(sample_emails, f, indent=2, ensure_ascii=False)
    
    print(f"[SUCCESS] Dataset de amostra criado: {sample_path}\n")
    return sample_path


def main():
    """Executa exemplos"""
    print("\n" + "="*70)
    print("SISTEMA DE GOLD ANNOTATIONS - EXEMPLOS DE USO")
    print("="*70)
    
    # Criar diretório de output
    Path("./output").mkdir(exist_ok=True)
    
    # Criar dataset de amostra
    sample_path = create_sample_dataset()
    
    try:
        # Executar exemplos
        print("\n[1/5] Geração de Gold Annotations...")
        gold_path = example_1_basic_generation()
        
        if gold_path:
            print("\n[2/5] Carregamento e Inspeção...")
            annotations = example_2_load_and_inspect()
            
            if annotations:
                print("\n[3/5] Revisão Manual (Simulada)...")
                review_path = example_3_manual_review()
                
                print("\n[4/5] Validação...")
                example_4_validation()
                
                print("\n[5/5] Avaliação...")
                example_5_evaluation()
        
        print("\n" + "="*70)
        print("[OK] TODOS OS EXEMPLOS COMPLETADOS COM SUCESSO")
        print("="*70)
        print("\nArtefatos gerados em ./output/:")
        print("  - gold_annotations_v1.json")
        print("  - gold_annotations_reviewed.json")
        print("  - predictions_v1.json")
        print("  - evaluation_report.json")
        
    except Exception as e:
        print(f"\n[FAIL] Erro: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
