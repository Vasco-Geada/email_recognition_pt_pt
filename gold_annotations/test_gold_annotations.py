"""
test_gold_annotations.py
========================

Testes unitários para o sistema de gold annotations.

Testa:
- Extractores heurísticos
- Validadores
- Gerador
- Avaliador

Author: Generated for Email Recognition PT-PT Project
"""

import json
import tempfile
from pathlib import Path

# Importar módulos
from heuristic_extractors import (
    TriggerExtractor, ParticipantExtractor, TemporalExtractor,
    LocationExtractor, TopicExtractor, HeuristicAnnotationExtractor
)
from validators import AnnotationValidator, JSONValidator, ConsistencyValidator
from gold_annotations_generator import GoldAnnotationsGenerator
from evaluate_annotations import AnnotationEvaluator


class TestTriggerExtractor:
    """Testes para TriggerExtractor"""
    
    def __init__(self):
        self.extractor = TriggerExtractor()
    
    def test_basic_triggers(self):
        """Testa detecção de triggers básicos"""
        test_cases = [
            ("Vamos reunir amanhã", "reunir"),
            ("Marcar uma reunião para sexta", "marcar"),
            ("Cancelar a reunião", "cancelar"),
            ("Confirmar presença", "confirmar"),
        ]
        
        print("\n[TEST] TriggerExtractor - Triggers Básicos")
        for text, expected_trigger in test_cases:
            result = self.extractor.extract(text)
            if result.values and expected_trigger in result.values:
                print(f"  ✓ '{text[:30]}...' -> {expected_trigger}")
            else:
                print(f"  ✗ '{text[:30]}...' esperado {expected_trigger}, obtive {result.values}")


class TestParticipantExtractor:
    """Testes para ParticipantExtractor"""
    
    def __init__(self):
        self.extractor = ParticipantExtractor()
    
    def test_basic_participants(self):
        """Testa detecção de participantes"""
        test_cases = [
            ("Boas Ana, como estás?", "Ana"),
            ("Professor Silva está disponível?", "Silva"),
            ("Olá Dr. João, tudo bem?", "João"),
        ]
        
        print("\n[TEST] ParticipantExtractor - Participantes Básicos")
        for text, expected_participant in test_cases:
            result = self.extractor.extract(text)
            found = any(expected_participant.lower() in p.lower() for p in result.values)
            if found:
                print(f"  ✓ '{text[:40]}...' -> encontrado {expected_participant}")
            else:
                print(f"  ✗ '{text[:40]}...' não encontrou {expected_participant}")


class TestTemporalExtractor:
    """Testes para TemporalExtractor"""
    
    def __init__(self):
        self.extractor = TemporalExtractor()
    
    def test_basic_temporal(self):
        """Testa detecção de expressões temporais"""
        test_cases = [
            ("reunir amanhã", "amanhã"),
            ("sexta de manhã", "sexta"),
            ("às 15h", "15"),
            ("depois de almoço", "almoço"),
        ]
        
        print("\n[TEST] TemporalExtractor - Expressões Temporais")
        for text, expected_temporal in test_cases:
            result = self.extractor.extract(text)
            found = any(expected_temporal.lower() in expr.lower() for expr in result.values)
            if found:
                print(f"  ✓ '{text}' -> encontrado {expected_temporal}")
            else:
                print(f"  ✗ '{text}' não encontrou {expected_temporal} em {result.values}")


class TestLocationExtractor:
    """Testes para LocationExtractor"""
    
    def __init__(self):
        self.extractor = LocationExtractor()
    
    def test_basic_location(self):
        """Testa detecção de localizações"""
        test_cases = [
            ("reunir no Teams", "Teams"),
            ("via Zoom", "Zoom"),
            ("sala 2.3", "sala"),
            ("biblioteca", "biblioteca"),
        ]
        
        print("\n[TEST] LocationExtractor - Localizações")
        for text, expected_location in test_cases:
            result = self.extractor.extract(text)
            found = any(expected_location.lower() in loc.lower() for loc in result.values)
            if found:
                print(f"  ✓ '{text}' -> encontrado {expected_location}")
            else:
                print(f"  ✗ '{text}' não encontrou {expected_location}")


class TestTopicExtractor:
    """Testes para TopicExtractor"""
    
    def __init__(self):
        self.extractor = TopicExtractor()
    
    def test_basic_topic(self):
        """Testa detecção de tópicos"""
        test_cases = [
            ("falar sobre a dissertação", "dissertação"),
            ("dataset e pipeline", "pipeline"),
            ("métricas F1", "F1"),
        ]
        
        print("\n[TEST] TopicExtractor - Tópicos")
        for text, expected_topic in test_cases:
            result = self.extractor.extract(text)
            found = any(expected_topic.lower() in t.lower() for t in result.values)
            if found:
                print(f"  ✓ '{text}' -> encontrado {expected_topic}")
            else:
                print(f"  ✗ '{text}' não encontrou {expected_topic} em {result.values}")


class TestAnnotationValidator:
    """Testes para AnnotationValidator"""
    
    def __init__(self):
        self.validator = AnnotationValidator()
    
    def test_valid_annotation(self):
        """Testa validação de anotação válida"""
        annotation = {
            'id': 1,
            'text': 'Reunir amanhã',
            'intent': 'agendamento_reuniao',
            'trigger': ['reunir'],
            'arguments': {
                'participants': ['Ana'],
                'time': ['amanhã'],
                'location': [],
                'topic': [],
            }
        }
        
        print("\n[TEST] AnnotationValidator - Anotação Válida")
        result = self.validator.validate_single_annotation(annotation)
        if result.is_valid:
            print("  ✓ Anotação válida")
        else:
            print(f"  ✗ Anotação inválida: {result.errors}")
    
    def test_invalid_annotation(self):
        """Testa validação de anotação inválida"""
        annotation = {
            'id': 1,
            'text': 'Reunir amanhã',
            # Falta 'intent'
            'trigger': ['reunir'],
            'arguments': {}
        }
        
        print("\n[TEST] AnnotationValidator - Anotação Inválida")
        result = self.validator.validate_single_annotation(annotation)
        if not result.is_valid and result.errors:
            print(f"  ✓ Erros detectados: {len(result.errors)}")
        else:
            print("  ✗ Deveria ter detectado erros")
    
    def test_normalize_annotation(self):
        """Testa normalização de anotação"""
        annotation = {
            'id': 1,
            'text': 'Reunir amanhã',
            'intent': 'agendamento_reuniao',
            'trigger': 'reunir',  # String em vez de lista
            'arguments': {'participants': 'Ana'}  # String em vez de lista
        }
        
        print("\n[TEST] AnnotationValidator - Normalização")
        normalized = self.validator.normalize_annotation(annotation)
        
        # Verificar se trigger é lista
        if isinstance(normalized['trigger'], list):
            print("  ✓ Trigger convertido para lista")
        
        # Verificar se arguments são listas
        if isinstance(normalized['arguments']['participants'], list):
            print("  ✓ Arguments convertidos para listas")


class TestGenerator:
    """Testes para GoldAnnotationsGenerator"""
    
    def test_full_pipeline(self):
        """Testa pipeline completo"""
        print("\n[TEST] GoldAnnotationsGenerator - Pipeline Completo")
        
        # Criar dados de teste
        test_emails = [
            {
                'subject': 'Reunião amanhã',
                'body': 'Boas Ana, podemos reunir amanhã às 15h no Teams?',
                'label': 'agendamento_reuniao'
            },
            {
                'subject': 'Cancelar reunião',
                'body': 'Não consigo aparecer sexta. Desculpa!',
                'label': 'cancelamento_reuniao'
            }
        ]
        
        # Salvar em ficheiro temporário
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, 
                                        encoding='utf-8') as f:
            json.dump(test_emails, f)
            input_path = f.name
        
        output_path = input_path.replace('.json', '_output.json')
        
        try:
            # Executar gerador
            generator = GoldAnnotationsGenerator(verbose=False)
            success = generator.run(input_path, output_path)
            
            if success:
                # Verificar output
                with open(output_path, 'r', encoding='utf-8') as f:
                    annotations = json.load(f)
                
                if len(annotations) == len(test_emails):
                    print(f"  ✓ {len(annotations)} anotações geradas")
                else:
                    print(f"  ✗ Esperadas {len(test_emails)}, obtidas {len(annotations)}")
            else:
                print("  ✗ Falha ao gerar anotações")
        
        finally:
            # Limpar
            Path(input_path).unlink(missing_ok=True)
            Path(output_path).unlink(missing_ok=True)


class TestEvaluator:
    """Testes para AnnotationEvaluator"""
    
    def test_exact_match(self):
        """Testa avaliação de exact match"""
        print("\n[TEST] AnnotationEvaluator - Exact Match")
        
        gold = {
            'id': 1,
            'intent': 'agendamento_reuniao',
            'trigger': ['reunir'],
            'arguments': {
                'participants': ['Ana'],
                'time': ['amanhã às 15h'],
                'location': ['Teams'],
                'topic': []
            }
        }
        
        pred = gold.copy()  # Predição perfeita
        
        evaluator = AnnotationEvaluator()
        result = evaluator.evaluate_pair(gold, pred)
        
        if result['exact_match']:
            print("  ✓ Exact match detectado")
        else:
            print("  ✗ Exact match não detectado")


def run_all_tests():
    """Executa todos os testes"""
    print("\n" + "="*70)
    print("TESTES DO SISTEMA DE GOLD ANNOTATIONS")
    print("="*70)
    
    # Testes de extractores
    TestTriggerExtractor().test_basic_triggers()
    TestParticipantExtractor().test_basic_participants()
    TestTemporalExtractor().test_basic_temporal()
    TestLocationExtractor().test_basic_location()
    TestTopicExtractor().test_basic_topic()
    
    # Testes de validadores
    validator_tests = TestAnnotationValidator()
    validator_tests.test_valid_annotation()
    validator_tests.test_invalid_annotation()
    validator_tests.test_normalize_annotation()
    
    # Testes de gerador
    TestGenerator().test_full_pipeline()
    
    # Testes de avaliador
    TestEvaluator().test_exact_match()
    
    print("\n" + "="*70)
    print("✓ TESTES CONCLUÍDOS")
    print("="*70 + "\n")


if __name__ == '__main__':
    run_all_tests()
