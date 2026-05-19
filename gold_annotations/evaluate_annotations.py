"""
evaluate_annotations.py
=======================

Avaliador de gold annotations e predições.

Implementa:
- Exact Match
- Precision, Recall, F1
- Métricas por classe
- Matriz de confusão
- Comparação entre anotadores
- Relatórios detalhados

Author: Generated for Email Recognition PT-PT Project
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict
import math


@dataclass
class ClassificationMetrics:
    """Métricas de classificação para uma classe"""
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    true_negatives: int = 0
    
    @property
    def precision(self) -> float:
        """TP / (TP + FP)"""
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 0.0
    
    @property
    def recall(self) -> float:
        """TP / (TP + FN)"""
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 0.0
    
    @property
    def f1(self) -> float:
        """Harmonic mean de precision e recall"""
        p = self.precision
        r = self.recall
        denom = p + r
        return 2 * (p * r) / denom if denom > 0 else 0.0
    
    @property
    def accuracy(self) -> float:
        """(TP + TN) / (TP + TN + FP + FN)"""
        total = sum([self.true_positives, self.true_negatives, 
                    self.false_positives, self.false_negatives])
        return (self.true_positives + self.true_negatives) / total if total > 0 else 0.0


@dataclass
class EvaluationResult:
    """Resultado de avaliação completo"""
    intent_metrics: Dict[str, ClassificationMetrics]
    argument_metrics: Dict[str, ClassificationMetrics]
    overall_metrics: ClassificationMetrics
    exact_match_accuracy: float
    confusion_matrix: Dict[str, Dict[str, int]]
    error_analysis: Dict
    

class AnnotationEvaluator:
    """
    Avaliador de gold annotations.
    
    Compara:
    - Intents preditos vs gold
    - Arguments extraídos vs gold
    - Triggers detectados
    """
    
    def __init__(self):
        """Inicializa avaliador"""
        self.intent_metrics = defaultdict(lambda: ClassificationMetrics())
        self.argument_metrics = defaultdict(lambda: ClassificationMetrics())
        self.confusion_matrix = defaultdict(lambda: defaultdict(int))
        self.exact_matches = 0
        self.total_samples = 0
    
    def load_annotations(self, filepath: str) -> List[Dict]:
        """Carrega anotações de ficheiro JSON"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                annotations = json.load(f)
            return annotations if isinstance(annotations, list) else []
        except Exception as e:
            print(f"Erro ao carregar {filepath}: {str(e)}")
            return []
    
    def evaluate_intent(self, gold_intent: str, pred_intent: str) -> None:
        """
        Avalia intents.
        
        Args:
            gold_intent: Intent correto (gold)
            pred_intent: Intent predito
        """
        self.confusion_matrix[gold_intent][pred_intent] += 1
        
        if gold_intent == pred_intent:
            self.intent_metrics[gold_intent].true_positives += 1
        else:
            self.intent_metrics[gold_intent].false_negatives += 1
            self.intent_metrics[pred_intent].false_positives += 1
    
    def evaluate_arguments(self, gold_args: Dict[str, List[str]], 
                          pred_args: Dict[str, List[str]]) -> bool:
        """
        Avalia arguments (exact match por tipo).
        
        Args:
            gold_args: Arguments corretos
            pred_args: Arguments preditos
            
        Returns:
            True se match exato
        """
        is_exact_match = True
        
        for arg_type in ['participants', 'time', 'location', 'topic']:
            gold_values = set(gold_args.get(arg_type, []))
            pred_values = set(pred_args.get(arg_type, []))
            
            if gold_values == pred_values:
                self.argument_metrics[arg_type].true_positives += len(gold_values)
                self.argument_metrics[arg_type].true_negatives += 1
            else:
                # Calcular TP, FP, FN
                tp = len(gold_values & pred_values)
                fp = len(pred_values - gold_values)
                fn = len(gold_values - pred_values)
                
                self.argument_metrics[arg_type].true_positives += tp
                self.argument_metrics[arg_type].false_positives += fp
                self.argument_metrics[arg_type].false_negatives += fn
                
                is_exact_match = False
        
        return is_exact_match
    
    def evaluate_pair(self, gold_annotation: Dict, pred_annotation: Dict) -> Dict:
        """
        Avalia um par de anotações (gold vs predicted).
        
        Args:
            gold_annotation: Anotação correta
            pred_annotation: Anotação predita
            
        Returns:
            Dict com resultados da comparação
        """
        self.total_samples += 1
        
        gold_intent = gold_annotation.get('intent', 'unknown')
        pred_intent = pred_annotation.get('intent', 'unknown')
        
        self.evaluate_intent(gold_intent, pred_intent)
        
        gold_args = gold_annotation.get('arguments', {})
        pred_args = pred_annotation.get('arguments', {})
        
        is_exact_match = self.evaluate_arguments(gold_args, pred_args)
        
        if is_exact_match and gold_intent == pred_intent:
            self.exact_matches += 1
        
        return {
            'id': gold_annotation.get('id'),
            'intent_match': gold_intent == pred_intent,
            'args_match': is_exact_match,
            'exact_match': is_exact_match and gold_intent == pred_intent,
            'gold_intent': gold_intent,
            'pred_intent': pred_intent,
        }
    
    def evaluate_batch(self, gold_annotations: List[Dict], 
                      pred_annotations: List[Dict]) -> EvaluationResult:
        """
        Avalia um lote de anotações.
        
        Args:
            gold_annotations: Anotações corretas
            pred_annotations: Anotações preditas
            
        Returns:
            EvaluationResult completo
        """
        if len(gold_annotations) != len(pred_annotations):
            print(f"Aviso: número diferente de anotações ({len(gold_annotations)} vs {len(pred_annotations)})")
        
        # Criar índice por ID para matching rápido
        pred_by_id = {ann.get('id'): ann for ann in pred_annotations}
        
        error_analysis = {'mismatches': [], 'missing': [], 'extra': []}
        
        for gold_ann in gold_annotations:
            ann_id = gold_ann.get('id')
            
            if ann_id in pred_by_id:
                pred_ann = pred_by_id[ann_id]
                result = self.evaluate_pair(gold_ann, pred_ann)
                
                if not result['exact_match']:
                    error_analysis['mismatches'].append(result)
            else:
                error_analysis['missing'].append(ann_id)
        
        # Verificar predições extras (não em gold)
        for ann_id, pred_ann in pred_by_id.items():
            if not any(ann.get('id') == ann_id for ann in gold_annotations):
                error_analysis['extra'].append(ann_id)
        
        # Consolidar métricas gerais
        overall = ClassificationMetrics()
        for metrics in self.intent_metrics.values():
            overall.true_positives += metrics.true_positives
            overall.false_positives += metrics.false_positives
            overall.false_negatives += metrics.false_negatives
        
        exact_match_acc = self.exact_matches / self.total_samples if self.total_samples > 0 else 0.0
        
        return EvaluationResult(
            intent_metrics=dict(self.intent_metrics),
            argument_metrics=dict(self.argument_metrics),
            overall_metrics=overall,
            exact_match_accuracy=exact_match_acc,
            confusion_matrix=dict(self.confusion_matrix),
            error_analysis=error_analysis,
        )
    
    def print_metrics(self, result: EvaluationResult):
        """
        Imprime métricas formatadas.
        
        Args:
            result: EvaluationResult
        """
        print("\n" + "="*70)
        print("RELATÓRIO DE AVALIAÇÃO")
        print("="*70)
        
        # Métricas gerais
        print("\n[MÉTRICAS GERAIS]")
        print(f"  Total de amostras: {self.total_samples}")
        print(f"  Exact Match: {self.exact_matches}/{self.total_samples} ({result.exact_match_accuracy:.1%})")
        
        # Métricas por intent
        print("\n[MÉTRICAS POR INTENT]")
        print(f"{'Intent':<25} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Count':<10}")
        print("-" * 70)
        
        for intent, metrics in sorted(result.intent_metrics.items()):
            count = metrics.true_positives + metrics.false_negatives
            print(f"{intent:<25} {metrics.precision:<12.4f} {metrics.recall:<12.4f} {metrics.f1:<12.4f} {count:<10}")
        
        # Métricas gerais de intent
        print(f"\n{'OVERALL':<25} {result.overall_metrics.precision:<12.4f} {result.overall_metrics.recall:<12.4f} {result.overall_metrics.f1:<12.4f}")
        
        # Métricas por argument
        print("\n[MÉTRICAS POR ARGUMENT]")
        print(f"{'Argument':<25} {'Precision':<12} {'Recall':<12} {'F1':<12}")
        print("-" * 70)
        
        for arg_type, metrics in sorted(result.argument_metrics.items()):
            print(f"{arg_type:<25} {metrics.precision:<12.4f} {metrics.recall:<12.4f} {metrics.f1:<12.4f}")
        
        # Matriz de confusão
        if result.confusion_matrix:
            print("\n[MATRIZ DE CONFUSÃO - INTENTS]")
            # Header
            intents = sorted(set(result.confusion_matrix.keys()) | 
                            set(sum([list(v.keys()) for v in result.confusion_matrix.values()], [])))
            print(f"{'Pred \\ Gold':<15}", end='')
            for intent in intents:
                print(f"{intent:<15}", end='')
            print()
            print("-" * (15 + 15 * len(intents)))
            
            for gold_intent in intents:
                print(f"{gold_intent:<15}", end='')
                for pred_intent in intents:
                    count = result.confusion_matrix.get(gold_intent, {}).get(pred_intent, 0)
                    print(f"{count:<15}", end='')
                print()
        
        # Análise de erros
        print("\n[ANÁLISE DE ERROS]")
        print(f"  Mismatches: {len(result.error_analysis['mismatches'])}")
        if result.error_analysis['mismatches'][:3]:
            for error in result.error_analysis['mismatches'][:3]:
                print(f"    - ID {error['id']}: {error['gold_intent']} -> {error['pred_intent']}")
        
        print(f"  Missing predictions: {len(result.error_analysis['missing'])}")
        print(f"  Extra predictions: {len(result.error_analysis['extra'])}")
        
        print("\n" + "="*70)
    
    def save_report(self, result: EvaluationResult, output_path: str) -> bool:
        """
        Salva relatório em JSON.
        
        Args:
            result: EvaluationResult
            output_path: Caminho para ficheiro
            
        Returns:
            True se sucesso
        """
        try:
            report = {
                'summary': {
                    'total_samples': self.total_samples,
                    'exact_matches': self.exact_matches,
                    'exact_match_accuracy': result.exact_match_accuracy,
                },
                'intent_metrics': {
                    intent: {
                        'precision': metrics.precision,
                        'recall': metrics.recall,
                        'f1': metrics.f1,
                        'tp': metrics.true_positives,
                        'fp': metrics.false_positives,
                        'fn': metrics.false_negatives,
                    }
                    for intent, metrics in result.intent_metrics.items()
                },
                'argument_metrics': {
                    arg: {
                        'precision': metrics.precision,
                        'recall': metrics.recall,
                        'f1': metrics.f1,
                        'tp': metrics.true_positives,
                        'fp': metrics.false_positives,
                        'fn': metrics.false_negatives,
                    }
                    for arg, metrics in result.argument_metrics.items()
                },
                'overall_metrics': {
                    'precision': result.overall_metrics.precision,
                    'recall': result.overall_metrics.recall,
                    'f1': result.overall_metrics.f1,
                },
                'confusion_matrix': result.confusion_matrix,
                'error_analysis': result.error_analysis,
            }
            
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            print(f"[INFO] Relatório salvo em {output_path}")
            return True
        
        except Exception as e:
            print(f"[ERROR] Erro ao salvar relatório: {str(e)}")
            return False


def main():
    """Função principal para CLI"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Avaliador de Gold Annotations')
    parser.add_argument('gold', help='Ficheiro JSON com gold annotations')
    parser.add_argument('predictions', help='Ficheiro JSON com predições')
    parser.add_argument('-o', '--output', help='Ficheiro JSON para salvar relatório')
    parser.add_argument('-v', '--verbose', action='store_true', help='Modo verbose')
    
    args = parser.parse_args()
    
    evaluator = AnnotationEvaluator()
    
    print("[INFO] Carregando anotações...")
    gold = evaluator.load_annotations(args.gold)
    predictions = evaluator.load_annotations(args.predictions)
    
    if not gold or not predictions:
        print("[ERROR] Falha ao carregar ficheiros")
        return 1
    
    print(f"[INFO] Gold: {len(gold)} anotações")
    print(f"[INFO] Predictions: {len(predictions)} anotações")
    
    print("[INFO] Avaliando...")
    result = evaluator.evaluate_batch(gold, predictions)
    
    evaluator.print_metrics(result)
    
    if args.output:
        evaluator.save_report(result, args.output)
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
