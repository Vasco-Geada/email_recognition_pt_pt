"""
gold_annotations_generator.py
==============================

Gerador principal de gold annotations.

Orquestra o pipeline de:
1. Leitura de emails
2. Extração heurística
3. Validação
4. Normalização
5. Exportação

Suporta:
- Revisão manual posterior
- Edição fácil
- Formato avaliável

Author: Generated for Email Recognition PT-PT Project
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from dataclasses import asdict

from heuristic_extractors import HeuristicAnnotationExtractor, ExtractionResult
from validators import AnnotationValidator, ValidationResult, ConsistencyValidator


class GoldAnnotationsGenerator:
    """
    Gerador principal de gold annotations para projeto NLP.
    
    Pipeline:
    1. Carrega emails JSON
    2. Extrai features heuristicamente
    3. Valida estrutura
    4. Normaliza output
    5. Salva resultado
    """
    
    def __init__(self, verbose: bool = True):
        """
        Inicializa o gerador.
        
        Args:
            verbose: Se True, mostra progressão
        """
        self.verbose = verbose
        self.extractor = HeuristicAnnotationExtractor()
        self.validator = AnnotationValidator()
        self.consistency_validator = ConsistencyValidator()
        self.annotations = []
        self.errors = []
    
    def load_emails_json(self, filepath: str) -> List[Dict]:
        """
        Carrega emails de um ficheiro JSON.
        
        Espera formato:
        [
            {
                "subject": "...",
                "body": "...",
                "label": "..."
            }
        ]
        
        Args:
            filepath: Caminho para ficheiro JSON
            
        Returns:
            Lista de emails
            
        Raises:
            FileNotFoundError: Se ficheiro não existe
            json.JSONDecodeError: Se JSON inválido
        """
        if self.verbose:
            print(f"[INFO] Carregando emails de {filepath}")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                emails = json.load(f)
            
            if not isinstance(emails, list):
                raise ValueError("JSON deve ser uma lista de emails")
            
            if self.verbose:
                print(f"[INFO] {len(emails)} emails carregados")
            
            return emails
        
        except FileNotFoundError:
            raise FileNotFoundError(f"Ficheiro não encontrado: {filepath}")
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Erro JSON: {str(e)}", doc=filepath, pos=0)
    
    def process_email(self, email: Dict, email_id: int) -> Dict:
        """
        Processa um email individual.
        
        Args:
            email: Dict com email
            email_id: ID sequencial
            
        Returns:
            Anotação formatada
        """
        subject = email.get('subject', '')
        body = email.get('body', '')
        label = email.get('label', '')
        
        # Combinar subject e body
        full_text = f"{subject} {body}".strip()
        
        # Extrair features
        extractions = self.extractor.extract_all(full_text, subject=subject)
        
        # Montar anotação
        annotation = {
            'id': email_id,
            'text': body if body else subject,  # Preferir body
            'intent': label,
            'trigger': extractions['trigger'].values[0] if extractions['trigger'].values else '',
            'arguments': {
                'participants': extractions['participants'].values,
                'time': extractions['temporal'].values,
                'location': extractions['location'].values,
                'topic': extractions['topic'].values,
            },
            'confidence': {
                'trigger': extractions['trigger'].confidence,
                'participants': extractions['participants'].confidence,
                'temporal': extractions['temporal'].confidence,
                'location': extractions['location'].confidence,
                'topic': extractions['topic'].confidence,
            },
            'metadata': {
                'source_subject': subject,
                'extracted_at': datetime.now().isoformat(),
                'extraction_method': 'heuristic',
            }
        }
        
        return annotation
    
    def process_batch(self, emails: List[Dict]) -> Tuple[List[Dict], List[str]]:
        """
        Processa um lote de emails.
        
        Args:
            emails: Lista de emails
            
        Returns:
            Tuple (annotations, errors)
        """
        annotations = []
        errors = []
        
        for i, email in enumerate(emails):
            try:
                if self.verbose and (i + 1) % 10 == 0:
                    print(f"[PROGRESS] Processados {i + 1}/{len(emails)} emails")
                
                annotation = self.process_email(email, email_id=i + 1)
                annotations.append(annotation)
            
            except Exception as e:
                error_msg = f"Erro ao processar email {i + 1}: {str(e)}"
                errors.append(error_msg)
                if self.verbose:
                    print(f"[ERROR] {error_msg}")
        
        if self.verbose:
            print(f"[INFO] {len(annotations)} anotações geradas, {len(errors)} erros")
        
        return annotations, errors
    
    def validate_annotations(self, annotations: List[Dict]) -> Tuple[List[Dict], List[str]]:
        """
        Valida e normaliza anotações.
        
        Args:
            annotations: Lista de anotações
            
        Returns:
            Tuple (normalized_annotations, validation_messages)
        """
        if self.verbose:
            print("[INFO] Validando anotações...")
        
        messages = []
        normalized = []
        
        # Validar lote completo
        batch_result = self.validator.validate_batch(annotations)
        
        if batch_result.errors:
            messages.append(f"[VALIDAÇÃO] {len(batch_result.errors)} erros encontrados:")
            for error in batch_result.errors[:10]:  # Mostrar primeiros 10
                msg = f"  - {error.error_type}: {error.message}"
                messages.append(msg)
            if len(batch_result.errors) > 10:
                messages.append(f"  ... e mais {len(batch_result.errors) - 10} erros")
        
        if batch_result.warnings:
            messages.append(f"[AVISOS] {len(batch_result.warnings)} avisos:")
            for warning in batch_result.warnings[:5]:
                messages.append(f"  - {warning}")
            if len(batch_result.warnings) > 5:
                messages.append(f"  ... e mais {len(batch_result.warnings) - 5} avisos")
        
        # Normalizar todas as anotações
        for annotation in annotations:
            try:
                norm = self.validator.normalize_annotation(annotation)
                normalized.append(norm)
            except Exception as e:
                messages.append(f"Erro ao normalizar anotação {annotation.get('id')}: {str(e)}")
        
        # Verificar consistência
        consistency_warnings = self.consistency_validator.check_intent_trigger_consistency(normalized)
        if consistency_warnings:
            messages.append(f"[CONSISTÊNCIA] {len(consistency_warnings)} avisos:")
            for warning in consistency_warnings[:3]:
                messages.append(f"  - {warning}")
        
        if self.verbose:
            for msg in messages:
                print(msg)
        
        return normalized, messages
    
    def save_annotations(self, annotations: List[Dict], output_path: str, 
                        pretty: bool = True) -> bool:
        """
        Salva anotações em ficheiro JSON.
        
        Args:
            annotations: Lista de anotações
            output_path: Caminho para output
            pretty: Se True, formata JSON com indentação
            
        Returns:
            True se sucesso
        """
        try:
            if self.verbose:
                print(f"[INFO] Salvando {len(annotations)} anotações em {output_path}")
            
            # Criar diretório se não existir
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Serializar
            indent = 2 if pretty else None
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(annotations, f, indent=indent, ensure_ascii=False)
            
            if self.verbose:
                print(f"[SUCCESS] Anotações salvas com sucesso")
            
            return True
        
        except Exception as e:
            print(f"[ERROR] Erro ao salvar anotações: {str(e)}")
            return False
    
    def generate_report(self, annotations: List[Dict], errors: List[str]) -> Dict:
        """
        Gera relatório de processamento.
        
        Args:
            annotations: Lista de anotações
            errors: Lista de erros
            
        Returns:
            Dict com relatório
        """
        report = {
            'summary': {
                'total_emails': len(annotations) + len(errors),
                'successful': len(annotations),
                'errors': len(errors),
                'success_rate': len(annotations) / max(1, len(annotations) + len(errors)),
            },
            'intent_distribution': {},
            'trigger_statistics': {},
            'arguments_coverage': {
                'participants': 0,
                'time': 0,
                'location': 0,
                'topic': 0,
            },
            'confidence_average': {
                'trigger': 0.0,
                'participants': 0.0,
                'temporal': 0.0,
                'location': 0.0,
                'topic': 0.0,
            },
            'errors': errors[:20],  # Primeiros 20 erros
        }
        
        # Calcular distribuição de intents
        for ann in annotations:
            intent = ann.get('intent', 'unknown')
            report['intent_distribution'][intent] = report['intent_distribution'].get(intent, 0) + 1
            
            # Calcular trigger statistics
            triggers = ann.get('trigger', [])
            if isinstance(triggers, str):
                triggers = [triggers] if triggers else []
            elif not isinstance(triggers, list):
                triggers = []
            
            for trigger in triggers:
                if trigger:
                    report['trigger_statistics'][trigger] = report['trigger_statistics'].get(trigger, 0) + 1
            
            # Calcular cobertura de arguments
            args = ann.get('arguments', {})
            if args.get('participants'):
                report['arguments_coverage']['participants'] += 1
            if args.get('time'):
                report['arguments_coverage']['time'] += 1
            if args.get('location'):
                report['arguments_coverage']['location'] += 1
            if args.get('topic'):
                report['arguments_coverage']['topic'] += 1
            
            # Calcular média de confidence
            conf = ann.get('confidence', {})
            for key in report['confidence_average']:
                if key in conf:
                    report['confidence_average'][key] += conf[key]
        
        # Normalizar médias
        if annotations:
            for key in report['confidence_average']:
                report['confidence_average'][key] /= len(annotations)
        
        # Calcular percentagens de cobertura
        for key in report['arguments_coverage']:
            if annotations:
                report['arguments_coverage'][key] = {
                    'count': report['arguments_coverage'][key],
                    'percentage': report['arguments_coverage'][key] / len(annotations) * 100
                }
        
        return report
    
    def print_report(self, report: Dict):
        """
        Imprime relatório formatado.
        
        Args:
            report: Dict com relatório
        """
        print("\n" + "="*60)
        print("RELATÓRIO DE PROCESSAMENTO")
        print("="*60)
        
        summary = report.get('summary', {})
        print(f"\nResumo:")
        print(f"  Total de emails: {summary.get('total_emails', 0)}")
        print(f"  Processados com sucesso: {summary.get('successful', 0)}")
        print(f"  Erros: {summary.get('errors', 0)}")
        print(f"  Taxa de sucesso: {summary.get('success_rate', 0):.1%}")
        
        print(f"\nDistribuição de Intents:")
        for intent, count in report.get('intent_distribution', {}).items():
            print(f"  - {intent}: {count}")
        
        print(f"\nTriggers encontrados:")
        for trigger, count in report.get('trigger_statistics', {}).items():
            print(f"  - {trigger}: {count}")
        
        print(f"\nCobertura de Arguments:")
        for arg, data in report.get('arguments_coverage', {}).items():
            if isinstance(data, dict):
                print(f"  - {arg}: {data.get('count', 0)} ({data.get('percentage', 0):.1f}%)")
            else:
                print(f"  - {arg}: {data}")
        
        print(f"\nConfiança média:")
        for field, confidence in report.get('confidence_average', {}).items():
            print(f"  - {field}: {confidence:.2f}")
        
        print("\n" + "="*60)
    
    def run(self, input_json: str, output_json: str) -> bool:
        """
        Executa pipeline completo.
        
        Args:
            input_json: Caminho para JSON de input
            output_json: Caminho para JSON de output
            
        Returns:
            True se sucesso
        """
        try:
            # 1. Carregar emails
            emails = self.load_emails_json(input_json)
            
            # 2. Processar batch
            annotations, errors = self.process_batch(emails)
            
            # 3. Validar e normalizar
            normalized, validation_msgs = self.validate_annotations(annotations)
            
            # 4. Salvar
            success = self.save_annotations(normalized, output_json)
            
            # 5. Gerar relatório
            report = self.generate_report(normalized, errors)
            self.print_report(report)
            
            return success
        
        except Exception as e:
            print(f"[FATAL ERROR] {str(e)}")
            return False


def main():
    """Função principal para CLI"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Gerador de Gold Annotations para NLP'
    )
    parser.add_argument(
        'input',
        help='Ficheiro JSON de input com emails'
    )
    parser.add_argument(
        'output',
        help='Ficheiro JSON de output com gold annotations'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Modo verbose'
    )
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Modo silencioso'
    )
    
    args = parser.parse_args()
    
    verbose = not args.quiet and (args.verbose or not args.quiet)
    
    generator = GoldAnnotationsGenerator(verbose=verbose)
    success = generator.run(args.input, args.output)
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
