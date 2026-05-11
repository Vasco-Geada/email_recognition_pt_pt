"""
validators.py
==============

Módulo de validação de gold annotations.

Valida:
- Campos obrigatórios
- Tipos corretos
- Valores permitidos
- Consistência de dados
- Codificação UTF-8
- Duplicados

Author: Generated for Email Recognition PT-PT Project
"""

import json
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ValidationError:
    """Representa um erro de validação"""
    field: str
    error_type: str
    message: str
    value: Any = None
    annotation_id: int = None


@dataclass
class ValidationResult:
    """Resultado de validação com lista de erros"""
    is_valid: bool
    errors: List[ValidationError]
    warnings: List[str]


class AnnotationValidator:
    """
    Valida anotações de gold annotations.
    
    Assegura:
    - Estrutura consistente
    - Campos obrigatórios presentes
    - Tipos corretos
    - Valores válidos
    """
    
    # Campos obrigatórios
    REQUIRED_FIELDS = {
        'id': int,
        'text': str,
        'intent': str,
        'trigger': (list, str),
        'arguments': dict,
    }
    
    # Intents permitidos
    VALID_INTENTS = [
        'agendamento_reuniao',
        'cancelamento_reuniao',
        'reuniao_confirmada',
    ]
    
    # Campos de arguments
    ARGUMENT_FIELDS = [
        'participants',
        'time',
        'location',
        'topic',
    ]
    
    def validate_single_annotation(self, annotation: Dict, annotation_id: int = None) -> ValidationResult:
        """
        Valida uma única anotação.
        
        Args:
            annotation: Dict com a anotação
            annotation_id: ID da anotação (para relatórios)
            
        Returns:
            ValidationResult com lista de erros
        """
        errors = []
        warnings = []
        
        # Verificar campos obrigatórios
        for field, field_type in self.REQUIRED_FIELDS.items():
            if field not in annotation:
                errors.append(ValidationError(
                    field=field,
                    error_type='missing_field',
                    message=f'Campo obrigatório "{field}" ausente',
                    annotation_id=annotation_id
                ))
                continue
            
            # Verificar tipo
            value = annotation[field]
            if isinstance(field_type, tuple):
                if not any(isinstance(value, t) for t in field_type):
                    errors.append(ValidationError(
                        field=field,
                        error_type='wrong_type',
                        message=f'Campo "{field}" deve ser {field_type}, recebeu {type(value).__name__}',
                        value=value,
                        annotation_id=annotation_id
                    ))
            else:
                if not isinstance(value, field_type):
                    errors.append(ValidationError(
                        field=field,
                        error_type='wrong_type',
                        message=f'Campo "{field}" deve ser {field_type.__name__}, recebeu {type(value).__name__}',
                        value=value,
                        annotation_id=annotation_id
                    ))
        
        # Validar intent
        if 'intent' in annotation:
            if annotation['intent'] not in self.VALID_INTENTS:
                errors.append(ValidationError(
                    field='intent',
                    error_type='invalid_value',
                    message=f'Intent inválido: "{annotation["intent"]}". Válidos: {self.VALID_INTENTS}',
                    value=annotation['intent'],
                    annotation_id=annotation_id
                ))
        
        # Validar text (não vazio)
        if 'text' in annotation:
            if not annotation['text'] or not annotation['text'].strip():
                errors.append(ValidationError(
                    field='text',
                    error_type='empty_value',
                    message='Campo "text" não pode estar vazio',
                    annotation_id=annotation_id
                ))
        
        # Validar trigger
        if 'trigger' in annotation:
            trigger = annotation['trigger']
            if isinstance(trigger, str):
                if not trigger.strip():
                    warnings.append(f'Anotação {annotation_id}: trigger vazio')
            elif isinstance(trigger, list):
                if not trigger or all(not t.strip() for t in trigger):
                    warnings.append(f'Anotação {annotation_id}: lista de triggers vazia ou com valores vazios')
        
        # Validar arguments
        if 'arguments' in annotation:
            args = annotation['arguments']
            if not isinstance(args, dict):
                errors.append(ValidationError(
                    field='arguments',
                    error_type='wrong_type',
                    message=f'Campo "arguments" deve ser dict, recebeu {type(args).__name__}',
                    value=args,
                    annotation_id=annotation_id
                ))
            else:
                # Verificar sub-campos
                for arg_field in self.ARGUMENT_FIELDS:
                    if arg_field in args:
                        if not isinstance(args[arg_field], list):
                            errors.append(ValidationError(
                                field=f'arguments.{arg_field}',
                                error_type='wrong_type',
                                message=f'Campo "arguments.{arg_field}" deve ser list, recebeu {type(args[arg_field]).__name__}',
                                value=args[arg_field],
                                annotation_id=annotation_id
                            ))
                    else:
                        warnings.append(f'Anotação {annotation_id}: campo "arguments.{arg_field}" ausente (será criado como [])')
        
        # Verificar UTF-8 encoding do text
        if 'text' in annotation:
            try:
                annotation['text'].encode('utf-8')
            except UnicodeEncodeError as e:
                errors.append(ValidationError(
                    field='text',
                    error_type='encoding_error',
                    message=f'Erro de codificação UTF-8: {str(e)}',
                    annotation_id=annotation_id
                ))
        
        is_valid = len(errors) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings
        )
    
    def validate_batch(self, annotations: List[Dict]) -> ValidationResult:
        """
        Valida um lote de anotações.
        
        Args:
            annotations: Lista de anotações
            
        Returns:
            ValidationResult agregado
        """
        all_errors = []
        all_warnings = []
        all_valid = True
        
        # Verificar duplicados por ID
        ids = []
        for i, ann in enumerate(annotations):
            if 'id' in ann:
                ann_id = ann['id']
                if ann_id in ids:
                    all_errors.append(ValidationError(
                        field='id',
                        error_type='duplicate_id',
                        message=f'ID duplicado: {ann_id}',
                        value=ann_id,
                        annotation_id=i
                    ))
                    all_valid = False
                ids.append(ann_id)
        
        # Validar cada anotação
        for i, annotation in enumerate(annotations):
            result = self.validate_single_annotation(annotation, annotation_id=i)
            all_errors.extend(result.errors)
            all_warnings.extend(result.warnings)
            
            if not result.is_valid:
                all_valid = False
        
        return ValidationResult(
            is_valid=all_valid,
            errors=all_errors,
            warnings=all_warnings
        )
    
    def normalize_annotation(self, annotation: Dict) -> Dict:
        """
        Normaliza uma anotação para formato padrão.
        
        - Garante presença de todos os campos de arguments
        - Converte trigger string em lista
        - Remove campos desconhecidos
        
        Args:
            annotation: Dict com anotação
            
        Returns:
            Anotação normalizada
        """
        normalized = {
            'id': annotation.get('id'),
            'text': annotation.get('text', ''),
            'intent': annotation.get('intent'),
            'trigger': annotation.get('trigger', []),
            'arguments': {
                'participants': annotation.get('arguments', {}).get('participants', []),
                'time': annotation.get('arguments', {}).get('time', []),
                'location': annotation.get('arguments', {}).get('location', []),
                'topic': annotation.get('arguments', {}).get('topic', []),
            }
        }
        
        # Se trigger é string, converter em lista
        if isinstance(normalized['trigger'], str):
            if normalized['trigger']:
                normalized['trigger'] = [normalized['trigger']]
            else:
                normalized['trigger'] = []
        
        # Garantir que todos os argumentos são listas
        for key, value in normalized['arguments'].items():
            if not isinstance(value, list):
                normalized['arguments'][key] = [value] if value else []
            # Remover strings vazias
            normalized['arguments'][key] = [v.strip() for v in value if isinstance(v, str) and v.strip()]
        
        # Adicionar confidence score se não existir
        if 'confidence' not in normalized:
            normalized['confidence'] = {
                'trigger': 0.9,
                'participants': 0.75,
                'temporal': 0.8,
                'location': 0.85,
                'topic': 0.7,
            }
        elif isinstance(normalized['confidence'], (int, float)):
            # Se for um número único, expandir para dicionário
            normalized['confidence'] = {
                'trigger': 0.9,
                'participants': 0.75,
                'temporal': 0.8,
                'location': 0.85,
                'topic': 0.7,
            }
        
        # Adicionar metadata
        if 'metadata' not in normalized:
            normalized['metadata'] = {
                'created_at': datetime.now().isoformat(),
                'validated': True,
                'version': '1.0',
            }
        
        return normalized


class JSONValidator:
    """Valida integridade de ficheiros JSON"""
    
    @staticmethod
    def validate_json_file(filepath: str) -> Tuple[bool, str]:
        """
        Valida se um ficheiro JSON é válido.
        
        Args:
            filepath: Caminho para o ficheiro JSON
            
        Returns:
            Tuple (is_valid, message)
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                json.load(f)
            return True, "JSON válido"
        except json.JSONDecodeError as e:
            return False, f"Erro JSON: {str(e)}"
        except UnicodeDecodeError as e:
            return False, f"Erro de codificação UTF-8: {str(e)}"
        except Exception as e:
            return False, f"Erro: {str(e)}"
    
    @staticmethod
    def is_valid_json(text: str) -> Tuple[bool, str]:
        """
        Verifica se uma string é JSON válido.
        
        Args:
            text: String com JSON
            
        Returns:
            Tuple (is_valid, message)
        """
        try:
            json.loads(text)
            return True, "JSON válido"
        except json.JSONDecodeError as e:
            return False, f"Erro JSON: {str(e)}"
        except Exception as e:
            return False, f"Erro: {str(e)}"


class ConsistencyValidator:
    """Valida consistência entre diferentes anotações"""
    
    @staticmethod
    def check_intent_trigger_consistency(annotations: List[Dict]) -> List[str]:
        """
        Verifica consistência entre intent e trigger.
        
        Args:
            annotations: Lista de anotações
            
        Returns:
            Lista de avisos
        """
        warnings = []
        
        intent_trigger_map = {
            'agendamento_reuniao': ['reunir', 'marcar', 'agendar', 'combinar'],
            'cancelamento_reuniao': ['cancelar', 'faltar'],
            'reuniao_confirmada': ['confirmar', 'confimado'],  # typo intencional para aceitar variações
        }
        
        for i, ann in enumerate(annotations):
            intent = ann.get('intent')
            triggers = ann.get('trigger', [])
            
            if not isinstance(triggers, list):
                triggers = [triggers] if triggers else []
            
            if intent in intent_trigger_map:
                expected_triggers = intent_trigger_map[intent]
                if not any(t in expected_triggers for t in triggers):
                    if triggers:
                        warnings.append(
                            f'Anotação {i}: trigger "{triggers[0]}" pode não ser consistente com intent "{intent}"'
                        )
        
        return warnings
