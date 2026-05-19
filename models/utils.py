# -*- coding: utf-8 -*-
"""
Funções Auxiliares para o Sistema de Classificação de Emails.

Este módulo fornece funções compartilhadas para:
- Carregamento de datasets
- Pré-processamento de textos
- Combinação de campos de email
- Limpeza de texto

Funções:
    load_dataset: Carrega dataset em JSON.
    preprocess_text: Pré-processa um texto individual.
    preprocess_texts: Pré-processa múltiplos textos.
    combine_text_fields: Combina subject e body.
    remove_email_signatures: Remove assinaturas.
    remove_threads: Remove threads de resposta.
"""

import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Optional, Union

# Configurar logging
logger = logging.getLogger(__name__)


# Constantes
PORTUGUESE_STOPWORDS = {
    'a', 'à', 'o', 'os', 'as', 'um', 'uma', 'uns', 'umas',
    'de', 'do', 'da', 'dos', 'das', 'e', 'ou', 'é', 'são',
    'em', 'com', 'para', 'por', 'que', 'se', 'não', 'sim',
    'este', 'esse', 'aquele', 'meu', 'teu', 'seu', 'nosso',
    'vosso', 'isso', 'isto', 'aquilo', 'ele', 'ela', 'eles',
    'elas', 'nós', 'vós', 'mim', 'ti', 'lhe', 'me', 'te',
    'nos', 'vos', 'foi', 'era', 'era', 'ser', 'estar', 'ir',
    'fazer', 'dar', 'dizer', 'estar', 'tem', 'há', 'todo',
    'muito', 'pouco', 'certo', 'mais', 'menos', 'bem', 'mal',
    'aqui', 'aí', 'lá', 'ali', 'cá', 'agora', 'hoje', 'ontem'
}

EMAIL_SIGNATURES = [
    r'Cumprimentos.*',
    r'Melhores cumprimentos.*',
    r'Obrigado.*',
    r'Com os melhores cumprimentos.*',
    r'Atenciosamente.*',
    r'Atentamente.*',
    r'Um abraço.*',
    r'Abraços.*',
]

EMAIL_THREADS = [
    r'On .* wrote:',
    r'Subject:.*',
    r'From:.*',
    r'To:.*',
    r'De:.*',
    r'Para:.*',
    r'Assunto:.*',
    r'-----Original Message-----',
    r'-------- Mensagem original --------',
    r'>.*',
]


def load_dataset(json_path: str) -> List[Dict]:
    """
    Carrega dataset em formato JSON.
    
    Args:
        json_path: Caminho do ficheiro JSON.
    
    Returns:
        Lista de dicionários com emails.
    
    Raises:
        FileNotFoundError: Se ficheiro não existe.
        json.JSONDecodeError: Se JSON inválido.
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            data = [data]
        
        logger.info(f"Dataset carregado: {len(data)} emails")
        return data
        
    except FileNotFoundError:
        logger.error(f"Ficheiro não encontrado: {json_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Erro ao decodificar JSON: {str(e)}")
        raise


def remove_email_signatures(text: str) -> str:
    """
    Remove assinaturas comuns de emails.
    
    Args:
        text: Texto do email.
    
    Returns:
        Texto sem assinatura.
    """
    for pattern in EMAIL_SIGNATURES:
        text = re.split(pattern, text, flags=re.IGNORECASE)[0]
    
    return text.strip()


def remove_threads(text: str) -> str:
    """
    Remove threads de resposta (histórico de email).
    
    Args:
        text: Texto do email.
    
    Returns:
        Texto sem threads.
    """
    for pattern in EMAIL_THREADS:
        text = re.split(pattern, text, flags=re.IGNORECASE)[0]
    
    return text.strip()


def normalize_text(text: str) -> str:
    """
    Normaliza texto base.
    
    Args:
        text: Texto a normalizar.
    
    Returns:
        Texto normalizado.
    """
    # Converter para string se necessário
    if not isinstance(text, str):
        text = str(text)
    
    # Strip
    text = text.strip()
    
    # Normalizar espaços
    text = re.sub(r'\s+', ' ', text)
    
    # Remover caracteres especiais mas manter acentos e pontuação básica
    text = re.sub(r'[^\w\s.,!?@:/\-à-úÀ-Úç]', '', text)
    
    return text


def clean_text(
    text: str,
    remove_punctuation: bool = False,
    remove_digits: bool = False,
    remove_stopwords: bool = False,
    lowercase: bool = True
) -> str:
    """
    Limpeza completa de texto.
    
    Args:
        text: Texto a limpar.
        remove_punctuation: Remove pontuação.
        remove_digits: Remove dígitos.
        remove_stopwords: Remove stopwords PT.
        lowercase: Converte para minúsculas.
    
    Returns:
        Texto limpo.
    """
    # Normalizar
    text = normalize_text(text)
    
    # Lowercase
    if lowercase:
        text = text.lower()
    
    # Remover pontuação
    if remove_punctuation:
        text = re.sub(r'[.,!?;:-]', '', text)
    
    # Remover dígitos
    if remove_digits:
        text = re.sub(r'\d+', '', text)
    
    # Remover stopwords
    if remove_stopwords:
        words = text.split()
        words = [w for w in words if w.lower() not in PORTUGUESE_STOPWORDS]
        text = ' '.join(words)
    
    # Normalizar espaços novamente
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def preprocess_text(
    text: str,
    remove_signatures: bool = True,
    remove_threads_history: bool = True,
    remove_punctuation: bool = False,
    remove_stopwords: bool = False,
    lowercase: bool = True
) -> str:
    """
    Pré-processa um texto individual.
    
    Args:
        text: Texto a pré-processar.
        remove_signatures: Remove assinaturas.
        remove_threads_history: Remove threads.
        remove_punctuation: Remove pontuação.
        remove_stopwords: Remove stopwords.
        lowercase: Lowercase.
    
    Returns:
        Texto pré-processado.
    """
    # Remover threads
    if remove_threads_history:
        text = remove_threads(text)
    
    # Remover assinaturas
    if remove_signatures:
        text = remove_email_signatures(text)
    
    # Limpar
    text = clean_text(
        text,
        remove_punctuation=remove_punctuation,
        remove_stopwords=remove_stopwords,
        lowercase=lowercase
    )
    
    return text


def preprocess_texts(
    texts: List[str],
    remove_signatures: bool = True,
    remove_threads_history: bool = True,
    remove_punctuation: bool = False,
    remove_stopwords: bool = False,
    lowercase: bool = True
) -> List[str]:
    """
    Pré-processa múltiplos textos.
    
    Args:
        texts: Lista de textos.
        remove_signatures: Remove assinaturas.
        remove_threads_history: Remove threads.
        remove_punctuation: Remove pontuação.
        remove_stopwords: Remove stopwords.
        lowercase: Lowercase.
    
    Returns:
        Lista de textos pré-processados.
    """
    processed = []
    
    for idx, text in enumerate(texts):
        try:
            processed_text = preprocess_text(
                text,
                remove_signatures=remove_signatures,
                remove_threads_history=remove_threads_history,
                remove_punctuation=remove_punctuation,
                remove_stopwords=remove_stopwords,
                lowercase=lowercase
            )
            processed.append(processed_text)
        except Exception as e:
            logger.warning(f"Erro ao processar texto {idx}: {str(e)}")
            processed.append("")
    
    return processed


def combine_text_fields(
    email_dict: Dict,
    subject_weight: float = 1.0,
    body_weight: float = 2.0
) -> str:
    """
    Combina subject e body de um email.
    
    Args:
        email_dict: Dicionário com campos email.
        subject_weight: Peso do subject (repetições).
        body_weight: Peso do body (repetições).
    
    Returns:
        Texto combinado.
    """
    parts = []
    
    # Subject
    subject = email_dict.get('subject', '')
    if subject:
        subject_text = str(subject).strip()
        # Repetir subject conforme peso
        parts.extend([subject_text] * max(1, int(subject_weight)))
    
    # Body
    body = email_dict.get('body', '')
    if body:
        body_text = str(body).strip()
        # Repetir body conforme peso
        parts.extend([body_text] * max(1, int(body_weight)))
    
    # Combinar
    combined = ' '.join(parts).strip()
    
    return combined


def get_class_distribution(
    labels: List[str],
    verbose: bool = True
) -> Dict[str, Dict]:
    """
    Calcula distribuição de classes.
    
    Args:
        labels: Lista de labels.
        verbose: Se imprime resultado.
    
    Returns:
        Dicionário com distribuição.
    """
    unique_labels = set(labels)
    distribution = {}
    total = len(labels)
    
    for label in sorted(unique_labels):
        count = labels.count(label)
        percentage = (count / total) * 100
        distribution[label] = {
            'count': count,
            'percentage': percentage
        }
        
        if verbose:
            logger.info(f"{label}: {count} ({percentage:.1f}%)")
    
    return distribution


def validate_dataset(json_path: str) -> Dict:
    """
    Valida integridade do dataset.
    
    Args:
        json_path: Caminho do dataset.
    
    Returns:
        Dicionário com informações de validação.
    """
    logger.info(f"Validando dataset: {json_path}")
    
    try:
        emails = load_dataset(json_path)
        
        validation = {
            'total_emails': len(emails),
            'has_errors': False,
            'errors': [],
            'warnings': [],
            'missing_fields': {}
        }
        
        # Verificar cada email
        for idx, email in enumerate(emails):
            if not isinstance(email, dict):
                validation['has_errors'] = True
                validation['errors'].append(
                    f"Email {idx}: tipo inválido (esperado dict, obtido {type(email)})"
                )
                continue
            
            # Campos obrigatórios
            if 'label' not in email:
                if 'label' not in validation['missing_fields']:
                    validation['missing_fields']['label'] = 0
                validation['missing_fields']['label'] += 1
            
            if not email.get('body'):
                validation['warnings'].append(
                    f"Email {idx}: body vazio"
                )
        
        if validation['errors']:
            validation['has_errors'] = True
        
        logger.info(f"✓ Validação completa: {validation['total_emails']} emails")
        if validation['errors']:
            logger.warning(f"  Erros encontrados: {len(validation['errors'])}")
        if validation['warnings']:
            logger.warning(f"  Avisos: {len(validation['warnings'])}")
        
        return validation
        
    except Exception as e:
        logger.error(f"Erro ao validar dataset: {str(e)}")
        raise


def save_predictions_to_json(
    predictions: List[Dict],
    output_path: str
) -> None:
    """
    Guarda predições em ficheiro JSON.
    
    Args:
        predictions: Lista de predições.
        output_path: Caminho de saída.
    """
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✓ Predições guardadas em: {output_path}")
        
    except Exception as e:
        logger.error(f"Erro ao guardar predições: {str(e)}")
        raise
