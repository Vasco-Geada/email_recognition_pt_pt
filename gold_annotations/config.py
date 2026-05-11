"""
config.py
=========

Ficheiro de configuração para o sistema de gold annotations.

Define padrões, constantes e configurações globais.

Author: Generated for Email Recognition PT-PT Project
"""

# ============================================================================
# CONFIGURAÇÃO DE TRIGGERS
# ============================================================================

TRIGGER_PATTERNS = {
    'reunir': [
        r'\breunir(?:mos|em|ão|ia)?\b',
        r'\breunião\b',
        r'\buma\s+reunião\b',
    ],
    'marcar': [
        r'\bmarcar(?:\s+(?:uma|a|o))?(?:\s+reunião)?\b',
        r'\bmarcado\b',
    ],
    'cancelar': [
        r'\bcancelar(?:\s+(?:a|o|uma))?(?:\s+reunião)?\b',
        r'\bcancelamento\b',
        r'\bfaltar(?:\s+à|ao)?(?:\s+reunião)?\b',
    ],
    'confirmar': [
        r'\bconfirmar\b',
        r'\bconfirmado\b',
    ],
    'agendar': [
        r'\bagendar\b',
        r'\bagendado\b',
    ],
    'combinar': [
        r'\bcombinar\b',
        r'\bcombinado\b',
    ],
    'encontrar': [
        r'\bencontrar(?:\s+(?:com|a))?(?:\s+se)?\b',
        r'\bencontro\b',
    ],
    'falar': [
        r'\bfalar(?:\s+(?:com|sobre))?\b',
        r'\bconversa(?:r|ção)?\b',
    ],
    'discutir': [
        r'\bdiscutir\b',
        r'\bdiscussão\b',
    ],
}

# ============================================================================
# CONFIGURAÇÃO DE LOCALIZAÇÕES
# ============================================================================

LOCATION_PATTERNS = {
    'plataformas_online': [
        r'\b(?:MS\s+)?Teams\b',
        r'\bZoom\b',
        r'\bDiscord\b',
        r'\bSkype\b',
        r'\bGoogle\s+Meet\b',
    ],
    'salas': [
        r'\bsala\s+(?:\d+[.-]?\d*|[A-Z]\d+)',
        r'\bbloco\s+[A-Z0-9]+',
    ],
    'espacos': [
        r'\bbiblioteca\b',
        r'\blaboratório\b',
        r'\blab\b',
        r'\boficina\b',
    ],
}

# ============================================================================
# CONFIGURAÇÃO DE EXPRESSÕES TEMPORAIS
# ============================================================================

TEMPORAL_PATTERNS = {
    'dias_relativos': [
        r'\b(?:amanhã|hoje|ontem)\b',
        r'\b(?:segunda|terça|quarta|quinta|sexta|sábado|domingo)(?:-feira)?\b',
    ],
    'semanas': [
        r'\b(?:próxima|semana\s+que\s+vem)\s+semana\b',
    ],
    'horas': [
        r'(?:às|ao|a)\s*(\d{1,2})(?:\s*[:h])?\s*(\d{2})?\s*(?:h|horas)?\b',
        r'(\d{1,2}):(\d{2})',
    ],
    'periodos': [
        r'\b(?:de\s+)?(?:manhã|manha)\b',
        r'\b(?:à\s+)?(?:tarde|à tarde)\b',
        r'\b(?:à\s+)?(?:noite|à noite)\b',
        r'\b(?:depois\s+de\s+)?almoço\b',
    ],
}

# ============================================================================
# CONFIGURAÇÃO DE TÓPICOS
# ============================================================================

TOPIC_PATTERNS = {
    'trabalhos_academicos': [
        r'\bdissertação\b',
        r'\btese\b',
        r'\brelatório\b',
        r'\btrabalho\b',
        r'\bprojeto\b',
    ],
    'conceitos_tecn': [
        r'\b(?:pipeline|NLP|IA|AI)\b',
        r'\b(?:BERT|GPT|RNN|LSTM)\b',
        r'\b(?:dataset|corpus|embeddings?)\b',
        r'\b(?:precisão|recall|F1|accuracy)\b',
    ],
}

# ============================================================================
# CONFIGURAÇÃO DE VALIDAÇÃO
# ============================================================================

VALID_INTENTS = [
    'agendamento_reuniao',
    'cancelamento_reuniao',
    'reuniao_confirmada',
]

REQUIRED_FIELDS = {
    'id': int,
    'text': str,
    'intent': str,
    'trigger': (list, str),
    'arguments': dict,
}

ARGUMENT_FIELDS = [
    'participants',
    'time',
    'location',
    'topic',
]

# ============================================================================
# CONFIGURAÇÃO DE CONFIDENCE
# ============================================================================

DEFAULT_CONFIDENCE = {
    'trigger': 0.90,
    'participants': 0.75,
    'temporal': 0.80,
    'location': 0.85,
    'topic': 0.70,
}

# ============================================================================
# CONFIGURAÇÃO DE OUTPUT
# ============================================================================

OUTPUT_CONFIG = {
    'json_indent': 2,
    'ensure_ascii': False,
    'encoding': 'utf-8',
}

# ============================================================================
# CONFIGURAÇÃO DE LOGGING
# ============================================================================

LOG_LEVELS = {
    'DEBUG': 0,
    'INFO': 1,
    'WARNING': 2,
    'ERROR': 3,
}

DEFAULT_LOG_LEVEL = 'INFO'

# ============================================================================
# CONFIGURAÇÃO DE PERFORMANCE
# ============================================================================

BATCH_SIZE = 100  # Processar em lotes de 100
REPORT_INTERVAL = 10  # Mostrar progresso a cada 10 items

# ============================================================================
# TÍTULOS E PREFIXOS
# ============================================================================

TITLES = [
    'Professor', 'Professora', 'Prof.', 'Profa.',
    'Dr.', 'Dra.', 'Doutor', 'Doutora',
    'Eng.', 'Engenheiro', 'Engenheira',
    'Sr.', 'Sra.', 'Senhor', 'Senhora',
]

# ============================================================================
# PARTICIPANTES COMUNS
# ============================================================================

COMMON_PARTICIPANTS = {
    'João', 'Maria', 'Ana', 'Pedro', 'Silva',
    'Costa', 'Rodrigues', 'Santos', 'Oliveira',
}

# ============================================================================
# PALAVRAS COMUNS A EXCLUIR (FILTRAGEM)
# ============================================================================

COMMON_WORDS = {
    'o', 'a', 'os', 'as', 'um', 'uma', 'uns', 'umas',
    'e', 'ou', 'mas', 'porém', 'contudo',
    'de', 'em', 'por', 'para', 'com', 'sem',
    'que', 'qual', 'quanto', 'quando', 'onde',
}

# ============================================================================
# CONFIGURAÇÕES DE TESTE
# ============================================================================

TEST_CONFIG = {
    'sample_size': 5,
    'timeout_seconds': 30,
    'max_errors_to_report': 10,
}

# ============================================================================
# FUNÇÃO HELPER PARA CARREGAR CONFIG
# ============================================================================

def get_config(key: str, default=None):
    """
    Getter para valores de configuração.
    
    Args:
        key: Chave da configuração (e.g., 'DEFAULT_LOG_LEVEL')
        default: Valor padrão se chave não existir
        
    Returns:
        Valor de configuração
    """
    return globals().get(key, default)


def set_config(key: str, value):
    """
    Setter para valores de configuração.
    
    Args:
        key: Chave da configuração
        value: Novo valor
    """
    globals()[key] = value


if __name__ == '__main__':
    # Teste da configuração
    print("Configuração do Sistema de Gold Annotations")
    print("=" * 50)
    print(f"Intents válidos: {VALID_INTENTS}")
    print(f"Triggers: {list(TRIGGER_PATTERNS.keys())}")
    print(f"Locations: {list(LOCATION_PATTERNS.keys())}")
    print(f"Topics: {list(TOPIC_PATTERNS.keys())}")
    print(f"Argument fields: {ARGUMENT_FIELDS}")
    print(f"Default confidence: {DEFAULT_CONFIDENCE}")
