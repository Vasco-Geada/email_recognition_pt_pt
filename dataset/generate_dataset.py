import json
import random
from faker import Faker

fake = Faker("pt_PT")

# =========================================================
# CONFIG
# =========================================================

TOTAL_EMAILS = 300

OUTPUT_PATH = "dataset/realistic_emails_v2.json"

# =========================================================
# PERSONAS
# =========================================================

PERSONAS = {
    "aluno_informal": {
        "openings": ["Boas", "Hey", "Yo", "Olá"],
        "closings": ["Abraço", "Até já", "Cumps", "Bjs", ""],
        "emoji_prob": 0.25,
        "short_prob": 0.35
    },

    "aluno_stressado": {
        "openings": ["Boas", "Hey"],
        "closings": ["", "Abraço"],
        "emoji_prob": 0.10,
        "short_prob": 0.60
    },

    "professor": {
        "openings": ["Bom dia", "Boa tarde", "Viva"],
        "closings": ["Cumprimentos", "Obrigado"],
        "emoji_prob": 0.01,
        "short_prob": 0.10
    },

    "aluno_internacional": {
        "openings": ["Hello", "Hi", "Boas"],
        "closings": ["Thanks", "Best regards"],
        "emoji_prob": 0.10,
        "short_prob": 0.25
    }
}

# =========================================================
# DATA
# =========================================================

NAMES = [
    "Ana", "João", "Maria", "Pedro", "Tiago",
    "Sofia", "Rita", "Miguel", "Ricardo",
    "Diana", "Bruno", "Inês", "Marta",
    "André", "Carolina"
]

PROFESSORS = [
    "Professor Silva",
    "Professora Ana Costa",
    "Professor Ricardo Mendes"
]

TOPICS = [
    "dissertação",
    "pipeline NLP",
    "dataset",
    "baseline",
    "BERT",
    "apresentação",
    "relatório",
    "artigo",
    "resultados",
    "métricas F1",
    "deadline",
    "experiências",
    "revisão bibliográfica",
    "trabalho de grupo",
    "Visualização de Dados",
    "Segurança Informática"
]

LOCATIONS = [
    "Teams",
    "Zoom",
    "Discord",
    "sala 2.3",
    "biblioteca",
    "laboratório",
    "gabinete",
    "campus",
    "bar da faculdade"
]

TEMPORAL_EXPRESSIONS = [
    "amanhã",
    "sexta",
    "segunda",
    "terça",
    "quarta à tarde",
    "quinta de manhã",
    "amanhã às 15h",
    "sexta às 16h",
    "pelas 14h",
    "ao final do dia",
    "depois de almoço",
    "mais logo",
    "na próxima semana",
    "daqui a 2 dias",
    "antes da aula",
    "depois do laboratório",
    "após o seminário",
    "entre cadeiras"
]

EMOJIS = [
    "👍",
    "😅",
    "😂",
    "👌",
    "🔥"
]

REAL_EXPRESSIONS = [
    "logo se vê",
    "tranquilo",
    "na boa",
    "isso resolve-se",
    "vemos isso amanhã",
    "depois falamos"
]

THREAD_PREFIXES = [
    "",
    "Re: ",
    "RE: ",
    "Fw: "
]

NOISE = [
    "",
    "\nSent from my iPhone",
    "\nEnviado do Outlook",
    "\nEnviado do telemóvel",
    "\nDISCLAIMER: mensagem automática"
]

# =========================================================
# TYPOS
# =========================================================

TYPO_MAP = {
    "amanhã": "amanha",
    "reunião": "reuniao",
    "também": "tb",
    "porque": "pq",
    "está": "ta",
    "você": "voce"
}

# =========================================================
# SUBJECTS
# =========================================================

SUBJECTS = {
    "agendamento_reuniao": [
        "Reunião",
        "Call",
        "Projeto",
        "Dissertação",
        "Meeting",
        "Precisamos de falar",
        "Discussão resultados"
    ],

    "cancelamento_reuniao": [
        "Cancelamento",
        "Mudança de planos",
        "Imprevisto",
        "Não vou conseguir",
        "Adiar reunião"
    ],

    "reuniao_confirmada": [
        "Confirmado",
        "Tudo certo",
        "Combinado",
        "Reunião confirmada"
    ]
}

# =========================================================
# TEMPLATES
# =========================================================

AGENDAMENTO = [
    "{opening} {name}, podemos reunir {time} para falar sobre {topic}?",

    "{opening} {name}, tens disponibilidade {time}?",

    "Precisamos de discutir o {topic}. Dá para reunir {time}?",

    "Podemos fazer uma quick call {time}?",

    "{name}, consegues aparecer no {location} {time}?",

    "Temos de alinhar a parte do {topic}.",

    "Quando puderes temos de combinar uma reunião.",

    "Se calhar era melhor reunir {time}.",

    "Bora fazer uma reunião rápida {time}?",

    "Consegues reunir depois da aula?"
]

CANCELAMENTO = [
    "{opening} {name}, afinal vou ter de cancelar a reunião de {time}.",

    "Não vou conseguir aparecer {time}.",

    "A reunião fica sem efeito.",

    "Surgiu um imprevisto.",

    "Afinal já não consigo reunir.",

    "Vou ter aula nesse horário.",

    "Peço desculpa mas temos de adiar.",

    "Não vou conseguir ir ao {location}.",

    "A reunião sobre {topic} vai ter de ficar para outro dia.",

    "Hoje não vai dar 😅"
]

CONFIRMACAO = [
    "Perfeito, fica então combinado para {time}.",

    "Confirmado 👍",

    "Tudo certo para {time}.",

    "Fechado então.",

    "Ok, parece-me bem.",

    "Combinado, vemos isso {time}.",

    "Eu apareço no {location}.",

    "Está ótimo para mim.",

    "Na boa 👍",

    "Pode ser."
]

SHORT_REPLIES = [
    "Pode ser.",
    "Confirmado.",
    "Ok 👍",
    "Na boa.",
    "Tranquilo.",
    "Combinado.",
    "Logo vemos.",
    "Depois falamos.",
    "15h é complicado.",
    "Acho que sexta já dá."
]

# =========================================================
# HELPERS
# =========================================================

def apply_typos(text):

    for correct, wrong in TYPO_MAP.items():

        if correct in text and random.random() < 0.15:
            text = text.replace(correct, wrong)

    return text


def maybe_add_emoji(text, probability):

    if random.random() < probability:
        text += " " + random.choice(EMOJIS)

    return text


def generate_signature(persona):

    return random.choice(
        PERSONAS[persona]["closings"]
    )


def generate_subject(label):

    prefix = random.choice(THREAD_PREFIXES)

    return prefix + random.choice(
        SUBJECTS[label]
    )


def generate_persona():

    return random.choice(
        list(PERSONAS.keys())
    )


def maybe_make_short(persona):

    return (
        random.random()
        < PERSONAS[persona]["short_prob"]
    )


# =========================================================
# EMAIL BUILDER
# =========================================================

def build_email(label):

    persona = generate_persona()

    opening = random.choice(
        PERSONAS[persona]["openings"]
    )

    name = random.choice(
        NAMES + PROFESSORS
    )

    topic = random.choice(TOPICS)

    location = random.choice(LOCATIONS)

    time = random.choice(TEMPORAL_EXPRESSIONS)

    if label == "agendamento_reuniao":
        template = random.choice(AGENDAMENTO)

    elif label == "cancelamento_reuniao":
        template = random.choice(CANCELAMENTO)

    else:
        template = random.choice(CONFIRMACAO)

    # emails curtos
    if maybe_make_short(persona):
        body = random.choice(SHORT_REPLIES)

    else:

        body = template.format(
            opening=opening,
            name=name,
            topic=topic,
            location=location,
            time=time
        )

    # expressão real ocasional
    if random.random() < 0.20:
        body += f" {random.choice(REAL_EXPRESSIONS)}"

    # emoji
    body = maybe_add_emoji(
        body,
        PERSONAS[persona]["emoji_prob"]
    )

    # typos
    body = apply_typos(body)

    # assinatura
    signature = generate_signature(persona)

    # ruído
    noise = random.choice(NOISE)

    # multiline realism
    if random.random() < 0.25:
        body = body.replace(". ", ".\n")

    final_body = f"{body}\n\n{signature}{noise}"

    return {
        "subject": generate_subject(label),
        "body": final_body.strip(),
        "label": label,
        "persona": persona
    }

# =========================================================
# DATASET GENERATION
# =========================================================

emails = []

for _ in range(120):
    emails.append(
        build_email("agendamento_reuniao")
    )

for _ in range(90):
    emails.append(
        build_email("cancelamento_reuniao")
    )

for _ in range(90):
    emails.append(
        build_email("reuniao_confirmada")
    )

random.shuffle(emails)

# =========================================================
# SAVE
# =========================================================

with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        emails,
        f,
        ensure_ascii=False,
        indent=2
    )

print(f"{len(emails)} emails gerados com sucesso!")
print(f"Guardado em: {OUTPUT_PATH}")