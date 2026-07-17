import argparse
import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional


DEFAULT_OUTPUT = "dataset/realistic_emails_v3.json"

LABEL_COUNTS = {
    "agendamento_reuniao": 3000,
    "cancelamento_reuniao": 3000,
    "reuniao_confirmada": 3000,
    "nao_reuniao": 1000,
}

PEOPLE = [
    {"name": "Ana Martins", "email": "ana.martins@universidade.pt", "role": "aluno"},
    {"name": "Joao Pereira", "email": "joao.pereira@universidade.pt", "role": "aluno"},
    {"name": "Mariana Lopes", "email": "mariana.lopes@universidade.pt", "role": "aluno"},
    {"name": "Rita Fernandes", "email": "rita.fernandes@universidade.pt", "role": "aluno"},
    {"name": "Miguel Santos", "email": "miguel.santos@universidade.pt", "role": "aluno"},
    {"name": "Carolina Silva", "email": "carolina.silva@universidade.pt", "role": "aluno"},
    {"name": "Prof. Ricardo Mendes", "email": "ricardo.mendes@universidade.pt", "role": "professor"},
    {"name": "Professora Ana Costa", "email": "ana.costa@universidade.pt", "role": "professor"},
    {"name": "Professor Luis Almeida", "email": "luis.almeida@universidade.pt", "role": "professor"},
    {"name": "Professora Sofia Carvalho", "email": "sofia.carvalho@universidade.pt", "role": "professor"},
    {"name": "Diretor do Curso", "email": "direcao.curso@universidade.pt", "role": "direcao"},
    {"name": "Conselho Pedagogico", "email": "pedagogico@universidade.pt", "role": "orgao_gestao"},
    {"name": "Secretaria Academica", "email": "secretaria@universidade.pt", "role": "secretaria"},
    {"name": "Gabinete de Estagios", "email": "estagios@universidade.pt", "role": "orgao_gestao"},
    {"name": "Coordenacao de Mestrado", "email": "mestrado@universidade.pt", "role": "direcao"},
]

ROLE_PAIRS = [
    ("aluno", "professor"),
    ("professor", "aluno"),
    ("professor", "professor"),
    ("professor", "direcao"),
    ("direcao", "professor"),
    ("professor", "secretaria"),
    ("secretaria", "professor"),
    ("professor", "orgao_gestao"),
    ("orgao_gestao", "professor"),
    ("aluno", "secretaria"),
    ("secretaria", "aluno"),
]

TOPICS = [
    "orientacao da dissertacao",
    "revisao do plano de trabalhos",
    "calendario de avaliacoes",
    "preparacao da apresentacao",
    "distribuicao de servico docente",
    "validacao de atas",
    "analise dos resultados dos estudantes",
    "processo de estagio",
    "credibilidade dos dados recolhidos",
    "relatorio de progresso",
    "proposta de projeto",
    "submissao de artigo",
    "horarios do semestre",
    "coordenacao da unidade curricular",
    "pedido de equivalencias",
    "acompanhamento pedagogico",
    "avaliacao continua",
    "planeamento de aulas",
    "revisao bibliografica",
    "preparacao da defesa",
]

LOCATIONS = [
    "Remoto",
    "Presencial",
]

TIMES = [
    "amanha as 10h",
    "amanha as 15h",
    "sexta as 14h30",
    "segunda de manha",
    "terca as 11h",
    "quarta a tarde",
    "quinta as 16h",
    "na proxima semana",
    "depois da aula",
    "antes do seminario",
    "ao final do dia",
    "durante a manha",
    "depois de almoco",
    "dia 18 as 9h30",
    "dia 22 as 17h",
]

OPENINGS = {
    "aluno": ["Bom dia", "Boa tarde", "Ola", "Caro Professor", "Cara Professora"],
    "professor": ["Bom dia", "Boa tarde", "Caro colega", "Cara colega", "Viva"],
    "direcao": ["Exmo. Professor", "Cara Professora", "Bom dia", "Boa tarde"],
    "secretaria": ["Bom dia", "Boa tarde", "Caro/a docente", "Estimado/a estudante"],
    "orgao_gestao": ["Bom dia", "Boa tarde", "Caro/a colega", "Exmo./a Senhor/a"],
}

CLOSINGS = {
    "aluno": ["Obrigado", "Obrigada", "Cumprimentos", "Com os melhores cumprimentos"],
    "professor": ["Cumprimentos", "Obrigado", "Com os melhores cumprimentos"],
    "direcao": ["Cumprimentos", "Com os melhores cumprimentos"],
    "secretaria": ["Cumprimentos", "Com os melhores cumprimentos"],
    "orgao_gestao": ["Cumprimentos", "Com os melhores cumprimentos"],
}

SUBJECTS = {
    "agendamento_reuniao": [
        "Pedido de reuniao",
        "Marcacao de reuniao",
        "Remarcacao de reuniao",
        "Necessidade de remarcar",
        "Disponibilidade para reuniao",
        "Reuniao sobre {topic}",
        "Agendamento - {topic}",
        "Ponto de situacao",
    ],
    "cancelamento_reuniao": [
        "Cancelamento de reuniao",
        "Imprevisto - {topic}",
        "Reuniao sem efeito",
        "Indisponibilidade para a reuniao",
    ],
    "reuniao_confirmada": [
        "Reuniao confirmada",
        "Confirmacao - {topic}",
        "Tudo certo para a reuniao",
        "Confirmado",
        "Re: Marcacao de reuniao",
    ],
    "nao_reuniao": [
        "Envio de documentos",
        "Informacao academica",
        "Pedido de esclarecimento",
        "Atualizacao de processo",
        "Entrega de relatorio",
        "Aviso aos estudantes",
        "Documentacao em falta",
    ],
}

AGENDAMENTO_TEMPLATES = [
    "{opening} {recipient_name},\n\nGostaria de saber se tem disponibilidade para reunirmos {time}, em {location}, para falarmos sobre {topic}.",
    "{opening} {recipient_name},\n\nPodemos marcar uma reuniao {time} para alinharmos os proximos passos relativos a {topic}?",
    "{opening} {recipient_name},\n\nPodemos remarcar a reuniao sobre {topic} para {time}?",
    "{opening},\n\nGostaria de propor uma remarcacao da reuniao sobre {topic} para {time}.",
    "{opening} {recipient_name},\n\nConseguimos reagendar a reuniao sobre {topic} para {time}, em {location}?",
    "{opening},\n\nSeria possivel agendar uma breve reuniao {time}? O objetivo e discutir {topic}.",
    "{opening} {recipient_name},\n\nPreciso de esclarecer alguns pontos sobre {topic}. Consegue reunir {time}?",
    "{opening},\n\nProponho fazermos uma reuniao {time}, em {location}, para fechar o ponto de situacao de {topic}.",
    "{opening} {recipient_name},\n\nQuando tiver disponibilidade, gostava de combinar uma reuniao sobre {topic}. {time} seria possivel?",
    "{opening},\n\nPodemos falar {time} por {location}? Queria rever consigo o tema de {topic}.",
]

CANCELAMENTO_TEMPLATES = [
    "{opening} {recipient_name},\n\nInfelizmente tenho de cancelar a reuniao prevista para {time} sobre {topic}.",
    "{opening},\n\nPor motivo imprevisto, a reuniao de {time} fica sem efeito.",
    "{opening} {recipient_name},\n\nNao vou conseguir estar presente na reuniao em {location}, pelo que terei de cancelar.",
    "{opening},\n\nPeço desculpa, mas terei de desmarcar a reuniao relacionada com {topic}.",
    "{opening} {recipient_name},\n\nSurgiu uma sobreposicao no horario e nao conseguirei participar na reuniao {time}.",
    "{opening},\n\nA reuniao marcada para {time}, em {location}, tera de ser adiada.",
    "{opening} {recipient_name},\n\nPor indicacao da direcao, a reuniao sobre {topic} fica cancelada ate nova comunicacao.",
]

CONFIRMACAO_TEMPLATES = [
    "{opening} {recipient_name},\n\nConfirmo a reuniao {time}, em {location}, para discutirmos {topic}.",
    "{opening},\n\nFica entao combinado para {time}.",
    "{opening} {recipient_name},\n\nDa minha parte esta confirmado. Encontramo-nos em {location}.",
    "{opening},\n\nConfirmado para {time}. Levarei os documentos relativos a {topic}.",
    "{opening} {recipient_name},\n\nTudo certo para a reuniao sobre {topic}. Ate {time}.",
    "{opening},\n\nConfirmo a minha presenca na reuniao em {location}.",
    "{opening} {recipient_name},\n\nPode ficar marcado conforme proposto. Obrigado pela confirmacao.",
]

NAO_REUNIAO_TEMPLATES = [
    "{opening} {recipient_name},\n\nEnvio em anexo o documento atualizado relativo a {topic}.",
    "{opening},\n\nInformamos que o prazo para submissao dos elementos de avaliacao termina esta sexta-feira.",
    "{opening} {recipient_name},\n\nSegue a informacao solicitada pela secretaria sobre {topic}.",
    "{opening},\n\nA pauta foi atualizada no sistema academico. Por favor confirme se consegue aceder.",
    "{opening} {recipient_name},\n\nAgradeco o envio do relatorio. Irei analisar e dar feedback assim que possivel.",
    "{opening},\n\nRelembro que a aula de amanha sera dedicada a exercicios praticos.",
    "{opening} {recipient_name},\n\nO pedido de equivalencia foi encaminhado para apreciacao do conselho cientifico.",
    "{opening},\n\nPartilho a versao revista da ata para validacao.",
    "{opening} {recipient_name},\n\nAinda falta entregar a declaracao de presenca no portal academico.",
    "{opening},\n\nA bibliografia recomendada para esta semana ja se encontra disponivel no Moodle.",
]

THREAD_PREFIXES = ["", "Re: ", "RE: ", "Fw: "]
NOISE = ["", "\n\nEnviado do Outlook", "\n\nEnviado do telemovel", "\n\n--\nMensagem enviada automaticamente"]

SENT_DATETIME_START = datetime(2026, 2, 2, 8, 0)
SENT_DATETIME_DAYS = 120
SENT_HOURS = list(range(8, 19))
SENT_MINUTES = [0, 5, 10, 15, 20, 30, 40, 45, 50]


def people_by_role(role: str) -> List[Dict[str, str]]:
    return [person for person in PEOPLE if person["role"] == role]


def choose_sender_recipient() -> tuple[Dict[str, str], Dict[str, str]]:
    sender_role, recipient_role = random.choice(ROLE_PAIRS)
    sender = random.choice(people_by_role(sender_role))
    recipient = random.choice(people_by_role(recipient_role))
    if sender["email"] == recipient["email"]:
        return choose_sender_recipient()
    return sender, recipient


def maybe(value: str, probability: float) -> Optional[str]:
    return value if random.random() < probability else None


def format_subject(label: str, topic: str) -> str:
    subject = random.choice(SUBJECTS[label]).format(topic=topic)
    return random.choice(THREAD_PREFIXES) + subject


def random_sent_datetime() -> str:
    sent_date = SENT_DATETIME_START + timedelta(days=random.randint(0, SENT_DATETIME_DAYS))
    sent_date = sent_date.replace(
        hour=random.choice(SENT_HOURS),
        minute=random.choice(SENT_MINUTES),
        second=0,
        microsecond=0,
    )
    return sent_date.isoformat()


def build_email(label: str, email_id: int) -> Dict[str, object]:
    sender, recipient = choose_sender_recipient()
    topic = random.choice(TOPICS)
    location = random.choice(LOCATIONS)
    time_expression = random.choice(TIMES)
    opening = random.choice(OPENINGS[sender["role"]])
    closing = random.choice(CLOSINGS[sender["role"]])

    if label == "agendamento_reuniao":
        template = random.choice(AGENDAMENTO_TEMPLATES)
        include_location = random.random() < 0.72
        include_time = random.random() < 0.86
    elif label == "cancelamento_reuniao":
        template = random.choice(CANCELAMENTO_TEMPLATES)
        include_location = random.random() < 0.46
        include_time = random.random() < 0.78
    elif label == "reuniao_confirmada":
        template = random.choice(CONFIRMACAO_TEMPLATES)
        include_location = random.random() < 0.70
        include_time = random.random() < 0.80
    else:
        template = random.choice(NAO_REUNIAO_TEMPLATES)
        include_location = False
        include_time = False

    body_location = location if include_location else "remoto"
    body_time = time_expression if include_time else "quando for conveniente"

    body = template.format(
        opening=opening,
        recipient_name=recipient["name"],
        topic=topic,
        location=body_location,
        time=body_time,
    )

    if random.random() < 0.18 and label != "nao_reuniao":
        body += "\n\nSe preferir, posso adaptar ao horario do departamento."
    if random.random() < 0.12:
        body += "\n\nObrigado pela atencao."

    body = f"{body}\n\n{closing},\n{sender['name']}{random.choice(NOISE)}"

    return {
        "id": email_id,
        "subject": format_subject(label, topic),
        "body": body.strip(),
        "label": label,
        "sender": sender["name"],
        "sender_email": sender["email"],
        "sender_role": sender["role"],
        "recipient": recipient["name"],
        "recipient_email": recipient["email"],
        "recipient_role": recipient["role"],
        "sent_datetime": random_sent_datetime(),
        "topic": topic,
        "location": location if include_location and label != "nao_reuniao" else None,
        "time_expression": time_expression if include_time and label != "nao_reuniao" else None,
        "participants": [sender["name"], recipient["name"]] if label != "nao_reuniao" else [],
    }


def generate_dataset() -> List[Dict[str, object]]:
    emails: List[Dict[str, object]] = []
    email_id = 1
    for label, count in LABEL_COUNTS.items():
        for _ in range(count):
            emails.append(build_email(label, email_id))
            email_id += 1
    random.shuffle(emails)
    return emails


def label_distribution(emails: List[Dict[str, object]]) -> Dict[str, int]:
    counts = {label: 0 for label in LABEL_COUNTS}
    for email in emails:
        counts[str(email["label"])] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate realistic school/professional PT-PT emails."
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    emails = generate_dataset()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(emails, file, ensure_ascii=False, indent=2)

    print(f"{len(emails)} emails gerados com sucesso.")
    print(f"Distribuicao: {label_distribution(emails)}")
    print(f"Guardado em: {output_path}")


if __name__ == "__main__":
    main()
