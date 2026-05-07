import random
import json
from datetime import datetime, timedelta

# 📌 dados base
names = ["Ana", "João", "Maria", "Pedro", "Sofia", "Ricardo"]
locations = ["sala A", "sala B", "Teams", "Zoom", "escritório"]
topics = [
    "projeto da tese",
    "relatório",
    "planeamento",
    "cliente XPTO",
    "revisão do código"
]

temporal_expressions = [
    "amanhã",
    "sexta",
    "segunda",
    "para a semana",
    "amanhã à tarde",
    "às 15h",
    "às 10h",
    "depois de almoço, perto das 14:30h?"
]

weekDay_expressions = [
    "segunda",
    "sexta",
    "terça",
    "quarta",
    "quinta",
    "próxima segunda",
    "próxima terça",
    "próxima quarta",
    "próxima quinta",
    "próxima sexta"
]

time_expressions = [
    "às 12h",
    "às 14h",
    "às 9:30h?",
    "às 16h?",
]

# 📌 templates

agendamento_templates = [
    "Olá {name}, podemos reunir {time} para falar sobre {topic}?",
    "Boas {name}, vamos marcar uma reunião {time}?",
    "Consegues reunir {time} no {location}?",
    "Podemos agendar uma call {time} sobre {topic}?",
    "Pretendo marcar uma entrevista para falarmos um pouco sobre os teus conhecimentos na área web e outros assuntos importantes. Podemos marcar para esta semana?",
    "Pode ser {time}?",
    "Pode então ficar marcado para quarta-feira. Pode ser {hour}?"
]

cancelamento_templates = [
    "Olá {name}, a reunião de {time} foi cancelada.",
    "Temos de cancelar a reunião {time}.",
    "A reunião marcada para {time} fica sem efeito.",
    "Desculpa, mas vou ter de desmarcar a reunião {time}.",
    "Peço imensa desculpa mas acha que podemos remarcar a entrevista para {weekday} à mesma hora?"
]

confirmacao_templates = [
    "Perfeito, fica então {time}.",
    "Confirmado para {time} no {location}.",
    "Ok, combinamos {time}.",
    "Fechado, reunião {time}.",
    "Sim sim claro. Sem problema.",
    "Fica então para {weekday} às {hour}."
    "Por mim está ótimo, pode ser ficar entao para {weekday} às {hour}."
]

def generate_email(template_list, label):
    name = random.choice(names)
    time = random.choice(temporal_expressions)
    hour = random.choice(time_expressions)
    weekday = random.choice(weekDay_expressions)
    location = random.choice(locations)
    topic = random.choice(topics)

    text = random.choice(template_list).format(
        name=name,
        time=time,
        hour=hour,
        weekday=weekday,
        location=location,
        topic=topic
    )

    return {
        "subject": "Reunião",
        "body": text,
        "label": label
    }

# 📌 gerar dataset
dataset = []

for _ in range(80):
    dataset.append(generate_email(agendamento_templates, "agendamento_reuniao"))

for _ in range(60):
    dataset.append(generate_email(cancelamento_templates, "cancelamento_reuniao"))

for _ in range(60):
    dataset.append(generate_email(confirmacao_templates, "reuniao_confirmada"))

# baralhar
random.shuffle(dataset)

# guardar
with open("dataset/temp_emails.json", "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)

print("Dataset gerado com sucesso!")