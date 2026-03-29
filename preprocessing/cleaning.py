import re

def normalizeText(text):
    text = text.strip()

    # normalize spaces
    text = re.sub(r"\s+", " ", text)

    text = re.sub(r"[^\w\s.,!?@:/\-à-úÀ-Ú]", "", text)

    return text

def removeEmailHistory(text):
    patterns = [
        r"On .* wrote:",
        r"Subject: .*",
        r"From: .*",
        r"To: .*",
        r"De: .*",
        r"Para: .*",
        r"Assunto: .*",
        r"-----Original Message-----",
        r"-------- Mensagem original --------",
        r">.*"
    ]

    for p in patterns:
        text = re.split(p, text)[0]

    return text.strip()

def removeSignature(text):
    patterns = [
        r"Cumprimentos.*",
        r"Melhores cumprimentos.*",
        r"Obrigado.*",
        r"Com os melhores cumprimentos.*"
    ]

    for p in patterns:
        text = re.split(p, text)[0]

    return text.strip()