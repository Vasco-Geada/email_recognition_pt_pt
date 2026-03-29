import spacy, re

nlp = spacy.load("pt_core_news_sm")

dias = {
    "2ª": "segunda",
    "3ª": "terca",
    "4ª": "quarta",
    "5ª": "quinta",
    "6ª": "sexta"
}

def isReply(email, body):
    if "Re:" in email.get("subject", ""):
        return True

    if ">" in body:
        return True

    return False

def threadLevel(body):
    return body.count(">")

def splitSentences(text):
    doc = nlp(text)
    return [sent.text for sent in doc.sents]

def tokenization(text):
    doc = nlp(text)
    return [token.text for token in doc]

def normalizeHours(text):
    text = re.sub(r"(\d{1,2})h", r"\1:00", text)
    return text

def extractMetadata(email, body):
    return {
        "is_reply": isReply(email, body),
        "thread_level": threadLevel(body),
    }
    