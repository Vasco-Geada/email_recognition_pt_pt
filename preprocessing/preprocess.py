import preprocessing.cleaning as cleaning, preprocessing.metadata as mData

def preprocessEmail(email):
    subject = email.get("subject", "")
    body = email.get("body", "")

    body = cleaning.normalizeText(body)
    body = cleaning.removeEmailHistory(body)
    body = cleaning.removeSignature(body)

    sentences = mData.splitSentences(body)
    tokens = mData.tokenization(body)

    metadata = mData.extractMetadata(email, body)

    return {
        "email_id": email.get("email_id"),
        "email_date": email.get("email_date"),
        "subject": subject,
        "clean_body": body,
        "sentences": sentences,
        "tokens": tokens,
        **metadata
    }