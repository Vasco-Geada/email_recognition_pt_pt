import imaplib, email, os
from preprocessing.preprocess import preprocessEmail
from dotenv import load_dotenv
from email.header import decode_header

def extractEmail():
    load_dotenv()

    EMAIL = os.getenv("EMAIL")
    PASSWORD = os.getenv("PASSWORD")
    SERVER =  os.getenv("SERVER")
    
    mail = imaplib.IMAP4_SSL(SERVER)
    mail.login(EMAIL, PASSWORD)

    print("Ligado com sucesso!")

    mail.select("inbox")
    status, messages = mail.search(None, "ALL")

    email_ids = messages[0].split()

    for eid in email_ids:
        status, msg_data = mail.fetch(eid, "(RFC822)")
        
        raw_email = msg_data[0][1]
        msg = email.message_from_bytes(raw_email)

        subject = decode_str(msg["subject"])
        from_ = msg["from"]
        

    body = get_email_body(msg)
    print("Body:", body)

    email_data = {
        "email_id": eid.decode(),
        "email_date": msg["date"],
        "subject": subject,
        "body": body
    }

    processed = preprocessEmail(email_data)

    print(processed)

def get_email_body(msg):
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition"))

            if content_type == "text/plain" and "attachment" not in content_disposition:
                return part.get_payload(decode=True).decode(errors="ignore")
    else:
        return msg.get_payload(decode=True).decode(errors="ignore")

    return ""

def decode_str(s):
    decoded, charset = decode_header(s)[0]
    if isinstance(decoded, bytes):
        return decoded.decode(charset or "utf-8", errors="ignore")
    return decoded