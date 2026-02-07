
import smtplib
import os
import logging
import time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from app.core.config import EMAIL_FROM, EMAIL_PASSWORD, SMTP_SERVER, SMTP_PORT, FRONTEND_URL

logger = logging.getLogger(__name__)

def send_verification_email(to_email: str, token: str):
    verify_link = f"{FRONTEND_URL}/verify-email?token={token}"
    subject = "Verify your email"
    body = f"Click this link to verify your email: {verify_link}"
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = EMAIL_FROM
    msg["To"] = to_email

    try:
        logger.info(f"Connecting to SMTP server: {SMTP_SERVER}:{SMTP_PORT}")
        if SMTP_PORT == 465:
            server = smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT)
        else:
            server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
            server.starttls()
            
        server.login(EMAIL_FROM, EMAIL_PASSWORD)
        server.sendmail(EMAIL_FROM, to_email, msg.as_string())
        server.quit()
        logger.info(f"Verification email sent to {to_email}")
    except Exception as e:
        logger.error(f"Failed to send verification email via {SMTP_SERVER}:{SMTP_PORT}: {e}")
        raise

def send_report_email_with_attachment(to_email: str, subject: str, body: str, file_path: str):
    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["From"] = EMAIL_FROM
    msg["To"] = to_email
    
    # Attach body
    msg.attach(MIMEText(body, "plain"))
    
    # Attach file
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            part = MIMEApplication(f.read(), Name=os.path.basename(file_path))
        part['Content-Disposition'] = f'attachment; filename="{os.path.basename(file_path)}"'
        msg.attach(part)
    else:
        logger.warning(f"Attachment not found: {file_path}")

    max_retries = 3
    for attempt in range(max_retries):
        try:
            logger.info(f"Connecting to SMTP server: {SMTP_SERVER}:{SMTP_PORT} (Attempt {attempt + 1}/{max_retries})")
            if SMTP_PORT == 465:
                server = smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT)
            else:
                server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=30)
                server.starttls()
                
            server.login(EMAIL_FROM, EMAIL_PASSWORD)
            server.sendmail(EMAIL_FROM, to_email, msg.as_string())
            server.quit()
            logger.info(f"Report email sent to {to_email}")
            return
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
            else:
                logger.error(f"Failed to send report email via {SMTP_SERVER}:{SMTP_PORT} after {max_retries} attempts: {e}")
                raise
