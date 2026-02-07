
import smtplib
import os
import logging
import time
import socket
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from app.core.config import EMAIL_FROM, EMAIL_PASSWORD, SMTP_SERVER, SMTP_PORT, FRONTEND_URL

logger = logging.getLogger(__name__)

import socket

def send_verification_email(to_email: str, token: str):
    verify_link = f"{FRONTEND_URL}/verify-email?token={token}"
    subject = "Verify your email"
    body = f"Click this link to verify your email: {verify_link}"
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = EMAIL_FROM
    msg["To"] = to_email

    for port in [SMTP_PORT, 465]:
        try:
            logger.info(f"Connecting to SMTP server: {SMTP_SERVER}:{port}")
            context = smtplib.ssl.create_default_context()
            
            if port == 465:
                server = smtplib.SMTP_SSL(SMTP_SERVER, port, context=context)
            else:
                server = smtplib.SMTP(SMTP_SERVER, port, timeout=15)
                # Try explicit connect to catch network errors early
                try:
                    server.connect(SMTP_SERVER, port)
                except (socket.error, OSError) as e:
                    logger.warning(f"Connection to port {port} failed ({e}). Trying next port...")
                    server.close()
                    continue
                
                server.ehlo()
                server.starttls(context=context)
                server.ehlo()
            
            # If we reached here, connection is good. Try login.
            server.login(EMAIL_FROM, EMAIL_PASSWORD)
            server.sendmail(EMAIL_FROM, to_email, msg.as_string())
            server.quit()
            logger.info(f"Verification email sent to {to_email}")
            return # Success!
            
        except Exception as e:
            logger.warning(f"Failed to send via port {port}: {e}")
            if port == 465: # If last attempt failed
                 logger.error("All email attempts failed.")
                 raise e

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
    # Try ports in order: Configured Port, then 465 (fallback)
    # Use a set to avoid duplicates if SMTP_PORT is already 465
    ports_to_try = [SMTP_PORT]
    if SMTP_PORT != 465:
        ports_to_try.append(465)

    for port in ports_to_try:
        try:
            logger.info(f"Connecting to SMTP server: {SMTP_SERVER}:{port}")
            context = smtplib.ssl.create_default_context()
            
            if port == 465:
                server = smtplib.SMTP_SSL(SMTP_SERVER, port, context=context)
            else:
                server = smtplib.SMTP(SMTP_SERVER, port, timeout=30)
                server.connect(SMTP_SERVER, port)
                server.ehlo()
                server.starttls(context=context)
                server.ehlo()
                
            server.login(EMAIL_FROM, EMAIL_PASSWORD)
            server.sendmail(EMAIL_FROM, to_email, msg.as_string())
            server.quit()
            logger.info(f"Report email sent to {to_email}")
            return
            
        except Exception as e:
            logger.warning(f"Failed to send report via port {port}: {e}")
            if port == 465: # If last fallback failed
                logger.error(f"All report email attempts failed.")
                raise e
