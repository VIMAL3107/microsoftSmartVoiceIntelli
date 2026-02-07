
import os
import sys
import logging

# Add the project root to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services.email_service import send_report_email_with_attachment
from app.core.config import EMAIL_FROM

# Configure logging to see the output
logging.basicConfig(level=logging.INFO)

def test_email_sending():
    print("--- Testing Email Sending with Retry Logic ---")
    
    # Create a dummy attachment file
    dummy_file = "test_attachment.txt"
    with open(dummy_file, "w") as f:
        f.write("This is a test attachment content.")
    
    try:
        # Send to the sender's own email for testing
        to_email = EMAIL_FROM
        subject = "Test Email with Retry Logic"
        body = "This is a test email to verify the retry mechanism. connection check."
        
        print(f"Attempting to send email to {to_email}...")
        send_report_email_with_attachment(to_email, subject, body, dummy_file)
        print("\n[SUCCESS] Email function executed successfully.")
        
    except Exception as e:
        print(f"\n[FAILURE] Email sending failed: {e}")
        
    finally:
        # Clean up
        if os.path.exists(dummy_file):
            os.remove(dummy_file)
            print("Cleaned up test file.")

if __name__ == "__main__":
    test_email_sending()
