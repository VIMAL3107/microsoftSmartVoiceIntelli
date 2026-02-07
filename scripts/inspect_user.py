
import sys
import os

# Add the project root to the python path so we can import app modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.core.database import SessionLocal
from app.models.user import User

def check_user(username):
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.username == username).first()
        if user:
            print(f"✅ User found: {user.username} (ID: {user.id})")
            print(f"   Email: {user.email}")
            print(f"   Verified: {user.is_email_verified}")
        else:
            print(f"❌ User '{username}' NOT found in the database.")
    except Exception as e:
        print(f"Error checking user: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    USERNAME = "vimal011"
    check_user(USERNAME)
