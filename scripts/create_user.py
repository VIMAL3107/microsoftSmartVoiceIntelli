
import sys
import os

# Add the project root to the python path so we can import app modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.core.database import SessionLocal
from app.models.user import User
from app.services.security import hash_password

def create_manual_user(username, password, email):
    db = SessionLocal()
    try:
        # Check if user exists
        existing_user = db.query(User).filter(User.username == username).first()
        if existing_user:
            print(f"User '{username}' already exists.")
            
            # Optional: Update password if needed
            # existing_user.password = hash_password(password)
            # existing_user.is_email_verified = True
            # db.commit()
            # print("Updated existing user password and verification status.")
            return

        # Create new user
        new_user = User(
            username=username,
            email=email,
            password=hash_password(password),
            is_email_verified=True,  # Auto-verify since we are creating manually
            email_verification_token=None,
            token_expiry=None
        )
        db.add(new_user)
        db.commit()
        print(f"Successfully created user '{username}' with email '{email}'.")
        
    except Exception as e:
        print(f"Error creating user: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    # You can change these values or pass them as arguments
    USERNAME = "vimal011"
    PASSWORD = "1234"
    EMAIL = "vimal011@example.com"  # Use a dummy email if needed
    
    print(f"Creating user: {USERNAME}...")
    create_manual_user(USERNAME, PASSWORD, EMAIL)
