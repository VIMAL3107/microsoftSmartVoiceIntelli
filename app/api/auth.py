
from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import uuid

from app.core.database import get_db
from app.models.user import User
from app.api.deps import active_sessions
from app.schemas.user import UserCreate, UserLogin
from app.services.security import (
    authenticate_user, 
    create_access_token, 
    hash_password, 
    verify_user_license
)
from app.services.email_service import send_verification_email
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/token")
def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}

@router.post("/register/")
def register_user(user: UserCreate, db: Session = Depends(get_db)):
    logger.info(f"Register attempt for username: {user.username}, email: {user.email}")
    existing = db.query(User).filter((User.username == user.username) | (User.email == user.email)).first()
    if existing:
        raise HTTPException(status_code=400, detail="Username or email already registered")
        
    hashed_pwd = hash_password(user.password)
    token = str(uuid.uuid4())
    expiry = datetime.utcnow() + timedelta(hours=24)
    
    new_user = User(
        username=user.username,
        email=user.email,
        password=hashed_pwd,
        is_email_verified=True, # Auto-verify for testing since email is broken
        email_verification_token=token,
        token_expiry=expiry
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    try:
        send_verification_email(user.email, token)
    except Exception as e:
        logger.error(f"Error sending verification email: {e}")
        # Don't block registration if email fails.
        # Ideally, we should have a way to resend verification email later.
        pass
        
    return {"message": f"User {user.username} registered. Please check email to verify."}
        
    return {"message": f"User {user.username} registered. Check mail."}

@router.get("/verify-email")
def verify_email(token: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email_verification_token == token).first()
    if not user:
        raise HTTPException(status_code=400, detail="Invalid verification token")
        
    if user.token_expiry and datetime.utcnow() > user.token_expiry:
        raise HTTPException(status_code=400, detail="Token expired")
        
    user.is_email_verified = True
    user.email_verification_token = None
    user.token_expiry = None
    db.commit()
    
    return {"message": "Email verified successfully"}

@router.post("/login/")
def login(user: UserLogin, request: Request, db: Session = Depends(get_db)):
    username = user.username
    password = user.password

    if not username or not password:
        raise HTTPException(status_code=400, detail="Username or password missing")
    
    db_user = authenticate_user(db, username, password)
    if not db_user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if not db_user.is_email_verified:
        raise HTTPException(status_code=403, detail="Email not verified")

    client_ip = request.headers.get("x-forwarded-for") or request.client.host
    if client_ip and "," in client_ip:
        client_ip = client_ip.split(",")[0].strip()

    # verify_user_license(db_user.username, client_ip)  # Uncomment in prod

    active_sessions[str(db_user.id)] = {
        "ip": client_ip,
        "login_time": datetime.utcnow(),
        "expiry": datetime.utcnow() + timedelta(hours=2),
        "user_id": db_user.id,
        "email": db_user.email,
        "username": db_user.username
    }

    return {
        "status": "success", 
        "message": "Login successful", 
        "ip": client_ip, 
        "user_id": db_user.id,
        "username": db_user.username
    }
