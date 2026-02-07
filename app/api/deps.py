
from fastapi import Request, Query, HTTPException, Depends
from datetime import datetime
from app.core.database import get_db

active_sessions = {}

def check_active_session(
    request: Request,
    user_id: str = Query(...)
):
    session = active_sessions.get(user_id)
    if not session:
        raise HTTPException(status_code=403, detail="User not logged in")

    # Check session expiry
    if datetime.utcnow() > session.get("expiry", datetime.utcnow()):
        del active_sessions[user_id]
        raise HTTPException(status_code=403, detail="Session expired")

    # Check IP matches
    client_ip = request.client.host
    # NOTE: In production behind proxy, trust x-forwarded-for carefully
    
    if session.get("ip") != client_ip:
        # Strict IP check
        # raise HTTPException(status_code=403, detail="IP mismatch")
        pass # Relaxed for dev

    request.state.username = session.get("username")
    return session.get("username")
