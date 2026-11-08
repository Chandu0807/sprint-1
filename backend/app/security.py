"""security.py - password hashing, login tokens (JWT) and role checks."""
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import func, select

from . import config, models

bearer = HTTPBearer(auto_error=False)

# demo accounts created on first start. CHANGE or remove these before any real use.
DEFAULT_USERS = [("officer", "officer123", "loan_officer"),
                 ("verifier", "verifier123", "field_verifier"),
                 ("supervisor", "supervisor123", "supervisor")]


def hash_password(p: str) -> str:
    return bcrypt.hashpw(p.encode(), bcrypt.gensalt()).decode()


def verify_password(p: str, hashed: str) -> bool:
    return bcrypt.checkpw(p.encode(), hashed.encode())


def create_token(username: str, role: str) -> str:
    exp = datetime.now(timezone.utc) + timedelta(hours=config.TOKEN_HOURS)
    return jwt.encode({"sub": username, "role": role, "exp": exp}, config.SECRET_KEY, algorithm="HS256")


def get_current_user(creds: HTTPAuthorizationCredentials = Depends(bearer)):
    if creds is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        data = jwt.decode(creds.credentials, config.SECRET_KEY, algorithms=["HS256"])
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return {"username": data["sub"], "role": data["role"]}


def require_roles(*roles):
    def checker(user=Depends(get_current_user)):
        if user["role"] not in roles:
            raise HTTPException(status_code=403, detail="Your role is not allowed to do this")
        return user
    return checker


def seed_users(db):
    if db.scalar(select(func.count()).select_from(models.User)) == 0:
        for username, password, role in DEFAULT_USERS:
            db.add(models.User(username=username, password_hash=hash_password(password), role=role))
        db.commit()
        print("Created demo users: officer / verifier / supervisor (change these passwords!)")
