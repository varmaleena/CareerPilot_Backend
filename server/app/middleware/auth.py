from fastapi import Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
from app.config import settings

security = HTTPBearer()


async def verify_token(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> dict:
    """Verify JWT token from Supabase."""
    token = credentials.credentials
    
    try:
        payload = jwt.decode(
            token,
            settings.SUPABASE_JWT_SECRET,
            algorithms=[settings.JWT_ALGORITHM],
            audience="authenticated",
        )
        return payload
    except JWTError as e:
        raise HTTPException(
            status_code=401,
            detail=f"Invalid authentication token: {e}",
        )


async def get_current_user(payload: dict = Depends(verify_token)) -> dict:
    """Extract user info from verified token."""
    user_id = payload.get("sub")
    email = payload.get("email")
    
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token payload")
    
    return {
        "id": user_id,
        "email": email,
        "role": payload.get("role", "user"),
    }
