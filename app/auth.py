from fastapi import Header, HTTPException
import httpx
from typing import Optional
from app.config import settings

# Optional local JWT verification
try:
    from jose import jwt
    from jose.exceptions import JWTError  # noqa: F401
except Exception:
    jwt = None

_jwks_cache: Optional[dict] = None


async def _load_jwks():
    global _jwks_cache
    if not settings.JWT_JWKS_URL:
        return None
    if _jwks_cache:
        return _jwks_cache
    async with httpx.AsyncClient(timeout=5) as client:
        r = await client.get(settings.JWT_JWKS_URL)
        r.raise_for_status()
        _jwks_cache = r.json()
        return _jwks_cache


def extract_user_id(token: str):
    if not jwt:
        return settings.AUTH_DEFAULT_USER_ID
    # вытаскиваем payload без проверки подписи
    claims = jwt.get_unverified_claims(token)
    return claims.get('user_id', claims.get('uid', claims.get('id', settings.AUTH_DEFAULT_USER_ID)))


async def _verify_local(token: str) -> Optional[str]:
    if not jwt or not settings.JWT_JWKS_URL:
        return None
    await _load_jwks()  # загружаем, но в этом демо не используем конкретный ключ
    try:
        claims = jwt.get_unverified_claims(token)
        if settings.JWT_ISSUER and claims.get("iss") != settings.JWT_ISSUER:
            return None
        aud = claims.get("aud")
        if settings.JWT_AUDIENCE and (
                (isinstance(aud, str) and aud != settings.JWT_AUDIENCE)
                or (isinstance(aud, list) and settings.JWT_AUDIENCE not in aud)
        ):
            return None
        return claims.get("sub") or claims.get("user_id")
    except Exception:
        return None


async def verify_bearer_token(authorization: str | None = Header(default=None)) -> int:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = authorization.split(" ", 1)[1]

    if settings.AUTH_STUB_ENABLED:
        return settings.AUTH_STUB_USER_ID

    # 1) Trust canonical Django verify endpoint
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.post(settings.AUTH_VERIFY_URL, json={"token": token})
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=401, detail=f"Token verification failed: {exc}") from exc

    if resp.status_code != 200:
        raise HTTPException(status_code=401, detail="Token verification failed")

    return extract_user_id(token)
