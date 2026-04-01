"""JWTAuthMiddleware — API key / Bearer token validation."""

from __future__ import annotations

import os

from fastapi import HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

_bearer = HTTPBearer(auto_error=False)
_API_KEY = os.getenv("QUANTEDGE_API_KEY", "dev-insecure-key")


class JWTAuthMiddleware:
    """FastAPI dependency — validates Bearer token or API key.

    Usage in router:
        Depends(JWTAuthMiddleware())
    """

    def __call__(
        self,
        credentials: HTTPAuthorizationCredentials | None = Security(_bearer),
    ) -> str:
        if credentials is None or credentials.credentials != _API_KEY:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing API key",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return credentials.credentials
