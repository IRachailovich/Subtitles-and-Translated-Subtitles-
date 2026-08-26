import base64
import hashlib
import json
import urllib.error
import urllib.request
from dataclasses import dataclass

import jwt
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from fastapi import Header, HTTPException, Request, status
from jwt import PyJWKClient

from .db import get_or_create_user


@dataclass(frozen=True)
class Identity:
    subject: str
    email: str | None = None


class TokenVerifier:
    def __init__(self, settings):
        self.settings = settings
        self.jwks = PyJWKClient(settings.auth_jwks_url) if settings.auth_jwks_url else None

    def verify(self, authorization, dev_user=None):
        if self.settings.dev_auth_enabled and dev_user:
            return Identity(subject=f"dev:{dev_user}", email=dev_user if "@" in dev_user else None)
        if not authorization or not authorization.lower().startswith("bearer "):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
        token = authorization.split(None, 1)[1]
        try:
            algorithm = jwt.get_unverified_header(token).get("alg")
            if algorithm == "HS256":
                return self._verify_symmetric_supabase_token(token)
            if algorithm not in {"RS256", "ES256"}:
                raise ValueError("Unsupported JWT signing algorithm")
            signing_key = self.jwks.get_signing_key_from_jwt(token).key
            claims = jwt.decode(
                token,
                signing_key,
                algorithms=[algorithm],
                audience=self.settings.auth_audience,
                issuer=self.settings.auth_issuer,
            )
        except Exception as exc:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid access token") from exc
        subject = claims.get("sub")
        if not subject:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Access token has no subject")
        return Identity(subject=subject, email=claims.get("email"))

    def _verify_symmetric_supabase_token(self, token):
        if not self.settings.auth_public_url or not self.settings.auth_public_key:
            raise ValueError("Symmetric token verification is not configured")
        request = urllib.request.Request(
            f"{self.settings.auth_public_url.rstrip('/')}/auth/v1/user",
            headers={
                "Authorization": f"Bearer {token}",
                "apikey": self.settings.auth_public_key,
            },
        )
        with urllib.request.urlopen(request, timeout=15) as response:
            user = json.loads(response.read().decode("utf-8"))
        subject = user.get("id")
        if not subject:
            raise ValueError("Auth server did not return a user")
        return Identity(subject=subject, email=user.get("email"))


class CredentialCipher:
    def __init__(self, encoded_key, allow_development_key=False):
        if encoded_key:
            key = base64.urlsafe_b64decode(encoded_key.encode("ascii"))
        elif allow_development_key:
            key = hashlib.sha256(b"subgen-local-development-only").digest()
        else:
            raise RuntimeError("Credential encryption key is not configured")
        if len(key) != 32:
            raise RuntimeError("Credential encryption key must be 32 bytes")
        self.aes = AESGCM(key)

    @staticmethod
    def _aad(user_id, provider, profile):
        return f"subgen:v1:{user_id}:{provider}:{profile}".encode("utf-8")

    def encrypt(self, user_id, provider, profile, secret):
        import os

        nonce = os.urandom(12)
        encrypted = self.aes.encrypt(nonce, secret.encode("utf-8"), self._aad(user_id, provider, profile))
        return base64.urlsafe_b64encode(nonce + encrypted).decode("ascii")

    def decrypt(self, user_id, provider, profile, payload):
        raw = base64.urlsafe_b64decode(payload.encode("ascii"))
        plaintext = self.aes.decrypt(raw[:12], raw[12:], self._aad(user_id, provider, profile))
        return plaintext.decode("utf-8")


def install_auth_dependencies(app, session_factory, verifier):
    def current_user(
        request: Request,
        authorization: str | None = Header(default=None),
        x_subgen_dev_user: str | None = Header(default=None),
    ):
        identity = verifier.verify(authorization, x_subgen_dev_user)
        with session_factory.begin() as session:
            user = get_or_create_user(session, identity.subject, identity.email)
            return {"id": user.id, "subject": user.auth_subject, "email": user.email}

    app.state.current_user_dependency = current_user
    return current_user
