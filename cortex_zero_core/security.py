"""
HALT-NN Security Module

Provides:
- Rate limiting per IP
- Input validation and sanitization
- Request size limits
- Security headers
- API key authentication (optional)
- Content Security Policy
"""

import re
import time
import hashlib
import secrets
from typing import Optional, Dict, Set
from functools import wraps
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# RATE LIMITING
# =============================================================================

class RateLimiter:
    """
    Simple in-memory rate limiter using sliding window.
    
    Tracks requests per IP and blocks excessive traffic.
    """
    
    def __init__(self, requests_per_minute: int = 60, requests_per_hour: int = 500):
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self._minute_windows: Dict[str, list] = {}
        self._hour_windows: Dict[str, list] = {}
        self._blocked_ips: Dict[str, float] = {}  # IP -> unblock time
        self._block_duration = 300  # 5 minutes
    
    def is_allowed(self, ip: str) -> tuple[bool, str]:
        """Check if request from IP is allowed."""
        now = time.time()
        
        # Check if IP is blocked
        if ip in self._blocked_ips:
            if now < self._blocked_ips[ip]:
                remaining = int(self._blocked_ips[ip] - now)
                return False, f"Blocked for {remaining}s due to rate limit violation"
            else:
                del self._blocked_ips[ip]
        
        # Clean old entries
        minute_ago = now - 60
        hour_ago = now - 3600
        
        # Minute window
        if ip not in self._minute_windows:
            self._minute_windows[ip] = []
        self._minute_windows[ip] = [t for t in self._minute_windows[ip] if t > minute_ago]
        
        # Hour window
        if ip not in self._hour_windows:
            self._hour_windows[ip] = []
        self._hour_windows[ip] = [t for t in self._hour_windows[ip] if t > hour_ago]
        
        # Check limits
        if len(self._minute_windows[ip]) >= self.requests_per_minute:
            self._blocked_ips[ip] = now + self._block_duration
            logger.warning(f"Rate limit exceeded (minute): {ip}")
            return False, "Rate limit exceeded (60 requests/minute)"
        
        if len(self._hour_windows[ip]) >= self.requests_per_hour:
            self._blocked_ips[ip] = now + self._block_duration
            logger.warning(f"Rate limit exceeded (hour): {ip}")
            return False, "Rate limit exceeded (500 requests/hour)"
        
        # Record request
        self._minute_windows[ip].append(now)
        self._hour_windows[ip].append(now)
        
        return True, "OK"
    
    def get_stats(self) -> Dict:
        """Get rate limiter statistics."""
        return {
            "tracked_ips": len(self._minute_windows),
            "blocked_ips": len(self._blocked_ips),
            "requests_per_minute_limit": self.requests_per_minute,
            "requests_per_hour_limit": self.requests_per_hour
        }


# Global rate limiter instance
rate_limiter = RateLimiter()


# =============================================================================
# INPUT VALIDATION & SANITIZATION
# =============================================================================

class InputValidator:
    """Validate and sanitize user inputs."""
    
    # Maximum lengths
    MAX_QUERY_LENGTH = 5000
    MAX_EVIDENCE_LENGTH = 50000
    MAX_SOURCE_ID_LENGTH = 500
    
    # Dangerous patterns
    DANGEROUS_PATTERNS = [
        r'<script[^>]*>.*?</script>',  # XSS
        r'javascript:',
        r'on\w+\s*=',  # Event handlers
        r'data:text/html',
        r'vbscript:',
        r'\{\{.*?\}\}',  # Template injection
        r'\$\{.*?\}',    # Template literals
        r'<!--.*?-->',   # HTML comments
    ]
    
    # SQL injection patterns
    SQL_PATTERNS = [
        r";\s*(DROP|DELETE|UPDATE|INSERT|ALTER|CREATE|TRUNCATE)",
        r"'\s*(OR|AND)\s*'?\d*'?\s*=\s*'?\d*",
        r"UNION\s+SELECT",
        r"--\s*$",
    ]
    
    @classmethod
    def sanitize_text(cls, text: str) -> str:
        """Remove dangerous content from text."""
        if not text:
            return ""
        
        # Remove null bytes
        text = text.replace('\x00', '')
        
        # Escape HTML entities
        text = text.replace('<', '&lt;').replace('>', '&gt;')
        
        # Remove control characters except newlines/tabs
        text = ''.join(c for c in text if c >= ' ' or c in '\n\t\r')
        
        return text.strip()
    
    @classmethod
    def validate_query(cls, query: str) -> tuple[bool, str, str]:
        """
        Validate a query string.
        
        Returns: (is_valid, error_message, sanitized_query)
        """
        if not query:
            return False, "Query cannot be empty", ""
        
        if len(query) > cls.MAX_QUERY_LENGTH:
            return False, f"Query too long (max {cls.MAX_QUERY_LENGTH} chars)", ""
        
        # Check for dangerous patterns
        query_lower = query.lower()
        for pattern in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, query_lower, re.IGNORECASE):
                logger.warning(f"Dangerous pattern detected in query: {pattern}")
                return False, "Invalid query content", ""
        
        for pattern in cls.SQL_PATTERNS:
            if re.search(pattern, query_lower, re.IGNORECASE):
                logger.warning(f"SQL injection pattern detected: {pattern}")
                return False, "Invalid query content", ""
        
        return True, "", cls.sanitize_text(query)
    
    @classmethod
    def validate_evidence(cls, content: str, source_id: str) -> tuple[bool, str]:
        """Validate evidence content and source."""
        if not content:
            return False, "Evidence content cannot be empty"
        
        if len(content) > cls.MAX_EVIDENCE_LENGTH:
            return False, f"Evidence too long (max {cls.MAX_EVIDENCE_LENGTH} chars)"
        
        if len(source_id) > cls.MAX_SOURCE_ID_LENGTH:
            return False, f"Source ID too long (max {cls.MAX_SOURCE_ID_LENGTH} chars)"
        
        # Check for script injection
        for pattern in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, content.lower(), re.IGNORECASE):
                return False, "Invalid evidence content"
        
        return True, ""


# =============================================================================
# API KEY AUTHENTICATION
# =============================================================================

class APIKeyAuth:
    """Simple API key authentication."""
    
    def __init__(self):
        self._api_keys: Dict[str, Dict] = {}
        self._enabled = False
    
    def enable(self, master_key: Optional[str] = None):
        """Enable API key authentication."""
        self._enabled = True
        if master_key:
            self._api_keys[master_key] = {
                "name": "master",
                "created": time.time(),
                "permissions": ["read", "write", "admin"]
            }
        logger.info("API key authentication enabled")
    
    def generate_key(self, name: str, permissions: list = None) -> str:
        """Generate a new API key."""
        key = f"halt_{secrets.token_urlsafe(32)}"
        self._api_keys[key] = {
            "name": name,
            "created": time.time(),
            "permissions": permissions or ["read"]
        }
        return key
    
    def validate_key(self, key: str) -> tuple[bool, Optional[Dict]]:
        """Validate an API key."""
        if not self._enabled:
            return True, None
        
        if not key:
            return False, None
        
        # Handle "Bearer" prefix
        if key.startswith("Bearer "):
            key = key[7:]
        
        if key in self._api_keys:
            return True, self._api_keys[key]
        
        return False, None
    
    def has_permission(self, key: str, permission: str) -> bool:
        """Check if key has specific permission."""
        if not self._enabled:
            return True
        
        valid, info = self.validate_key(key)
        if not valid or not info:
            return False
        
        return permission in info.get("permissions", [])
    
    @property
    def is_enabled(self) -> bool:
        return self._enabled


# Global auth instance
api_auth = APIKeyAuth()


# =============================================================================
# SECURITY HEADERS
# =============================================================================

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "1; mode=block",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
    "Content-Security-Policy": (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: https:; "
        "font-src 'self'; "
        "connect-src 'self' ws: wss:; "
        "frame-ancestors 'none'"
    ),
}


def get_security_headers() -> Dict[str, str]:
    """Get security headers for responses."""
    return SECURITY_HEADERS.copy()


# =============================================================================
# REQUEST FINGERPRINTING (Detect automated attacks)
# =============================================================================

class RequestFingerprinter:
    """Detect suspicious request patterns."""
    
    def __init__(self):
        self._user_agents: Dict[str, int] = {}
        self._suspicious_uas: Set[str] = set()
    
    SUSPICIOUS_PATTERNS = [
        "sqlmap",
        "nikto",
        "nmap",
        "masscan",
        "python-requests",  # Without version often = bot
        "curl/",
        "wget/",
        "scanner",
        "exploit",
        "attack",
    ]
    
    def check_request(self, user_agent: str, ip: str) -> bool:
        """
        Check if request looks suspicious.
        
        Returns True if suspicious.
        """
        if not user_agent:
            return True  # No UA is suspicious
        
        ua_lower = user_agent.lower()
        
        for pattern in self.SUSPICIOUS_PATTERNS:
            if pattern in ua_lower:
                logger.warning(f"Suspicious UA detected from {ip}: {user_agent[:100]}")
                return True
        
        return False


# Global fingerprinter
fingerprinter = RequestFingerprinter()


# =============================================================================
# CONVENIENCE MIDDLEWARE FUNCTIONS
# =============================================================================

def create_security_middleware(app):
    """
    Create FastAPI middleware for security.
    
    Usage:
        from security import create_security_middleware
        app = FastAPI()
        create_security_middleware(app)
    """
    from fastapi import Request, HTTPException
    from fastapi.responses import JSONResponse
    
    @app.middleware("http")
    async def security_middleware(request: Request, call_next):
        # Get client IP
        ip = request.client.host if request.client else "unknown"
        
        # Check rate limit
        allowed, message = rate_limiter.is_allowed(ip)
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={"detail": message},
                headers={"Retry-After": "300"}
            )
        
        # Check for suspicious requests
        user_agent = request.headers.get("user-agent", "")
        if fingerprinter.check_request(user_agent, ip):
            logger.warning(f"Suspicious request blocked from {ip}")
            return JSONResponse(
                status_code=403,
                content={"detail": "Request blocked"}
            )
        
        # Check API key if auth is enabled
        if api_auth.is_enabled:
            auth_header = request.headers.get("authorization", "")
            # Skip auth for OPTIONS requests and static files
            if request.method != "OPTIONS" and not request.url.path.startswith("/static"):
                valid, _ = api_auth.validate_key(auth_header)
                if not valid:
                    return JSONResponse(
                        status_code=401,
                        content={"detail": "Invalid or missing API key"},
                        headers={"WWW-Authenticate": "Bearer"}
                    )
        
        # Process request
        response = await call_next(request)
        
        # Add security headers
        for header, value in SECURITY_HEADERS.items():
            response.headers[header] = value
        
        return response
    
    return app
