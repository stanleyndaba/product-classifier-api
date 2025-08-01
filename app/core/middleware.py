"""
Middleware components for request processing, security, and monitoring.
"""

import time
import uuid
from typing import Callable

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.gzip import GZipMiddleware

from .config import Settings
from .logging import RequestLogger
from .exceptions import RateLimitException
from ..utils.rate_limiter import RateLimiter


class RequestContextMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add request context and timing information.
    """
    
    def __init__(self, app, logger: RequestLogger = None):
        super().__init__(app)
        self.logger = logger or RequestLogger()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request with context and timing.
        
        Args:
            request: Incoming request
            call_next: Next middleware or endpoint
            
        Returns:
            Response: HTTP response
        """
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Record start time
        start_time = time.time()
        request.state.start_time = start_time
        
        # Get client information
        client_ip = self.get_client_ip(request)
        user_agent = request.headers.get("user-agent", "")
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Log successful request
            self.logger.log_request(
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration=duration,
                user_agent=user_agent,
                ip_address=client_ip,
                request_id=request_id
            )
            
            # Add response headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time"] = f"{duration:.3f}s"
            
            return response
            
        except Exception as exc:
            # Calculate duration for error case
            duration = time.time() - start_time
            
            # Log error
            self.logger.log_error(
                method=request.method,
                path=request.url.path,
                error=exc,
                request_id=request_id,
                duration=duration,
                ip_address=client_ip
            )
            
            raise exc
    
    def get_client_ip(self, request: Request) -> str:
        """
        Extract client IP address from request headers.
        
        Args:
            request: Incoming request
            
        Returns:
            str: Client IP address
        """
        # Check for forwarded headers (common in load balancers)
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        # Fall back to direct client IP
        return request.client.host if request.client else "unknown"


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware for rate limiting requests.
    """
    
    def __init__(self, app, rate_limiter: RateLimiter, enabled: bool = True):
        super().__init__(app)
        self.rate_limiter = rate_limiter
        self.enabled = enabled
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Apply rate limiting to requests.
        
        Args:
            request: Incoming request
            call_next: Next middleware or endpoint
            
        Returns:
            Response: HTTP response
            
        Raises:
            RateLimitException: When rate limit is exceeded
        """
        if not self.enabled:
            return await call_next(request)
        
        # Get client identifier
        client_id = self.get_client_identifier(request)
        
        # Check rate limit
        allowed = await self.rate_limiter.is_allowed(client_id)
        
        if not allowed:
            # Get rate limit info
            limit_info = await self.rate_limiter.get_limit_info(client_id)
            
            raise RateLimitException(
                message="Rate limit exceeded",
                details={
                    "limit": limit_info.get("limit"),
                    "window": limit_info.get("window"),
                    "retry_after": limit_info.get("retry_after")
                }
            )
        
        return await call_next(request)
    
    def get_client_identifier(self, request: Request) -> str:
        """
        Get client identifier for rate limiting.
        
        Args:
            request: Incoming request
            
        Returns:
            str: Client identifier
        """
        # Use IP address as identifier (could be enhanced with API keys)
        client_ip = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
        if not client_ip:
            client_ip = request.headers.get("x-real-ip", "")
        if not client_ip:
            client_ip = request.client.host if request.client else "unknown"
        
        return client_ip


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add security headers to responses.
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Add security headers to response.
        
        Args:
            request: Incoming request
            call_next: Next middleware or endpoint
            
        Returns:
            Response: HTTP response with security headers
        """
        response = await call_next(request)
        
        # Add security headers
        security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Content-Security-Policy": "default-src 'self'"
        }
        
        for header, value in security_headers.items():
            response.headers[header] = value
        
        return response


class MetricsMiddleware(BaseHTTPMiddleware):
    """
    Middleware to collect application metrics.
    """
    
    def __init__(self, app):
        super().__init__(app)
        self.metrics = {
            "requests_total": 0,
            "requests_by_status": {},
            "requests_by_path": {},
            "response_times": [],
            "active_requests": 0
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Collect metrics for request.
        
        Args:
            request: Incoming request
            call_next: Next middleware or endpoint
            
        Returns:
            Response: HTTP response
        """
        start_time = time.time()
        
        # Increment active requests
        self.metrics["active_requests"] += 1
        
        try:
            response = await call_next(request)
            
            # Calculate response time
            response_time = time.time() - start_time
            
            # Update metrics
            self.update_metrics(request, response, response_time)
            
            return response
            
        except Exception as exc:
            # Update error metrics
            response_time = time.time() - start_time
            self.update_error_metrics(request, exc, response_time)
            raise exc
        
        finally:
            # Decrement active requests
            self.metrics["active_requests"] -= 1
    
    def update_metrics(self, request: Request, response: Response, response_time: float) -> None:
        """
        Update success metrics.
        
        Args:
            request: HTTP request
            response: HTTP response
            response_time: Response time in seconds
        """
        # Total requests
        self.metrics["requests_total"] += 1
        
        # Requests by status code
        status_code = response.status_code
        self.metrics["requests_by_status"][status_code] = (
            self.metrics["requests_by_status"].get(status_code, 0) + 1
        )
        
        # Requests by path
        path = request.url.path
        self.metrics["requests_by_path"][path] = (
            self.metrics["requests_by_path"].get(path, 0) + 1
        )
        
        # Response times (keep last 1000)
        self.metrics["response_times"].append(response_time)
        if len(self.metrics["response_times"]) > 1000:
            self.metrics["response_times"] = self.metrics["response_times"][-1000:]
    
    def update_error_metrics(self, request: Request, exc: Exception, response_time: float) -> None:
        """
        Update error metrics.
        
        Args:
            request: HTTP request
            exc: Exception that occurred
            response_time: Response time in seconds
        """
        # Count as 500 error
        self.metrics["requests_by_status"][500] = (
            self.metrics["requests_by_status"].get(500, 0) + 1
        )
        
        # Add to response times
        self.metrics["response_times"].append(response_time)
        if len(self.metrics["response_times"]) > 1000:
            self.metrics["response_times"] = self.metrics["response_times"][-1000:]
    
    def get_metrics(self) -> dict:
        """
        Get current metrics.
        
        Returns:
            dict: Current application metrics
        """
        response_times = self.metrics["response_times"]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        return {
            "requests_total": self.metrics["requests_total"],
            "requests_by_status": self.metrics["requests_by_status"],
            "requests_by_path": self.metrics["requests_by_path"],
            "active_requests": self.metrics["active_requests"],
            "average_response_time": round(avg_response_time, 3),
            "response_time_percentiles": self.calculate_percentiles(response_times)
        }
    
    def calculate_percentiles(self, response_times: list) -> dict:
        """
        Calculate response time percentiles.
        
        Args:
            response_times: List of response times
            
        Returns:
            dict: Response time percentiles
        """
        if not response_times:
            return {}
        
        sorted_times = sorted(response_times)
        length = len(sorted_times)
        
        percentiles = {}
        for p in [50, 90, 95, 99]:
            index = int((p / 100) * length) - 1
            if index < 0:
                index = 0
            percentiles[f"p{p}"] = round(sorted_times[index], 3)
        
        return percentiles


def setup_middleware(app: FastAPI, settings: Settings) -> None:
    """
    Setup middleware for the FastAPI application.
    
    Args:
        app: FastAPI application instance
        settings: Application settings
    """
    # Initialize components
    request_logger = RequestLogger()
    rate_limiter = RateLimiter()
    
    # Add middleware in reverse order (they are processed in LIFO order)
    
    # Metrics middleware (outermost)
    metrics_middleware = MetricsMiddleware(app)
    app.add_middleware(BaseHTTPMiddleware, dispatch=metrics_middleware.dispatch)
    app.state.metrics_middleware = metrics_middleware
    
    # Security headers middleware
    app.add_middleware(SecurityHeadersMiddleware)
    
    # Rate limiting middleware
    app.add_middleware(
        BaseHTTPMiddleware,
        dispatch=RateLimitMiddleware(
            app, 
            rate_limiter, 
            enabled=settings.rate_limit_enabled
        ).dispatch
    )
    
    # Request context middleware
    app.add_middleware(
        BaseHTTPMiddleware,
        dispatch=RequestContextMiddleware(app, request_logger).dispatch
    )
    
    # GZIP compression middleware
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=settings.cors_methods,
        allow_headers=settings.cors_headers,
    )
    
    # Trusted host middleware (if hosts are specified)
    if settings.allowed_hosts and settings.allowed_hosts != ["*"]:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=settings.allowed_hosts
        )