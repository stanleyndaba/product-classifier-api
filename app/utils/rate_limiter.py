"""
Rate limiting utilities for request throttling.
"""

import time
from collections import defaultdict
from typing import Dict, Optional

from ..core.config import get_settings


class RateLimiter:
    """
    Simple in-memory rate limiter.
    
    For production use, consider Redis-based rate limiting.
    """
    
    def __init__(self):
        """Initialize rate limiter."""
        self.settings = get_settings()
        self.requests: Dict[str, list] = defaultdict(list)
    
    async def is_allowed(self, client_id: str) -> bool:
        """
        Check if request is allowed for client.
        
        Args:
            client_id: Client identifier
            
        Returns:
            bool: True if request is allowed
        """
        if not self.settings.rate_limit_enabled:
            return True
        
        current_time = time.time()
        window_start = current_time - self.settings.rate_limit_window
        
        # Clean old requests
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id] 
            if req_time > window_start
        ]
        
        # Check if under limit
        if len(self.requests[client_id]) < self.settings.rate_limit_requests:
            self.requests[client_id].append(current_time)
            return True
        
        return False
    
    async def get_limit_info(self, client_id: str) -> Dict[str, any]:
        """
        Get rate limit information for client.
        
        Args:
            client_id: Client identifier
            
        Returns:
            Dict[str, any]: Rate limit information
        """
        current_time = time.time()
        window_start = current_time - self.settings.rate_limit_window
        
        # Clean old requests
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id] 
            if req_time > window_start
        ]
        
        current_requests = len(self.requests[client_id])
        
        # Calculate retry after
        retry_after = 0
        if current_requests >= self.settings.rate_limit_requests:
            oldest_request = min(self.requests[client_id])
            retry_after = oldest_request + self.settings.rate_limit_window - current_time
        
        return {
            "limit": self.settings.rate_limit_requests,
            "window": self.settings.rate_limit_window,
            "current": current_requests,
            "remaining": max(0, self.settings.rate_limit_requests - current_requests),
            "retry_after": max(0, retry_after)
        }