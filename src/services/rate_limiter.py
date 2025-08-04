"""
Rate limiting middleware for the Smart Product Classifier API.
"""

import time
import asyncio
from collections import defaultdict, deque
from typing import Dict, Optional
import redis
import logging

from ..utils.logger import get_logger

logger = get_logger(__name__)


class InMemoryRateLimiter:
    """In-memory rate limiter using sliding window."""
    
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.window_size = 60  # 1 minute window
        self.requests: Dict[str, deque] = defaultdict(deque)
        self._lock = asyncio.Lock()
    
    async def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed for the given identifier."""
        async with self._lock:
            now = time.time()
            user_requests = self.requests[identifier]
            
            # Remove old requests outside the window
            while user_requests and user_requests[0] < now - self.window_size:
                user_requests.popleft()
            
            # Check if limit is exceeded
            if len(user_requests) >= self.requests_per_minute:
                return False
            
            # Add current request
            user_requests.append(now)
            return True
    
    async def get_remaining_requests(self, identifier: str) -> int:
        """Get remaining requests for the identifier."""
        async with self._lock:
            now = time.time()
            user_requests = self.requests[identifier]
            
            # Remove old requests outside the window
            while user_requests and user_requests[0] < now - self.window_size:
                user_requests.popleft()
            
            return max(0, self.requests_per_minute - len(user_requests))
    
    async def reset_limits(self, identifier: Optional[str] = None):
        """Reset limits for specific identifier or all."""
        async with self._lock:
            if identifier:
                self.requests[identifier].clear()
            else:
                self.requests.clear()


class RedisRateLimiter:
    """Redis-based rate limiter for distributed systems."""
    
    def __init__(
        self,
        requests_per_minute: int = 60,
        redis_host: str = 'localhost',
        redis_port: int = 6379,
        redis_db: int = 0,
        key_prefix: str = 'rate_limit'
    ):
        self.requests_per_minute = requests_per_minute
        self.window_size = 60  # 1 minute window
        self.key_prefix = key_prefix
        
        try:
            self.redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                decode_responses=True
            )
            # Test connection
            self.redis_client.ping()
            self.redis_available = True
            logger.info("Redis rate limiter initialized successfully")
        except Exception as e:
            logger.warning(f"Redis not available, falling back to in-memory: {e}")
            self.redis_available = False
            self.fallback = InMemoryRateLimiter(requests_per_minute)
    
    def _get_key(self, identifier: str) -> str:
        """Generate Redis key for identifier."""
        return f"{self.key_prefix}:{identifier}"
    
    async def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed for the given identifier."""
        if not self.redis_available:
            return await self.fallback.is_allowed(identifier)
        
        try:
            key = self._get_key(identifier)
            now = int(time.time())
            window_start = now - self.window_size
            
            # Use Redis pipeline for atomic operations
            pipe = self.redis_client.pipeline()
            
            # Remove old entries
            pipe.zremrangebyscore(key, 0, window_start)
            
            # Count current requests
            pipe.zcard(key)
            
            # Execute pipeline
            results = pipe.execute()
            current_count = results[1]
            
            # Check if limit exceeded
            if current_count >= self.requests_per_minute:
                return False
            
            # Add current request
            self.redis_client.zadd(key, {str(now): now})
            self.redis_client.expire(key, self.window_size)
            
            return True
            
        except Exception as e:
            logger.error(f"Redis rate limiter error: {e}")
            # Fallback to allowing request if Redis fails
            return True
    
    async def get_remaining_requests(self, identifier: str) -> int:
        """Get remaining requests for the identifier."""
        if not self.redis_available:
            return await self.fallback.get_remaining_requests(identifier)
        
        try:
            key = self._get_key(identifier)
            now = int(time.time())
            window_start = now - self.window_size
            
            # Remove old entries and count current
            pipe = self.redis_client.pipeline()
            pipe.zremrangebyscore(key, 0, window_start)
            pipe.zcard(key)
            results = pipe.execute()
            
            current_count = results[1]
            return max(0, self.requests_per_minute - current_count)
            
        except Exception as e:
            logger.error(f"Redis rate limiter error: {e}")
            return self.requests_per_minute
    
    async def reset_limits(self, identifier: Optional[str] = None):
        """Reset limits for specific identifier or all."""
        if not self.redis_available:
            return await self.fallback.reset_limits(identifier)
        
        try:
            if identifier:
                key = self._get_key(identifier)
                self.redis_client.delete(key)
            else:
                # Delete all rate limit keys
                pattern = f"{self.key_prefix}:*"
                for key in self.redis_client.scan_iter(match=pattern):
                    self.redis_client.delete(key)
                    
        except Exception as e:
            logger.error(f"Redis rate limiter reset error: {e}")


class RateLimiter:
    """Main rate limiter class that chooses between Redis and in-memory."""
    
    def __init__(
        self,
        requests_per_minute: int = 60,
        use_redis: bool = True,
        redis_config: Optional[Dict] = None
    ):
        self.requests_per_minute = requests_per_minute
        
        if use_redis and redis_config:
            try:
                self.limiter = RedisRateLimiter(
                    requests_per_minute=requests_per_minute,
                    **redis_config
                )
            except Exception as e:
                logger.warning(f"Failed to initialize Redis rate limiter: {e}")
                self.limiter = InMemoryRateLimiter(requests_per_minute)
        else:
            self.limiter = InMemoryRateLimiter(requests_per_minute)
    
    async def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed."""
        return await self.limiter.is_allowed(identifier)
    
    async def get_remaining_requests(self, identifier: str) -> int:
        """Get remaining requests."""
        return await self.limiter.get_remaining_requests(identifier)
    
    async def reset_limits(self, identifier: Optional[str] = None):
        """Reset limits."""
        return await self.limiter.reset_limits(identifier)
    
    def get_limit_info(self, identifier: str) -> Dict[str, int]:
        """Get rate limit information."""
        return {
            "requests_per_minute": self.requests_per_minute,
            "window_size_seconds": 60
        }