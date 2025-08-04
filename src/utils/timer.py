"""
Timer utilities for performance measurement.
"""

import time
import functools
from typing import Optional, Callable, Any
from contextlib import contextmanager

from .logger import get_logger

logger = get_logger(__name__)


class Timer:
    """Simple timer for measuring execution time."""
    
    def __init__(self):
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
    
    def start(self) -> 'Timer':
        """Start the timer."""
        self.start_time = time.time()
        self.end_time = None
        return self
    
    def stop(self) -> float:
        """
        Stop the timer and return elapsed time.
        
        Returns:
            Elapsed time in seconds
        """
        if self.start_time is None:
            raise ValueError("Timer not started")
        
        self.end_time = time.time()
        return self.elapsed()
    
    def elapsed(self) -> float:
        """
        Get elapsed time in seconds.
        
        Returns:
            Elapsed time in seconds
        """
        if self.start_time is None:
            return 0.0
        
        end = self.end_time if self.end_time is not None else time.time()
        return end - self.start_time
    
    def elapsed_ms(self) -> float:
        """
        Get elapsed time in milliseconds.
        
        Returns:
            Elapsed time in milliseconds
        """
        return self.elapsed() * 1000
    
    def reset(self) -> 'Timer':
        """Reset the timer."""
        self.start_time = None
        self.end_time = None
        return self
    
    def __enter__(self) -> 'Timer':
        """Context manager entry."""
        return self.start()
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()


class PerformanceTimer:
    """Advanced timer with statistics tracking."""
    
    def __init__(self, name: str = "operation"):
        self.name = name
        self.measurements = []
        self.current_start = None
    
    def start(self):
        """Start timing."""
        self.current_start = time.time()
    
    def stop(self) -> float:
        """
        Stop timing and record measurement.
        
        Returns:
            Elapsed time in seconds
        """
        if self.current_start is None:
            raise ValueError("Timer not started")
        
        elapsed = time.time() - self.current_start
        self.measurements.append(elapsed)
        self.current_start = None
        return elapsed
    
    @contextmanager
    def measure(self):
        """Context manager for measuring execution time."""
        self.start()
        try:
            yield self
        finally:
            self.stop()
    
    def get_stats(self) -> dict:
        """
        Get performance statistics.
        
        Returns:
            Dictionary with timing statistics
        """
        if not self.measurements:
            return {
                "name": self.name,
                "count": 0,
                "total": 0.0,
                "average": 0.0,
                "min": 0.0,
                "max": 0.0
            }
        
        total = sum(self.measurements)
        count = len(self.measurements)
        
        return {
            "name": self.name,
            "count": count,
            "total": total,
            "average": total / count,
            "min": min(self.measurements),
            "max": max(self.measurements),
            "total_ms": total * 1000,
            "average_ms": (total / count) * 1000,
            "min_ms": min(self.measurements) * 1000,
            "max_ms": max(self.measurements) * 1000
        }
    
    def reset(self):
        """Reset all measurements."""
        self.measurements.clear()
        self.current_start = None
    
    def log_stats(self):
        """Log performance statistics."""
        stats = self.get_stats()
        if stats["count"] > 0:
            logger.info(
                f"Performance stats for {stats['name']}: "
                f"count={stats['count']}, "
                f"avg={stats['average_ms']:.2f}ms, "
                f"min={stats['min_ms']:.2f}ms, "
                f"max={stats['max_ms']:.2f}ms"
            )


def timeit(func: Optional[Callable] = None, *, log_result: bool = True, name: Optional[str] = None):
    """
    Decorator to time function execution.
    
    Args:
        func: Function to decorate (when used without parameters)
        log_result: Whether to log the timing result
        name: Custom name for the timer (defaults to function name)
        
    Returns:
        Decorated function or decorator
    """
    def decorator(f: Callable) -> Callable:
        timer_name = name or f.__name__
        
        @functools.wraps(f)
        def wrapper(*args, **kwargs) -> Any:
            timer = Timer()
            timer.start()
            
            try:
                result = f(*args, **kwargs)
                elapsed_ms = timer.elapsed_ms()
                
                if log_result:
                    logger.info(f"Function '{timer_name}' executed in {elapsed_ms:.2f}ms")
                
                return result
            except Exception as e:
                elapsed_ms = timer.elapsed_ms()
                logger.error(f"Function '{timer_name}' failed after {elapsed_ms:.2f}ms: {e}")
                raise
        
        return wrapper
    
    if func is None:
        # Called with parameters: @timeit(log_result=False)
        return decorator
    else:
        # Called without parameters: @timeit
        return decorator(func)


def async_timeit(func: Optional[Callable] = None, *, log_result: bool = True, name: Optional[str] = None):
    """
    Decorator to time async function execution.
    
    Args:
        func: Async function to decorate (when used without parameters)
        log_result: Whether to log the timing result
        name: Custom name for the timer (defaults to function name)
        
    Returns:
        Decorated async function or decorator
    """
    def decorator(f: Callable) -> Callable:
        timer_name = name or f.__name__
        
        @functools.wraps(f)
        async def wrapper(*args, **kwargs) -> Any:
            timer = Timer()
            timer.start()
            
            try:
                result = await f(*args, **kwargs)
                elapsed_ms = timer.elapsed_ms()
                
                if log_result:
                    logger.info(f"Async function '{timer_name}' executed in {elapsed_ms:.2f}ms")
                
                return result
            except Exception as e:
                elapsed_ms = timer.elapsed_ms()
                logger.error(f"Async function '{timer_name}' failed after {elapsed_ms:.2f}ms: {e}")
                raise
        
        return wrapper
    
    if func is None:
        # Called with parameters: @async_timeit(log_result=False)
        return decorator
    else:
        # Called without parameters: @async_timeit
        return decorator(func)


class Benchmark:
    """Benchmark utility for comparing performance."""
    
    def __init__(self):
        self.timers = {}
    
    def get_timer(self, name: str) -> PerformanceTimer:
        """
        Get or create a timer by name.
        
        Args:
            name: Timer name
            
        Returns:
            PerformanceTimer instance
        """
        if name not in self.timers:
            self.timers[name] = PerformanceTimer(name)
        return self.timers[name]
    
    def measure(self, name: str):
        """
        Context manager for measuring named operations.
        
        Args:
            name: Operation name
            
        Returns:
            Context manager
        """
        return self.get_timer(name).measure()
    
    def get_all_stats(self) -> dict:
        """
        Get statistics for all timers.
        
        Returns:
            Dictionary with all timer statistics
        """
        return {name: timer.get_stats() for name, timer in self.timers.items()}
    
    def log_all_stats(self):
        """Log statistics for all timers."""
        for timer in self.timers.values():
            timer.log_stats()
    
    def reset_all(self):
        """Reset all timers."""
        for timer in self.timers.values():
            timer.reset()
    
    def compare(self, name1: str, name2: str) -> dict:
        """
        Compare performance between two operations.
        
        Args:
            name1: First operation name
            name2: Second operation name
            
        Returns:
            Comparison statistics
        """
        if name1 not in self.timers or name2 not in self.timers:
            raise ValueError("Both operations must have measurements")
        
        stats1 = self.timers[name1].get_stats()
        stats2 = self.timers[name2].get_stats()
        
        if stats1["count"] == 0 or stats2["count"] == 0:
            raise ValueError("Both operations must have measurements")
        
        ratio = stats1["average"] / stats2["average"]
        difference_ms = (stats1["average"] - stats2["average"]) * 1000
        
        return {
            "operation1": stats1,
            "operation2": stats2,
            "ratio": ratio,
            "difference_ms": difference_ms,
            "faster": name1 if stats1["average"] < stats2["average"] else name2
        }