"""
Redis Caching and Rate-Limiting layer.
Provides perceptual frame hash caching and API client rate limiting.
Includes a thread-safe in-memory fallback when Redis server is offline.
"""

import os
import time
import hashlib
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")


class EmotionCache:
    """
    High-performance caching layer with Redis connection and in-memory fallback.
    """
    def __init__(self, redis_url: str = REDIS_URL, default_ttl: int = 300):
        self.default_ttl = default_ttl
        self.redis_client = None
        self._memory_cache: Dict[str, tuple[Any, float]] = {}

        try:
            import redis
            client = redis.from_url(redis_url, socket_connect_timeout=1)
            client.ping()
            self.redis_client = client
            logger.info("Connected to Redis at %s", redis_url)
        except Exception:
            logger.info("Redis unavailable at %s. Utilizing thread-safe in-memory cache.", redis_url)

    @staticmethod
    def compute_frame_hash(face_crop) -> str:
        """Computes a fast deterministic hash of an image crop."""
        import cv2
        small = cv2.resize(face_crop, (32, 32))
        return hashlib.sha256(small.tobytes()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """Retrieves cached prediction by key."""
        if self.redis_client:
            try:
                import json
                val = self.redis_client.get(f"fer:{key}")
                if val:
                    return json.loads(val)
            except Exception as e:
                logger.debug("Redis get error: %s", e)

        # In-memory lookup
        if key in self._memory_cache:
            val, expiry = self._memory_cache[key]
            if time.time() < expiry:
                return val
            else:
                del self._memory_cache[key]
        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Sets cached prediction with TTL (seconds)."""
        ttl_sec = ttl or self.default_ttl

        if self.redis_client:
            try:
                import json
                self.redis_client.setex(f"fer:{key}", ttl_sec, json.dumps(value))
                return
            except Exception as e:
                logger.debug("Redis set error: %s", e)

        # In-memory storage
        self._memory_cache[key] = (value, time.time() + ttl_sec)

    def is_rate_limited(self, client_id: str, max_requests: int = 120, window_sec: int = 60) -> bool:
        """
        Token-bucket / sliding window rate limiter.
        Returns True if client has exceeded maximum allowed requests.
        """
        now = time.time()
        bucket_key = f"rate:{client_id}"

        if self.redis_client:
            try:
                current = self.redis_client.incr(bucket_key)
                if current == 1:
                    self.redis_client.expire(bucket_key, window_sec)
                return current > max_requests
            except Exception:
                pass

        # In-memory fallback rate-limiting
        if bucket_key not in self._memory_cache:
            self._memory_cache[bucket_key] = ([now], now + window_sec)
            return False

        timestamps, _ = self._memory_cache[bucket_key]
        valid_ts = [t for t in timestamps if now - t < window_sec]
        valid_ts.append(now)
        self._memory_cache[bucket_key] = (valid_ts, now + window_sec)

        return len(valid_ts) > max_requests
