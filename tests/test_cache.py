"""
Tests for EmotionCache module.
"""

import pytest
import numpy as np
from cache.redis_client import EmotionCache


@pytest.fixture
def cache():
    return EmotionCache()


def test_cache_hashing_and_retrieval(cache):
    """Verifies that frame hash caching and retrieval works."""
    dummy_crop = np.zeros((64, 64, 3), dtype=np.uint8)
    h = cache.compute_frame_hash(dummy_crop)

    test_payload = {"dominant_emotion": "surprise", "confidence": 0.88}
    cache.set(h, test_payload, ttl=60)

    cached_val = cache.get(h)
    assert cached_val is not None
    assert cached_val["dominant_emotion"] == "surprise"


def test_rate_limiter_allows_normal_traffic(cache):
    """Normal traffic within threshold should not be rate limited."""
    client_id = "test_user_normal"
    for _ in range(5):
        assert cache.is_rate_limited(client_id, max_requests=10, window_sec=60) is False
