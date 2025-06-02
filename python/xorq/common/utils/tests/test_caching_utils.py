from pathlib import Path

from xorq.common.utils.caching_utils import user_cache_dir


def test_default_caching_dir():
    actual_dir = user_cache_dir()
    assert actual_dir is not None
    assert isinstance(actual_dir, (str, Path))
