import sys
from pathlib import Path

from xorq.common.utils.caching_utils import user_cache_dir


def tuple_contains(a, b) -> bool:
    from difflib import SequenceMatcher

    s = SequenceMatcher(None, b, a)
    return s.find_longest_match().size == len(b)


def test_default_caching_dir():
    actual_dir = user_cache_dir()
    assert actual_dir is not None
    assert isinstance(actual_dir, (str, Path))

    expected_parts = (
        ("AppData", "Local", "xorq", "cache")
        if sys.platform == "win32"
        else (".cache", "xorq")
    )
    assert tuple_contains(Path(actual_dir).parts, expected_parts)
