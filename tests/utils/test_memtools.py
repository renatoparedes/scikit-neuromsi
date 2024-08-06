from skneuromsi.utils import memtools

# =============================================================================
# TESTS MEM USAGE
# =============================================================================


def test_memory_usage():
    """Test the `MemoryUsage` class."""
    memory_usage = memtools.memory_usage(1)
    assert memory_usage.size == 32
    assert memory_usage.hsize == "32 Bytes"
    assert repr(memory_usage) == "<memusage '32 Bytes'>"
