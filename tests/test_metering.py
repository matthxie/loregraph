"""Cost accounting the app consumes as a real spend cap (kg/metering.py)."""
import pytest


def test_openai_cache_write_tokens_are_carved_out_and_priced():
    """Cache WRITES are a third rate and live inside prompt_tokens, like cached reads.

    Missing them under-bills every chained call. The app consumes this meter's dollars as a
    real daily spend cap, so this is the direction that actually costs money.
    """
    import types
    from kg.metering import UsageMeter, price

    usage = types.SimpleNamespace(
        prompt_tokens=5990,
        completion_tokens=4,
        prompt_tokens_details=types.SimpleNamespace(cached_tokens=4774, cache_write_tokens=829),
    )
    msg = types.SimpleNamespace(usage=usage, choices=[types.SimpleNamespace(finish_reason="stop")])
    meter = UsageMeter()
    rec = meter.record("extract", "gpt-5.6-terra", msg)

    # 5990 total = 4774 cached + 829 written + 387 plain — three disjoint slices.
    assert rec.cache_read == 4774
    assert rec.cache_write == 829
    assert rec.input_tokens == 387
    expected = price("gpt-5.6-terra", 387, 4, cache_read=4774, cache_write=829)
    assert rec.usd == pytest.approx(expected)
    # And it must exceed the old under-billing that ignored the write premium.
    assert rec.usd > price("gpt-5.6-terra", 1216, 4, cache_read=4774, cache_write=0)
