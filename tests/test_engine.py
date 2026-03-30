from pramana_engine.examples import SCENARIOS
from pramana_engine.models import InferenceStatus


def test_valid_inference_is_accepted():
    engine, request = SCENARIOS["valid"]()
    result = engine.infer(request)

    assert result.status == InferenceStatus.VALID
    assert result.accepted is True


def test_unjustified_inference_when_required_pramana_missing():
    engine, request = SCENARIOS["unjustified"]()
    result = engine.infer(request)

    assert result.status == InferenceStatus.UNJUSTIFIED
    assert result.accepted is False


def test_suspended_inference_for_near_threshold_reliability():
    engine, request = SCENARIOS["suspended_threshold"]()
    result = engine.infer(request)

    assert result.status == InferenceStatus.SUSPENDED
    assert result.accepted is False


def test_invalid_inference_pattern_is_rejected():
    engine, request = SCENARIOS["invalid"]()
    result = engine.infer(request)

    assert result.status == InferenceStatus.INVALID
    assert result.accepted is False


def test_suspended_when_defeater_present():
    engine, request = SCENARIOS["suspended_defeated"]()
    result = engine.infer(request)

    assert result.status == InferenceStatus.SUSPENDED
    assert result.accepted is False


def test_trace_is_machine_readable_dict():
    engine, request = SCENARIOS["valid"]()
    result = engine.infer(request)

    assert isinstance(result.to_dict(), dict)
    assert "trace" in result.to_dict()
    assert isinstance(result.to_dict()["trace"]["steps"], list)
