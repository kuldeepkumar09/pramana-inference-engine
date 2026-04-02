import logging

from pramana_engine.pramana_registry import normalize_pramana, authority_weight


def test_normalize_pramana_known_labels():
    assert normalize_pramana("pratyaksha") == "perception"
    assert normalize_pramana("Anumana") == "inference"
    assert normalize_pramana("shabda") == "testimony"
    assert normalize_pramana("upamana") == "comparison"
    assert normalize_pramana("arthapatti") == "postulation"


def test_normalize_pramana_unknown_logs_warning(caplog):
    """Unknown pramana label must fall back to 'testimony' and emit a warning."""
    with caplog.at_level(logging.WARNING, logger="pramana.registry"):
        result = normalize_pramana("pratyksah")  # intentional typo
    assert result == "testimony"
    assert "Unknown pramana label" in caplog.text
    assert "pratyksah" in caplog.text


def test_normalize_pramana_empty_string_logs_warning(caplog):
    """Empty string is also unknown and should log + fall back."""
    with caplog.at_level(logging.WARNING, logger="pramana.registry"):
        result = normalize_pramana("")
    assert result == "testimony"


def test_authority_weight_unknown_defaults_to_testimony_weight():
    """authority_weight for an unknown label falls through to testimony weight (0.6)."""
    assert authority_weight("completely_unknown") == 0.6
