"""
Single source of truth for all pramana definitions.
Import from here everywhere — never duplicate the list.
"""
from __future__ import annotations

import logging
from typing import Dict, Tuple

_logger = logging.getLogger("pramana.registry")

# Canonical pramana names in authority order (highest → lowest)
# All six Nyaya pramanas are represented, including Anupalabdhi (non-perception/absence).
ALL_PRAMANAS: Tuple[str, ...] = (
    "perception",
    "inference",
    "testimony",
    "comparison",
    "postulation",
    "non_perception",   # Anupalabdhi — inference from absence of perception
)

# Authority weights: derived from Nyaya Sutra ordering of epistemic priority
# perception=1.0 (direct sense contact), non_perception=0.35 (absence-mediated)
PRAMANA_AUTHORITY: Dict[str, float] = {
    "perception":    1.0,   # Pratyaksha — immediate sense-object contact
    "inference":     0.8,   # Anumana — hetu-vyapti mediated
    "testimony":     0.6,   # Shabda — credible speaker (apta-vacana)
    "comparison":    0.4,   # Upamana — analogical (similarity-based)
    "postulation":   0.2,   # Arthapatti — explanatory necessity
    "non_perception": 0.35, # Anupalabdhi — absence-cognition (between comparison and postulation)
}

# Sanskrit display names
PRAMANA_DISPLAY: Dict[str, str] = {
    "perception":    "Pratyaksha",
    "inference":     "Anumana",
    "testimony":     "Shabda",
    "comparison":    "Upamana",
    "postulation":   "Arthapatti",
    "non_perception": "Anupalabdhi",
}

# All aliases that map to a canonical pramana name
PRAMANA_ALIASES: Dict[str, Tuple[str, ...]] = {
    "perception":  ("pratyaksha", "pratyaksa", "perception", "sense-object", "sense object", "direct"),
    "inference":   ("anumana", "inference", "inferential", "hetu", "vyapti"),
    "testimony":   ("shabda", "sabda", "testimony", "verbal", "scriptural"),
    "comparison":  ("upamana", "comparison", "analogy"),
    "postulation": ("arthapatti", "postulation", "presumption"),
    "non_perception": (
        "anupalabdhi", "non-perception", "non_perception", "absence",
        "non-cognition", "non cognition", "abhava", "absence-cognition",
    ),
}


def normalize_pramana(label: str) -> str:
    """Map any pramana label (including Sanskrit) to its canonical name."""
    text = (label or "").strip().lower()
    for canonical, aliases in PRAMANA_ALIASES.items():
        if text == canonical or text in aliases:
            return canonical
    _logger.warning(
        "Unknown pramana label %r — defaulting to 'testimony'. "
        "Add an entry to PRAMANA_ALIASES to silence this.",
        label,
    )
    return "testimony"


def authority_weight(pramana: str) -> float:
    """Return the epistemic authority weight [0.0, 1.0] for a pramana."""
    return PRAMANA_AUTHORITY.get(normalize_pramana(pramana), 0.6)
