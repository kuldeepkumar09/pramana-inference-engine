"""
Single source of truth for all pramana definitions.
Import from here everywhere — never duplicate the list.
"""
from __future__ import annotations

from typing import Dict, Tuple

# Canonical pramana names in authority order (highest → lowest)
ALL_PRAMANAS: Tuple[str, ...] = (
    "perception",
    "inference",
    "testimony",
    "comparison",
    "postulation",
)

# Authority weights: derived from Nyaya Sutra ordering of epistemic priority
# perception=1.0 (direct sense contact), postulation=0.2 (explanatory guess)
PRAMANA_AUTHORITY: Dict[str, float] = {
    "perception":  1.0,   # Pratyaksha — immediate sense-object contact
    "inference":   0.8,   # Anumana — hetu-vyapti mediated
    "testimony":   0.6,   # Shabda — credible speaker (apta-vacana)
    "comparison":  0.4,   # Upamana — analogical
    "postulation": 0.2,   # Arthapatti — explanatory necessity
}

# Sanskrit display names
PRAMANA_DISPLAY: Dict[str, str] = {
    "perception":  "Pratyaksha",
    "inference":   "Anumana",
    "testimony":   "Shabda",
    "comparison":  "Upamana",
    "postulation": "Arthapatti",
}

# All aliases that map to a canonical pramana name
PRAMANA_ALIASES: Dict[str, Tuple[str, ...]] = {
    "perception":  ("pratyaksha", "pratyaksa", "perception", "sense-object", "sense object", "direct"),
    "inference":   ("anumana", "inference", "inferential", "hetu", "vyapti"),
    "testimony":   ("shabda", "sabda", "testimony", "verbal", "scriptural"),
    "comparison":  ("upamana", "comparison", "analogy"),
    "postulation": ("arthapatti", "postulation", "presumption"),
}


def normalize_pramana(label: str) -> str:
    """Map any pramana label (including Sanskrit) to its canonical name."""
    text = (label or "").strip().lower()
    for canonical, aliases in PRAMANA_ALIASES.items():
        if text == canonical or text in aliases:
            return canonical
    return "testimony"  # safe fallback


def authority_weight(pramana: str) -> float:
    """Return the epistemic authority weight [0.0, 1.0] for a pramana."""
    return PRAMANA_AUTHORITY.get(normalize_pramana(pramana), 0.6)
