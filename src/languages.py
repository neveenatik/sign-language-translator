"""
Sign-language registry.

Central definition of the sign languages the app can recognize. Each entry
pairs a label set (the classes the model predicts) with the model file to load
and display metadata (name, text direction). Adding a new language is a matter
of registering another `SignLanguage` here plus dropping in a trained model.

The pipeline is otherwise language-agnostic: MediaPipe produces a 63-dim hand
landmark vector and the classifier maps it to one of these labels.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Editing controls shared by every language.
CONTROL_LABELS: List[str] = ["space", "delete"]

# American Sign Language manual alphabet (A-Z) + editing controls.
ASL_LETTERS: List[str] = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z",
]
ASL_LABELS: List[str] = ASL_LETTERS + CONTROL_LABELS

# Arabic Sign Language manual alphabet (28 letters) + editing controls.
# NOTE: several ArSL letters are two-handed and/or dynamic (motion-based); this
# static single-hand alphabet covers the letters the current pipeline can model.
ARSL_LETTERS: List[str] = [
    "ا", "ب", "ت", "ث", "ج", "ح", "خ", "د", "ذ", "ر",
    "ز", "س", "ش", "ص", "ض", "ط", "ظ", "ع", "غ", "ف",
    "ق", "ك", "ل", "م", "ن", "ه", "و", "ي",
]
ARSL_LABELS: List[str] = ARSL_LETTERS + CONTROL_LABELS

# Dataset folder-name aliases -> canonical label. Lets image datasets that name
# folders with transliterations (or dataset-specific tokens) map onto our
# labels. Keys are matched case-insensitively. A value of None means "skip".
ASL_FOLDER_ALIASES: Dict[str, Optional[str]] = {
    "del": "delete",
    "space": "space",
    "nothing": None,
}

# Best-effort romanization aliases for the ArSL alphabet. Folders may instead be
# named directly with the Arabic letter, which always resolves.
ARSL_FOLDER_ALIASES: Dict[str, Optional[str]] = {
    "aleff": "ا", "alef": "ا", "alif": "ا",
    "bb": "ب", "baa": "ب", "beh": "ب",
    "ta": "ت", "taa": "ت", "teh": "ت",
    "thaa": "ث", "theh": "ث",
    "jeem": "ج", "jim": "ج", "geem": "ج",
    "hah": "ح", "haa": "ح",
    "khaa": "خ", "kha": "خ",
    "dal": "د", "daal": "د",
    "thal": "ذ", "dhal": "ذ", "zal": "ذ",
    "ra": "ر", "raa": "ر", "reh": "ر",
    "zay": "ز", "zayn": "ز", "zai": "ز",
    "seen": "س", "sin": "س",
    "sheen": "ش", "shin": "ش",
    "saad": "ص", "sad": "ص",
    "dhad": "ض", "daad": "ض", "dad": "ض",
    "tah": "ط", "emphatic_ta": "ط",
    "dha": "ظ", "zah": "ظ",
    "ain": "ع", "ayn": "ع",
    "ghain": "غ", "ghayn": "غ",
    "fa": "ف", "faa": "ف", "feh": "ف",
    "qaaf": "ق", "qaf": "ق", "gaaf": "ق",
    "kaaf": "ك", "kaf": "ك", "keh": "ك",
    "laam": "ل", "lam": "ل",
    "meem": "م", "mim": "م",
    "nun": "ن", "noon": "ن",
    "heh": "ه",
    "waw": "و", "wow": "و",
    "ya": "ي", "yaa": "ي", "yeh": "ي",
    "space": "space",
    "del": "delete",
    "nothing": None,
}


@dataclass(frozen=True)
class SignLanguage:
    """Metadata + label set for one recognizable sign language."""

    code: str            # short identifier, e.g. "asl"
    name: str            # human-readable name for the UI
    labels: List[str]    # ordered classes the classifier outputs
    model_path: str      # path to the trained .h5 model
    rtl: bool = False    # right-to-left text direction (Arabic, etc.)
    folder_aliases: Dict[str, Optional[str]] = field(default_factory=dict)


# Registered sign languages, keyed by `code`.
LANGUAGES: Dict[str, "SignLanguage"] = {
    "asl": SignLanguage(
        code="asl",
        name="American Sign Language (ASL)",
        labels=ASL_LABELS,
        model_path="models/asl_model.h5",
        rtl=False,
        folder_aliases=ASL_FOLDER_ALIASES,
    ),
    "arsl": SignLanguage(
        code="arsl",
        name="Arabic Sign Language (ArSL)",
        labels=ARSL_LABELS,
        model_path="models/arsl_model.h5",
        rtl=True,
        folder_aliases=ARSL_FOLDER_ALIASES,
    ),
}

DEFAULT_LANGUAGE = "asl"


def get_language(code: str) -> "SignLanguage":
    """Return the registered language for `code` (falls back to the default)."""
    return LANGUAGES.get(code, LANGUAGES[DEFAULT_LANGUAGE])


def available_languages() -> List["SignLanguage"]:
    """Return all registered languages in registration order."""
    return list(LANGUAGES.values())


def resolve_folder_label(language: "SignLanguage", folder_name: str) -> Optional[str]:
    """
    Map a dataset folder name to a canonical label for `language`, or None to skip.

    Resolution order:
      1. Exact label match (e.g. the Arabic letter itself, or "A").
      2. Case-insensitive label match (handles "a" -> "A").
      3. Registered folder alias (transliteration / dataset token).
    """
    name = folder_name.strip()
    labels_by_lower = {lbl.lower(): lbl for lbl in language.labels}

    if name in language.labels:
        return name
    if name.lower() in labels_by_lower:
        return labels_by_lower[name.lower()]
    if name.lower() in language.folder_aliases:
        return language.folder_aliases[name.lower()]
    return None

