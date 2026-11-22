"""Understanding the user's question.

Two parts:
1. Entity extraction (rules + fuzzy matching): countries, years, "top N",
   biggest/smallest.
2. Intent classification (ML): TF-IDF features + logistic regression,
   trained on data/intents.csv.

Before classification, country names and years are replaced with
placeholder tokens (<country>, <year>). That way the model learns the
*shape* of a question ("how much does <country> import from <country>")
instead of memorising which countries appeared in the training examples.
"""

from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import FeatureUnion, Pipeline

ROOT = Path(__file__).resolve().parents[1]
INTENTS_PATH = ROOT / "data" / "intents.csv"
MODEL_PATH = ROOT / "models" / "intent_model.joblib"
METRICS_PATH = ROOT / "models" / "intent_metrics.json"

# below this probability we ask the user to rephrase instead of guessing
MIN_CONFIDENCE = 0.30

COUNTRY_ALIASES = {
    "Germany": ["germany", "german", "deutschland"],
    "China": ["china", "chinese", "prc"],
    "USA": ["usa", "u.s.a", "u.s.", "united states", "america", "american"],
    "Japan": ["japan", "japanese"],
    "Netherlands": ["netherlands", "the netherlands", "dutch", "holland"],
    "Rep. of Korea": ["rep. of korea", "republic of korea", "south korea", "korea", "korean"],
}

BOTTOM_WORDS = ("smallest", "lowest", "least", "bottom", "minimum", "weakest")


@dataclass
class Entities:
    countries: list[str] = field(default_factory=list)  # in order of appearance
    years: list[int] = field(default_factory=list)
    top_n: int | None = None
    ascending: bool = False

    def to_dict(self) -> dict:
        return {
            "countries": self.countries,
            "years": self.years,
            "top_n": self.top_n,
            "ascending": self.ascending,
        }


def _alias_patterns():
    pats = []
    for country, aliases in COUNTRY_ALIASES.items():
        for a in sorted(aliases, key=len, reverse=True):
            pats.append((re.compile(r"(?<![a-z])" + re.escape(a) + r"(?![a-z])"), country, a))
    return pats


_PATTERNS = _alias_patterns()
_ALL_ALIASES = {a: c for c, al in COUNTRY_ALIASES.items() for a in al if " " not in a and "." not in a}
# real words that are close to a country name and must not be "corrected"
_COMMON_WORDS = {"chain", "china's", "japanese", "germane", "american", "korean", "dutch"}
_US_PATTERN = re.compile(r"\bUS\b|\bthe us\b|\bU\.S\b")


def extract_entities(text: str) -> Entities:
    low = text.lower()
    found: list[tuple[int, str]] = []
    taken: list[tuple[int, int]] = []

    for pat, country, _ in _PATTERNS:
        for m in pat.finditer(low):
            span = (m.start(), m.end())
            if any(s < span[1] and span[0] < e for s, e in taken):
                continue  # "south korea" already matched, skip "korea"
            taken.append(span)
            found.append((m.start(), country))

    # plain "us" is usually the pronoun, so only accept "US" in caps or "the us"
    for m in _US_PATTERN.finditer(text):
        if not any(s <= m.start() < e for s, e in taken):
            found.append((m.start(), "USA"))

    # simple typo tolerance: "germny", "netherland", "japn"
    for m in re.finditer(r"[a-z]{4,}", low):
        if any(s <= m.start() < e for s, e in taken) or m.group() in _COMMON_WORDS:
            continue
        close = difflib.get_close_matches(m.group(), _ALL_ALIASES.keys(), n=1, cutoff=0.82)
        if close:
            found.append((m.start(), _ALL_ALIASES[close[0]]))

    countries = []
    for _, c in sorted(found):
        if c not in countries:
            countries.append(c)

    years = sorted({int(y) for y in re.findall(r"\b(20\d{2}|19\d{2})\b", text)})

    top_n = None
    m = re.search(r"\b(?:top|bottom|first|best|biggest|largest|smallest|lowest)\s+(\d{1,2})\b", low) or re.search(
        r"\b(\d{1,2})\s+(?:biggest|largest|top|main|smallest|lowest)\b", low
    )
    if m:
        top_n = int(m.group(1))

    ascending = any(w in low for w in BOTTOM_WORDS)
    return Entities(countries=countries, years=years, top_n=top_n, ascending=ascending)


def normalize(text: str) -> str:
    """Lowercase and swap entities for placeholder tokens."""
    low = " " + text.lower() + " "
    for pat, _, _ in _PATTERNS:
        low = pat.sub(" xcountryx ", low)
    low = re.sub(r"\b(?:19|20)\d{2}\b", " xyearx ", low)
    low = re.sub(r"\b\d+\b", " xnumx ", low)
    low = re.sub(r"[^a-z' ]", " ", low)
    return re.sub(r"\s+", " ", low).strip()


def build_pipeline() -> Pipeline:
    features = FeatureUnion([
        ("words", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
        # character n-grams make it robust to small typos ("suplier", "forcast")
        ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)),
    ])
    clf = LogisticRegression(C=8.0, max_iter=3000, class_weight="balanced")
    return Pipeline([("features", features), ("clf", clf)])


def load_training_data(path: Path = INTENTS_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["text"] = df["text"].astype(str)
    return df


def evaluate(df: pd.DataFrame, folds: int = 5, seed: int = 42) -> dict:
    X = df["text"].map(normalize)
    y = df["intent"]
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    pred = cross_val_predict(build_pipeline(), X, y, cv=cv)
    report = classification_report(y, pred, output_dict=True, zero_division=0)
    return {
        "cv_folds": folds,
        "accuracy": round(report["accuracy"], 3),
        "macro_f1": round(report["macro avg"]["f1-score"], 3),
        "per_intent_f1": {k: round(v["f1-score"], 3) for k, v in report.items() if isinstance(v, dict) and k in set(y)},
        "n_examples": int(len(df)),
        "n_intents": int(y.nunique()),
    }


def train(save: bool = True, verbose: bool = False) -> Pipeline:
    df = load_training_data()
    metrics = evaluate(df)
    model = build_pipeline().fit(df["text"].map(normalize), df["intent"])

    if save:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        metrics["trained_at"] = datetime.now().isoformat(timespec="seconds")
        METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    if verbose:
        print(json.dumps(metrics, indent=2))
    return model


def load_model() -> Pipeline:
    """Load the saved model, or train a fresh one if it's missing/broken."""
    if MODEL_PATH.exists():
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            pass  # e.g. saved with a different sklearn version
    return train(save=True)


def load_metrics() -> dict:
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text())
    return {}


@dataclass
class Prediction:
    intent: str
    confidence: float
    alternatives: list[tuple[str, float]]


def predict_intent(model: Pipeline, text: str) -> Prediction:
    proba = model.predict_proba([normalize(text)])[0]
    classes = model.classes_
    order = np.argsort(proba)[::-1]
    best = order[0]
    alts = [(classes[i], round(float(proba[i]), 3)) for i in order[1:3]]
    return Prediction(intent=classes[best], confidence=float(proba[best]), alternatives=alts)
