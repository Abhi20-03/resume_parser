from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional


DEFAULT_HF_NER_MODEL = "dslim/bert-base-NER"


class HFNERExtractor:
    """Optional Hugging Face NER enrichment for names and organizations."""

    def __init__(self) -> None:
        self._pipeline = None
        self._model_name: Optional[str] = None

    def _is_enabled(self) -> bool:
        return os.getenv("HF_NER_ENABLED", "1").strip().lower() in {"1", "true", "yes"}

    def _get_model_name(self) -> str:
        if self._model_name:
            return self._model_name
        self._model_name = os.getenv("HF_NER_MODEL", "").strip() or DEFAULT_HF_NER_MODEL
        return self._model_name

    def _load_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        if not self._is_enabled():
            return None
        try:
            from transformers import pipeline
        except Exception:
            return None

        token = (
            os.getenv("HF_TOKEN", "").strip()
            or os.getenv("HUGGINGFACE_TOKEN", "").strip()
            or os.getenv("HUGGINGFACEHUB_API_TOKEN", "").strip()
        )

        kwargs: Dict[str, Any] = {
            "task": "token-classification",
            "model": self._get_model_name(),
            "aggregation_strategy": "simple",
        }
        if token:
            kwargs["token"] = token

        try:
            self._pipeline = pipeline(**kwargs)
        except Exception:
            self._pipeline = None
        return self._pipeline

    def _clean_entity(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text).strip()
        return text.replace(" .", ".")

    def extract(self, text: str) -> Dict[str, Any]:
        ner = self._load_pipeline()
        if ner is None or not text.strip():
            return {"name": None, "organizations": []}

        preview = text[:4000]
        try:
            entities = ner(preview)
        except Exception:
            return {"name": None, "organizations": []}

        names: List[str] = []
        organizations: List[str] = []

        for entity in entities:
            label = str(entity.get("entity_group", "")).upper()
            word = self._clean_entity(str(entity.get("word", "")))
            if len(word) < 2:
                continue
            if label == "PER":
                if word not in names:
                    names.append(word)
            elif label == "ORG":
                if word not in organizations:
                    organizations.append(word)

        preferred_name = names[0] if names else None
        return {"name": preferred_name, "organizations": organizations[:10]}

    @property
    def model_name(self) -> Optional[str]:
        if not self._is_enabled():
            return None
        return self._get_model_name()
