"""Hugging Face Inference API extractor for large hosted models."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

from app.services.groq_extractor import _extract_json_object, _normalize_keys, _truncate_text

HF_MODEL_ENV = "HF_MODEL"
HF_TOKEN_ENV = "HF_TOKEN"
HF_API_BASE_ENV = "HF_API_BASE"
#DEFAULT_HF_MODEL = ""
DEFAULT_HF_API_BASE = "https://api-inference.huggingface.co/models"


class HuggingFaceResumeExtractor:
    """
    Resume extraction via Hugging Face hosted inference.
    Requires HF_TOKEN. Uses large-model default unless HF_MODEL is set.
    """

    def __init__(self) -> None:
        self._model_name: Optional[str] = None
        self._token: str = os.getenv(HF_TOKEN_ENV, "").strip()
        self._last_generation_text: Optional[str] = None
        self._api_base: str = os.getenv(HF_API_BASE_ENV, "").strip() or DEFAULT_HF_API_BASE

    def _get_model(self) -> str:
        if self._model_name:
            return self._model_name
        self._model_name = os.getenv(HF_MODEL_ENV, "").strip() or DEFAULT_HF_MODEL
        return self._model_name

    def _empty_result(self) -> Dict[str, Any]:
        return {
            "contact": {"name": None, "email": None, "phone": None},
            "education": [],
            "work_experience": [],
            "skills": [],
            "additional_sections": [],
        }

    def _build_prompt(self, resume_text: str, pre_extracted: Optional[Dict[str, Any]] = None) -> str:
        t = _truncate_text(resume_text, 7000)
        hints = json.dumps(pre_extracted or {}, ensure_ascii=True)
        return (
            "You are a resume parser. Improve and complete resume JSON using extracted hints + raw resume text.\n\n"
            "Schema:\n"
            '{"contact":{"name":"","email":"","phone":""},'
            '"education":[{"institution":"","degree":"","graduation_year":""}],'
            '"work_experience":[{"company":"","position":"","description":"","duration":""}],'
            '"skills":["skill1","skill2"],'
            '"additional_sections":[{"title":"","items":[]}]}\n\n'
            "Rules:\n"
            "- Use pre_extracted_hints as a strong prior.\n"
            "- Fill missing fields from resume_text.\n"
            "- Preserve correct hint values unless resume_text clearly contradicts them.\n"
            "- Use null for missing values. graduation_year must be 4-digit year.\n"
            "- Output valid JSON only. No markdown, no extra text.\n\n"
            f"pre_extracted_hints:\n{hints}\n\n"
            f"resume_text:\n{t}"
        )

    def _call_hf(self, prompt: str) -> Optional[str]:
        if not self._token:
            return None
        model = self._get_model()
        url = f"{self._api_base}/{model}"
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 1024,
                "temperature": 0.0,
                "return_full_text": False,
            },
        }
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), method="POST")
        req.add_header("Authorization", f"Bearer {self._token}")
        req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                raw = resp.read().decode("utf-8", errors="ignore")
                data = json.loads(raw)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError):
            return None

        # Common formats: [{"generated_text":"..."}] or {"generated_text":"..."}
        if isinstance(data, list) and data:
            first = data[0]
            if isinstance(first, dict):
                return str(first.get("generated_text", "")).strip()
        if isinstance(data, dict):
            if "generated_text" in data:
                return str(data.get("generated_text", "")).strip()
            # Some models return error payloads
            if "error" in data:
                return None
        return None

    def extract(self, resume_text: str, pre_extracted: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self._last_generation_text = None
        prompt = self._build_prompt(resume_text, pre_extracted=pre_extracted)
        content = self._call_hf(prompt)
        self._last_generation_text = content or ""
        if not content:
            return self._empty_result()

        parsed = _extract_json_object(content)
        if not parsed:
            return self._empty_result()

        normalized = _normalize_keys(parsed)
        result = self._empty_result()
        for k in ["contact", "education", "work_experience", "skills", "additional_sections"]:
            v = normalized.get(k)
            if v is None:
                continue
            if k == "contact" and isinstance(v, dict):
                result["contact"].update((x, y) for x, y in v.items() if y)
            elif isinstance(v, list) and v:
                result[k] = v
        return result

    @property
    def model_name(self) -> Optional[str]:
        return self._get_model()

    @property
    def last_generation_text(self) -> Optional[str]:
        return self._last_generation_text

