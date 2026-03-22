"""Ollama-based resume extraction using local Llama models (e.g. llama3.2)."""

from __future__ import annotations

import json
import os
import re
import urllib.request
import urllib.error
from typing import Any, Dict, Optional

from app.services.groq_extractor import _extract_json_object, _normalize_keys, _truncate_text

DEFAULT_OLLAMA_MODEL = "llava:latest"
OLLAMA_BASE_URL = "http://localhost:11434"


def _ollama_chat(model: str, prompt: str, base_url: str = OLLAMA_BASE_URL, max_tokens: int = 1024) -> Optional[str]:
    """Call Ollama chat API and return the model's response text."""
    url = f"{base_url.rstrip('/')}/api/chat"
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"num_predict": max_tokens, "temperature": 0},
    }).encode("utf-8")

    req = urllib.request.Request(url, data=payload, method="POST")
    req.add_header("Content-Type", "application/json")

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode())
            return (data.get("message") or {}).get("content", "").strip()
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, TimeoutError):
        return None


class OllamaResumeExtractor:
    """
    Resume extraction via local Ollama (e.g. llama3.2). No API key needed.
    Requires Ollama running with the model pulled (e.g. ollama run llama3.2).
    """

    def __init__(self) -> None:
        self._model_name: Optional[str] = None
        self._base_url: str = os.getenv("OLLAMA_BASE_URL", "").strip() or OLLAMA_BASE_URL
        self._last_generation_text: Optional[str] = None

    def _get_model(self) -> str:
        if self._model_name:
            return self._model_name
        self._model_name = os.getenv("OLLAMA_MODEL", "").strip() or DEFAULT_OLLAMA_MODEL
        return self._model_name

    def _empty_result(self) -> Dict[str, Any]:
        return {
            "contact": {"name": None, "email": None, "phone": None},
            "education": [],
            "work_experience": [],
            "skills": [],
            "additional_sections": [],
        }

    def _build_full_prompt(self, resume_text: str, pre_extracted: Optional[Dict[str, Any]] = None) -> str:
        t = _truncate_text(resume_text, 6000)
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
            "- Use the pre_extracted_hints as a strong prior.\n"
            "- Fill missing fields from resume_text.\n"
            "- Preserve correct values from hints unless resume_text clearly contradicts them.\n"
            "- Use null for missing values. graduation_year as 4-digit year.\n"
            "- Output valid JSON only. No markdown, no extra text.\n\n"
            f"pre_extracted_hints:\n{hints}\n\n"
            f"resume_text:\n{t}"
        )

    def extract(self, resume_text: str, pre_extracted: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self._last_generation_text = None
        model = self._get_model()
        prompt = self._build_full_prompt(resume_text, pre_extracted=pre_extracted)

        content = _ollama_chat(model, prompt, self._base_url, max_tokens=1024)
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
