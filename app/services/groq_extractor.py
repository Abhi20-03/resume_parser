"""Groq API-based resume extraction using Llama 3. No model download required."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional

DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"
GROQ_MODEL_ENV = "GROQ_MODEL"


def _truncate_text(text: str, max_chars: int = 8000) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rsplit(maxsplit=1)[0] + "..."


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Extract a JSON object or array from model output."""
    if not text:
        return None

    def _parse(s: str) -> Optional[Dict[str, Any]]:
        try:
            parsed = json.loads(s)
            if isinstance(parsed, dict):
                return parsed
            if isinstance(parsed, list) and parsed:
                first = parsed[0]
                if isinstance(first, str):
                    return {"skills": parsed}
                if isinstance(first, dict):
                    if any("institution" in str(k).lower() or "degree" in str(k).lower() for k in first):
                        return {"education": parsed}
                    if any("company" in str(k).lower() or "position" in str(k).lower() for k in first):
                        return {"work_experience": parsed}
                return {"skills": parsed}
        except Exception:
            pass
        return None

    text = text.strip()
    out = _parse(text)
    if out:
        return out

    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        out = _parse(text[start : end + 1])
        if out:
            return out

    b, e = text.find("["), text.rfind("]")
    if b != -1 and e > b:
        try:
            arr = json.loads(text[b : e + 1])
            if isinstance(arr, list) and arr:
                first = arr[0]
                if isinstance(first, str):
                    return {"skills": arr}
                if isinstance(first, dict):
                    if any("institution" in str(k).lower() or "degree" in str(k).lower() for k in first):
                        return {"education": arr}
                    if any("company" in str(k).lower() or "position" in str(k).lower() for k in first):
                        return {"work_experience": arr}
                    return {"skills": arr}
        except Exception:
            pass

    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        return _parse(m.group(0))
    return None


def _normalize_keys(obj: Any) -> Any:
    aliases = {
        "workexperience": "work_experience",
        "work_experience": "work_experience",
        "experience": "work_experience",
        "education": "education",
        "skills": "skills",
        "contact": "contact",
        "contacts": "contact",
        "additional_sections": "additional_sections",
    }
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            nk = aliases.get(k.lower().replace(" ", "_"), k)
            out[nk] = _normalize_keys(v)
        return out
    if isinstance(obj, list):
        return [_normalize_keys(x) for x in obj]
    return obj


class GroqResumeExtractor:
    """
    Resume extraction via Groq API (Llama 3). No local model download.
    Requires GROQ_API_KEY environment variable.
    """

    def __init__(self) -> None:
        self._client = None
        self._model_name: Optional[str] = None
        self._last_generation_text: Optional[str] = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        if not api_key:
            return None
        try:
            from groq import Groq
            self._client = Groq(api_key=api_key)
            self._model_name = os.getenv(GROQ_MODEL_ENV, "").strip() or DEFAULT_GROQ_MODEL
            return self._client
        except ImportError:
            return None

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
        client = self._get_client()
        if client is None:
            return self._empty_result()

        model = self._model_name or DEFAULT_GROQ_MODEL
        prompt = self._build_full_prompt(resume_text, pre_extracted=pre_extracted)

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                temperature=0,
            )
            content = (response.choices[0].message.content or "").strip()
            self._last_generation_text = content

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
        except Exception:
            return self._empty_result()

    @property
    def model_name(self) -> Optional[str]:
        if self._model_name:
            return self._model_name
        self._get_client()
        return self._model_name

    @property
    def last_generation_text(self) -> Optional[str]:
        return self._last_generation_text
