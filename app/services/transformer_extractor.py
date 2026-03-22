from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional

from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer


MODEL_ENV_KEY = "RESUME_PARSER_MODEL"
DEFAULT_MODEL = "meta-llama/Llama-2-7b-chat-hf"  # Llama 2; requires HuggingFace license


class TransformerResumeExtractor:
    """
    Transformer-based extraction into JSON.

    Implementation strategy:
    - Convert the resume text into an instruction prompt.
    - Ask the model to output JSON only.
    - Parse the first JSON object found in the model output.
    """

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self._model_name: Optional[str] = None
        self._is_causal = False
        self._disabled = os.getenv("DISABLE_TRANSFORMER", "").strip().lower() in {"1", "true", "yes"}
        self._load_failed = False
        self._last_generation_text: Optional[str] = None

    def _load_if_needed(self) -> None:
        if self._disabled:
            return
        if self._load_failed:
            return
        if self._model is not None and self._tokenizer is not None:
            return

        model_name = os.getenv(MODEL_ENV_KEY, "").strip() or DEFAULT_MODEL
        self._model_name = model_name

        try:
            config = AutoConfig.from_pretrained(model_name)
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            model_type = getattr(config, "model_type", "").lower()

            if model_type in ("llama", "llama2", "mistral", "qwen", "phi", "gemma"):
                self._model = AutoModelForCausalLM.from_pretrained(model_name)
                self._is_causal = True
            else:
                self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                self._is_causal = False
        except Exception:
            self._load_failed = True
            self._model = None
            self._tokenizer = None

    def _wrap_chat_prompt(self, instruction: str) -> str:
        """Wrap prompt in Llama 2 chat format for causal/chat models."""
        if not self._is_causal:
            return instruction
        return (
            f"<s>[INST] <<SYS>>\n"
            "You are a resume parser. Extract the requested information and output ONLY valid JSON, no other text.\n"
            "<</SYS>>\n\n"
            f"{instruction} [/INST]"
        )

    def _truncate_for_context(self, text: str, max_chars: Optional[int] = None, from_end: bool = False) -> str:
        """Keep resume text within model context. from_end=True uses last part (skills often at end)."""
        if max_chars is None:
            max_chars = 3500 if self._is_causal else 1800
        if len(text) <= max_chars:
            return text
        if from_end:
            cut = text[-(max_chars - 3) :]
            idx = cut.find(" ")
            return ("..." + cut[idx + 1 :]) if idx != -1 else ("..." + cut)
        return text[: max_chars - 3].rsplit(maxsplit=1)[0] + "..."

    def _build_prompt(self, resume_text: str) -> str:
        """Compact prompt: resume first so model sees content before truncation."""
        t = self._truncate_for_context(resume_text)
        return (
            f"Resume:\n{t}\n\n"
            "Extract to JSON only. Schema: {\"contact\":{\"name\",\"email\",\"phone\"},"
            "\"education\":[{\"institution\",\"degree\",\"graduation_year\"}],"
            "\"work_experience\":[{\"company\",\"position\",\"description\",\"duration\"}],"
            "\"skills\":[\"skill\"],\"additional_sections\":[{\"title\",\"items\"}]}. "
            "Use null for missing. graduation_year as 4-digit year. Output valid JSON only."
        )

    def _build_section_prompt(self, section: str, resume_text: str) -> str:
        """Shorter prompt for section-specific extraction."""
        # Skills often at end of resume; others at top
        t = self._truncate_for_context(resume_text, from_end=(section == "skills"))
        if section == "contact":
            return f"Resume:\n{t}\n\nExtract name, email, phone. JSON: {{\"name\":\"\",\"email\":\"\",\"phone\":\"\"}}. Output JSON only."
        if section == "education":
            return f"Resume:\n{t}\n\nExtract education. JSON: [{{\"institution\":\"\",\"degree\":\"\",\"graduation_year\":\"\"}}]. Output JSON only."
        if section == "work_experience":
            return f"Resume:\n{t}\n\nExtract work experience. JSON: [{{\"company\":\"\",\"position\":\"\",\"description\":\"\",\"duration\":\"\"}}]. Output JSON only."
        if section == "skills":
            return f"Resume:\n{t}\n\nExtract skills. JSON: [\"skill1\",\"skill2\"]. Output JSON only."
        return f"Resume:\n{t}\n"

    # Alternate key names models may use
    _KEY_ALIASES = {
        "workexperience": "work_experience",
        "work_experience": "work_experience",
        "experience": "work_experience",
        "work experience": "work_experience",
        "education": "education",
        "skills": "skills",
        "contact": "contact",
        "contacts": "contact",
        "additional_sections": "additional_sections",
        "additional": "additional_sections",
    }

    @classmethod
    def _normalize_keys(cls, obj: Any) -> Any:
        """Map alternate JSON keys to our schema."""
        if isinstance(obj, dict):
            out: Dict[str, Any] = {}
            for k, v in obj.items():
                nk = cls._KEY_ALIASES.get(k.lower().replace(" ", "_"), k)
                out[nk] = cls._normalize_keys(v)
            return out
        if isinstance(obj, list):
            return [cls._normalize_keys(x) for x in obj]
        return obj

    @staticmethod
    def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
        """Extract a JSON object from model output."""
        if not text:
            return None

        def _parse(s: str) -> Optional[Dict[str, Any]]:
            try:
                parsed = json.loads(s)
                return parsed if isinstance(parsed, dict) else None
            except Exception:
                return None

        # Try strict parse
        out = _parse(text)
        if out:
            return out

        # Try first '{' to last '}'
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            out = _parse(text[start : end + 1])
            if out:
                return out

        # Try first [...] - could be skills or education/work_experience
        b = text.find("[")
        e = text.rfind("]")
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
                        return {"skills": arr}  # fallback
            except Exception:
                pass

        # Fallback: first {...}
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            return _parse(m.group(0))
        return None

    def _empty_result(self) -> Dict[str, Any]:
        return {
            "contact": {"name": None, "email": None, "phone": None},
            "education": [],
            "work_experience": [],
            "skills": [],
            "additional_sections": [],
        }

    def _run_generation(self, prompt: str, max_new_tokens: int = 400) -> str:
        """Run model and return decoded text."""
        prompt = self._wrap_chat_prompt(prompt)
        max_len = 4096 if self._is_causal else 512
        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_len)

        output_ids = self._model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self._tokenizer.eos_token_id or self._tokenizer.pad_token_id,
        )

        if self._is_causal:
            generated_ids = output_ids[0][inputs["input_ids"].shape[1] :]
            return self._tokenizer.decode(generated_ids, skip_special_tokens=True)
        return self._tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def _merge_into(self, base: Dict[str, Any], parsed: Optional[Dict], keys: list) -> None:
        """Merge parsed values into base for given keys."""
        if not parsed or not isinstance(parsed, dict):
            return
        normalized = self._normalize_keys(parsed)
        for key in keys:
            val = normalized.get(key)
            if val is None:
                val = normalized.get("work_experience" if key == "work_experience" else key)
            if val is not None and isinstance(val, list) and val and not base.get(key):
                base[key] = val
            elif val is not None and isinstance(val, dict) and key == "contact":
                contact_map = {"name": "name", "email": "email", "phone": "phone"}
                for k, v in val.items():
                    if not v:
                        continue
                    canon = contact_map.get(k.lower(), k)
                    if not base.get("contact", {}).get(canon):
                        base.setdefault("contact", {})[canon] = v

    def extract(self, resume_text: str, max_new_tokens: int = 768) -> Dict[str, Any]:
        self._last_generation_text = None
        self._load_if_needed()
        if self._disabled or self._model is None or self._tokenizer is None:
            return self._empty_result()

        multi_step = os.getenv("TRANSFORMER_MULTI_STEP", "1").strip().lower() in {"1", "true", "yes"}
        result = self._empty_result()
        decoded_parts: list = []

        try:
            if multi_step:
                # Section-specific extraction: simpler prompts, better focus
                for section, keys in [
                    ("contact", ["contact"]),
                    ("education", ["education"]),
                    ("work_experience", ["work_experience"]),
                    ("skills", ["skills"]),
                ]:
                    prompt = self._build_section_prompt(section, resume_text)
                    decoded = self._run_generation(prompt, max_new_tokens=300)
                    decoded_parts.append(decoded)
                    parsed = self._extract_json_object(decoded)
                    self._merge_into(result, parsed, keys)
                self._last_generation_text = " | ".join(decoded_parts)
            else:
                prompt = self._build_prompt(resume_text)
                decoded = self._run_generation(prompt, max_new_tokens=max_new_tokens)
                self._last_generation_text = decoded
                parsed = self._extract_json_object(decoded)
                if parsed:
                    normalized = self._normalize_keys(parsed)
                    for k in ["contact", "education", "work_experience", "skills", "additional_sections"]:
                        v = normalized.get(k)
                        if v is not None:
                            if k == "contact" and isinstance(v, dict):
                                result["contact"].update((x, y) for x, y in v.items() if y)
                            elif isinstance(v, list) and v:
                                result[k] = v

        except Exception:
            return self._empty_result()

        return result

    @property
    def model_name(self) -> Optional[str]:
        return self._model_name

    @property
    def last_generation_text(self) -> Optional[str]:
        return self._last_generation_text

