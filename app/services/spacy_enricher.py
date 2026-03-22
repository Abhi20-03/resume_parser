from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional


SKILL_TERMS = {
    "python",
    "java",
    "c++",
    "golang",
    "go",
    "docker",
    "kubernetes",
    "fastapi",
    "react",
    "node.js",
    "nodejs",
    "pytorch",
    "tensorflow",
    "langchain",
    "rag",
    "faiss",
    "transformers",
    "llm",
    "llms",
    "azure",
    "aws",
    "gcp",
    "git",
    "airflow",
    "jenkins",
    "terraform",
}

ROLE_KEYWORDS = (
    "engineer",
    "developer",
    "intern",
    "manager",
    "analyst",
    "architect",
    "lead",
    "consultant",
)


class SpacyEnricher:
    """
    Optional spaCy enrich step to improve structured extraction.
    Enabled via SPACY_ENABLED=1.
    """

    def __init__(self) -> None:
        self._enabled = os.getenv("SPACY_ENABLED", "").strip().lower() in {"1", "true", "yes"}
        self._model_name = os.getenv("SPACY_MODEL", "").strip() or "en_core_web_sm"
        self._nlp = None
        self._load_failed = False

    def _load_if_needed(self) -> None:
        if not self._enabled or self._load_failed or self._nlp is not None:
            return
        try:
            import spacy
            self._nlp = spacy.load(self._model_name)
        except Exception:
            self._load_failed = True
            self._nlp = None

    @staticmethod
    def _norm_spaces(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _is_role_like(text: Optional[str]) -> bool:
        if not text:
            return False
        low = text.lower()
        return any(k in low for k in ROLE_KEYWORDS)

    @staticmethod
    def _strip_date_tokens(text: Optional[str]) -> Optional[str]:
        if not text:
            return text
        # Remove compact or spaced month-year ranges from title/company fields.
        cleaned = re.sub(
            r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\s*\d{4}\s*[–-]\s*(?:Present|Current|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\s*\d{4})\b",
            "",
            text,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"\b(19|20)\d{2}\b", "", cleaned)
        cleaned = re.sub(r"[,\-|–]{2,}", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,-|–")
        return cleaned or None

    def _split_company_position(self, company: Optional[str], position: Optional[str], orgs: List[str]) -> tuple[Optional[str], Optional[str]]:
        """
        Separate mixed company/position values using spaCy ORG entities + role keywords.
        """
        company = self._norm_spaces(company or "") if company else None
        position = self._norm_spaces(position or "") if position else None

        # Clean date tokens from existing fields.
        company = self._strip_date_tokens(company)
        position = self._strip_date_tokens(position)

        # If same text in both fields and role-like, keep as position, look for company in ORGs.
        if company and position and company == position and self._is_role_like(position):
            for org in orgs:
                if org and org.lower() not in position.lower():
                    return org, position
            return None, position

        # If company looks like role text, move to position.
        if company and not position and self._is_role_like(company):
            position = company
            company = None

        # Split comma-separated mixed value.
        if company and "," in company and not position:
            parts = [self._norm_spaces(p) for p in company.split(",") if self._norm_spaces(p)]
            if len(parts) >= 2:
                role_part = next((p for p in parts if self._is_role_like(p)), None)
                org_part = next((p for p in parts if not self._is_role_like(p)), None)
                if role_part:
                    position = role_part
                if org_part:
                    company = org_part

        # Prefer ORG entity if company missing or still role-like.
        if (not company or self._is_role_like(company)) and orgs:
            for org in orgs:
                if not org:
                    continue
                if position and org.lower() in position.lower():
                    continue
                company = org
                break

        return company, position

    def enrich(self, extracted_text: str, parsed: Dict[str, Any]) -> Dict[str, Any]:
        self._load_if_needed()
        if not self._enabled or self._nlp is None:
            return parsed

        out = dict(parsed or {})
        if not isinstance(out.get("contact"), dict):
            out["contact"] = {"name": None, "email": None, "phone": None}
        if not isinstance(out.get("skills"), list):
            out["skills"] = []
        if not isinstance(out.get("education"), list):
            out["education"] = []
        if not isinstance(out.get("work_experience"), list):
            out["work_experience"] = []

        doc = self._nlp(extracted_text or "")

        persons: List[str] = []
        orgs: List[str] = []
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                persons.append(self._norm_spaces(ent.text))
            elif ent.label_ == "ORG":
                orgs.append(self._norm_spaces(ent.text))

        # Fill missing name from first PERSON near top.
        if out["contact"].get("name") in {None, ""} and persons:
            out["contact"]["name"] = persons[0]

        # Skill enrichment by exact token scan.
        text_low = (extracted_text or "").lower()
        existing = {str(s).strip().lower() for s in out["skills"] if str(s).strip()}
        for term in SKILL_TERMS:
            if re.search(rf"\b{re.escape(term)}\b", text_low):
                pretty = "LLMs" if term == "llms" else ("LLM" if term == "llm" else term)
                if pretty.lower() not in existing:
                    out["skills"].append(pretty)
                    existing.add(pretty.lower())

        # Resolve company/position mix-ups using ORG entities + role heuristics.
        org_idx = 0
        for item in out["work_experience"]:
            if not isinstance(item, dict):
                continue
            c, p = self._split_company_position(item.get("company"), item.get("position"), orgs[org_idx:])
            item["company"] = c
            item["position"] = p
            if c and org_idx < len(orgs) and c == orgs[org_idx]:
                org_idx += 1

        # If education empty, try adding likely education orgs.
        if not out["education"]:
            edu_orgs = [o for o in orgs if any(k in o.lower() for k in ("college", "university", "institute", "school"))]
            for o in edu_orgs[:2]:
                out["education"].append({"institution": o, "degree": None, "graduation_year": None})

        return out

