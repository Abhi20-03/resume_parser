from __future__ import annotations

import os

from dotenv import load_dotenv
load_dotenv()
from typing import Any, Dict

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi import HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.schemas import ResumeParseResult
from app.services.text_extractor import extract_contact_hints, extract_text, truncate_text
from app.services.groq_extractor import GroqResumeExtractor
from app.services.ollama_extractor import OllamaResumeExtractor
from app.services.hf_extractor import HuggingFaceResumeExtractor
from app.services.hybrid_extractor import HybridResumeExtractor
from app.services.spacy_enricher import SpacyEnricher


def _get_extractor():
    """
    Provider selection:
    - If LLM_PROVIDER is set (hf|groq|ollama), honor it.
    - Else auto-detect: Ollama -> HF -> Groq -> Ollama default.
    """
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    if provider == "hf":
        return HuggingFaceResumeExtractor()
    if provider == "groq":
        return GroqResumeExtractor()
    if provider == "ollama":
        return OllamaResumeExtractor()

    if os.getenv("OLLAMA_MODEL", "").strip() or os.getenv("USE_OLLAMA", "").strip().lower() in ("1", "true", "yes"):
        return OllamaResumeExtractor()
    if os.getenv("HF_TOKEN", "").strip():
        return HuggingFaceResumeExtractor()
    if os.getenv("GROQ_API_KEY", "").strip():
        return GroqResumeExtractor()
    return OllamaResumeExtractor()  # Safe local default


app = FastAPI(title="Resume Parser (FastAPI + Llama)")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

templates = Jinja2Templates(directory=TEMPLATES_DIR)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

extractor = _get_extractor()
hybrid_extractor = HybridResumeExtractor()
spacy_enricher = SpacyEnricher()

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(8 * 1024 * 1024)))  # 8MB default
MAX_TEXT_CHARS = int(os.getenv("MAX_TEXT_CHARS", "12000"))


def _should_use_hybrid_fallback(result: Dict[str, Any], transformer_preview: str) -> bool:
    """
    Trigger fallback when transformer result is incomplete OR model output is empty.
    This avoids returning empty payloads when the LLM call silently fails.
    """
    if not transformer_preview.strip():
        return True
    return (
        len(result.get("education", [])) == 0
        or len(result.get("work_experience", [])) == 0
        or len(result.get("skills", [])) == 0
        or (isinstance(result.get("contact"), dict) and result["contact"].get("name") in {None, ""})
    )


def _merge_missing_fields(base: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge incoming values into base only when base is missing/empty.
    """
    out = dict(base)
    if not isinstance(out.get("contact"), dict):
        out["contact"] = {"name": None, "email": None, "phone": None}
    if not isinstance(incoming.get("contact"), dict):
        incoming_contact = {}
    else:
        incoming_contact = incoming.get("contact", {})

    for k in ("name", "email", "phone"):
        if out["contact"].get(k) in {None, ""} and incoming_contact.get(k):
            out["contact"][k] = incoming_contact.get(k)

    for key in ("education", "work_experience", "skills", "projects"):
        if not isinstance(out.get(key), list):
            out[key] = []
        if not out.get(key) and isinstance(incoming.get(key), list) and incoming.get(key):
            out[key] = incoming.get(key)
    return out


def _normalize_result_schema(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure values match exact response columns and key names.
    """
    out = dict(result or {})
    if not isinstance(out.get("contact"), dict):
        out["contact"] = {}
    out["contact"] = {
        "name": out["contact"].get("name"),
        "email": out["contact"].get("email"),
        "phone": out["contact"].get("phone"),
    }

    def _norm_education(items: Any) -> list:
        if not isinstance(items, list):
            return []
        normalized = []
        for it in items:
            if not isinstance(it, dict):
                continue
            normalized.append(
                {
                    "institution": it.get("institution"),
                    "degree": it.get("degree"),
                    "graduation_year": it.get("graduation_year"),
                }
            )
        return normalized

    def _norm_work(items: Any) -> list:
        if not isinstance(items, list):
            return []
        normalized = []
        action_prefixes = ("built ", "developed ", "performed ", "implemented ", "learned ", "designed ", "enabled ", "created ")
        for it in items:
            if not isinstance(it, dict):
                continue
            company = it.get("company")
            position = it.get("position")
            description = it.get("description")
            duration = it.get("duration")

            # Guardrail: if company looks like an action sentence, move it to description.
            if isinstance(company, str) and company.strip().lower().startswith(action_prefixes):
                if not description:
                    description = company
                company = None

            normalized.append(
                {
                    "company": company,
                    "position": position,
                    "description": description,
                    "duration": duration,
                }
            )
        return normalized

    def _norm_skills(items: Any) -> list:
        if not isinstance(items, list):
            return []
        out_skills = []
        seen = set()
        for s in items:
            if s is None:
                continue
            text = str(s).strip()
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            out_skills.append(text)
        return out_skills

    def _norm_projects(items: Any) -> list:
        if not isinstance(items, list):
            return []
        normalized = []
        for it in items:
            if not isinstance(it, dict):
                continue
            tech = it.get("tech_stack")
            if not isinstance(tech, list):
                tech = []
            normalized.append(
                {
                    "name": it.get("name"),
                    "duration": it.get("duration"),
                    "tech_stack": [str(x).strip() for x in tech if str(x).strip()],
                    "description": it.get("description"),
                }
            )
        return normalized

    out["education"] = _norm_education(out.get("education"))
    out["work_experience"] = _norm_work(out.get("work_experience"))
    out["skills"] = _norm_skills(out.get("skills"))
    out["projects"] = _norm_projects(out.get("projects"))
    # Projects is now structured, so additional_sections is not returned in final API.
    out.pop("additional_sections", None)
    return out


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> Any:
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/api/parse", response_model=ResumeParseResult)
async def parse_resume(
    file: UploadFile = File(...),
    model_hint: str = Form(default=""),
) -> ResumeParseResult:
    filename = file.filename or "resume"

    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail=f"File too large (max {MAX_UPLOAD_BYTES} bytes).")

    try:
        extracted_text = extract_text(content, filename)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    extracted_text = truncate_text(extracted_text, MAX_TEXT_CHARS)

    # Lightweight contact hints (regex) to improve reliability.
    contact_hints = extract_contact_hints(extracted_text)

    # 1) Hybrid extraction first (pre-structured hints).
    hybrid_pre = hybrid_extractor.extract(extracted_text)
    result = dict(hybrid_pre)

    # Merge regex hints into hybrid hints.
    contact = result.get("contact", {}) if isinstance(result, dict) else {}
    if contact.get("email") in {None, ""}:
        contact["email"] = contact_hints.get("email")
    if contact.get("phone") in {None, ""}:
        contact["phone"] = contact_hints.get("phone")
    result["contact"] = contact

    # 1.5) Optional spaCy enrichment for richer hints.
    result = spacy_enricher.enrich(extracted_text, result)

    # 2) Model refinement using hybrid hints + raw resume text.
    model_result = extractor.extract(extracted_text, pre_extracted=result)
    result = _merge_missing_fields(model_result if isinstance(model_result, dict) else {}, result)

    result = _normalize_result_schema(result)

    # Optionally allow UI to pass a hint.
    transformer_model = extractor.model_name
    if model_hint and transformer_model:
        transformer_model = transformer_model + f" (hint: {model_hint})"

    return ResumeParseResult(
        filename=filename,
        content_type=file.content_type,
        contact=result.get("contact", {}),
        education=result.get("education", []),
        work_experience=result.get("work_experience", []),
        skills=result.get("skills", []),
        projects=result.get("projects", []),
        transformer_model=transformer_model,
        raw_text_characters=len(extracted_text),
    )


@app.post("/api/parse-debug")
async def parse_debug(
    file: UploadFile = File(...),
    model_hint: str = Form(default=""),
) -> Dict[str, Any]:
    """
    Debug endpoint:
    - returns extracted text preview
    - returns transformer raw output preview
    - returns the same parsed JSON the UI shows
    """
    filename = file.filename or "resume"
    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail=f"File too large (max {MAX_UPLOAD_BYTES} bytes).")

    try:
        extracted_text = extract_text(content, filename)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    extracted_text = truncate_text(extracted_text, MAX_TEXT_CHARS)
    contact_hints = extract_contact_hints(extracted_text)

    # 1) Hybrid extraction first (pre-structured hints).
    hybrid_pre = hybrid_extractor.extract(extracted_text)
    result = dict(hybrid_pre)

    # Merge regex hints into hybrid hints.
    contact = result.get("contact", {}) if isinstance(result, dict) else {}
    if contact.get("email") in {None, ""}:
        contact["email"] = contact_hints.get("email")
    if contact.get("phone") in {None, ""}:
        contact["phone"] = contact_hints.get("phone")
    result["contact"] = contact

    # 1.5) Optional spaCy enrichment for richer hints.
    result = spacy_enricher.enrich(extracted_text, result)

    # 2) Model refinement using hybrid hints + raw resume text.
    model_result = extractor.extract(extracted_text, pre_extracted=result)
    result = _merge_missing_fields(model_result if isinstance(model_result, dict) else {}, result)

    result = _normalize_result_schema(result)

    transformer_model = extractor.model_name
    if model_hint and transformer_model:
        transformer_model = transformer_model + f" (hint: {model_hint})"

    decoded_preview = (extractor.last_generation_text or "").strip()[:900]

    return {
        "filename": filename,
        "content_type": file.content_type,
        #"raw_text_characters": len(extracted_text),
        "extracted_text_preview": extracted_text[:1200],
        "contact_hints": contact_hints,
        #"transformer_model": transformer_model,
        "transformer_generation_preview": decoded_preview,
        "hybrid_pre_extracted": hybrid_pre,
        "model_refined_result": model_result if isinstance(model_result, dict) else {},
        "parsed_result": result,
    }

