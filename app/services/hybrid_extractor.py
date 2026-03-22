from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from app.services.text_extractor import EMAIL_RE, PHONE_RE


SECTION_HEADINGS = [
    "summary",
    "profile",
    "objective",
    "education",
    "educational qualification",
    "educational qualifications",
    "work experience",
    "work",
    "experience",
    "professional experience",
    "employment",
    "skills",
    "technical skills",
    "core skills",
    "tools",
    "certifications",
    "certificates",
    "projects",
]


EDUCATION_DEGREE_PATTERNS = [
    r"\bB\.?Tech\.?\b",
    r"\bB\.?E\.?\b",
    r"\bM\.?Tech\.?\b",
    r"\bM\.?E\.?\b",
    r"\bM\.?Sc\.?\b",
    r"\bB\.?Sc\.?\b",
    r"\bMBA\b",
    r"\bMCA\b",
    r"\bPhD\b",
    r"\bPh\.?D\.?\b",
    r"\bBachelor\b",
    r"\bMaster\b",
]

INSTITUTION_KEYWORDS = [
    "university",
    "college",
    "institute",
    "school",
    "technology",
]

POSITION_KEYWORDS = [
    "engineer",
    "developer",
    "manager",
    "lead",
    "intern",
    "analyst",
    "consultant",
    "designer",
    "architect",
    "technician",
    "research",
    "assistant",
]

COMPANY_HINT_KEYWORDS = [
    "inc",
    "ltd",
    "private",
    "llp",
    "technologies",
    "Technologies"
    "solutions",
    "systems",
    "software",
    "labs",
    "services",
    "pvt",
    "group",
]

YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")

# date ranges: 2019-2022, 2019/2022, 2019 – 2022
DATE_RANGE_RE = re.compile(r"\b(19\d{2}|20\d{2})\s*[-/–]\s*(19\d{2}|20\d{2})\b")


def _clean_line(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


def _split_nonempty_lines(text: str) -> List[str]:
    return [_clean_line(l) for l in text.splitlines() if _clean_line(l)]


def _normalize_section_key(heading: str) -> str:
    return re.sub(r"\s+", " ", heading.strip().lower())


def _detect_sections(lines: List[str]) -> Dict[str, List[str]]:
    """
    Lightweight section detector.

    Returns:
      { "education": [lines...], "work experience": [...], ... }
    """
    # Build a regex that matches common heading formats: "Education" or "EDUCATION:"
    heading_map: List[Tuple[str, re.Pattern[str]]] = []
    for h in SECTION_HEADINGS:
        key = _normalize_section_key(h)
        # Allow "Education" or "Education:" and also "Education: B.Tech ..." (heading + inline content).
        pattern = re.compile(rf"^\s*{re.escape(h)}\s*:?\s*(?P<rest>.*)$", re.IGNORECASE)
        heading_map.append((key, pattern))

    sections: Dict[str, List[str]] = {}
    current_key: Optional[str] = None
    current_lines: List[str] = []

    for line in lines:
        normalized_line = line.strip()
        matched_key: Optional[str] = None
        for key, pattern in heading_map:
            if pattern.match(normalized_line):
                matched_key = key
                break

        if matched_key:
            # flush previous
            if current_key is not None and current_lines:
                sections[current_key] = current_lines
            current_key = matched_key
            current_lines = []
            # If heading has inline content, seed the section with the remainder.
            m = None
            for k, pattern in heading_map:
                if k == matched_key:
                    m = pattern.match(normalized_line)
                    break
            if m:
                rest = m.groupdict().get("rest") or ""
                rest = _clean_line(rest)
                if rest:
                    current_lines.append(rest)
            continue

        if current_key is not None:
            current_lines.append(line)

    if current_key is not None and current_lines:
        sections[current_key] = current_lines

    return sections


def _pick_first_year(text: str) -> Optional[str]:
    m = YEAR_RE.search(text)
    return m.group(0) if m else None


def _pick_last_year(text: str) -> Optional[str]:
    years = YEAR_RE.findall(text)
    if not years:
        return None
    return years[-1]


def _extract_degree(text: str) -> Optional[str]:
    for p in EDUCATION_DEGREE_PATTERNS:
        m = re.search(p, text, flags=re.IGNORECASE)
        if m:
            return _clean_line(m.group(0))
    return None


def _extract_duration(text: str) -> Optional[str]:
    # Prefer explicit ranges
    m = DATE_RANGE_RE.search(text)
    if m:
        return f"{m.group(0).strip()}"
    y = _pick_first_year(text)
    return y


def _extract_institution(text: str) -> Optional[str]:
    low = text.lower()
    for kw in INSTITUTION_KEYWORDS:
        if kw in low:
            # Take the first sentence/line containing keyword.
            for part in re.split(r"[\n\r]+", text):
                part = _clean_line(part)
                if not part:
                    continue
                if kw in part.lower():
                    return part
    # fallback: first non-empty line
    parts = [p for p in re.split(r"[\n\r]+", text) if _clean_line(p)]
    if parts:
        return _clean_line(parts[0])
    return None


def _extract_contact_from_text(text: str) -> Dict[str, Optional[str]]:
    email_match = EMAIL_RE.search(text)
    phone_match = PHONE_RE.search(text)

    # Name heuristic: first line with alphabetic content near the top.
    lines = _split_nonempty_lines(text)
    name: Optional[str] = None
    for l in lines[:12]:
        # Skip lines that look like headings or have too many numbers
        if any(k in l.lower() for k in ["education", "experience", "skills"]):
            continue
        if len(re.findall(r"[A-Za-z]", l)) < 3:
            continue
        if EMAIL_RE.search(l) or PHONE_RE.search(l):
            continue
        name = l
        break

    return {"name": name, "email": email_match.group(0) if email_match else None, "phone": phone_match.group(0) if phone_match else None}


def _chunk_section(section_lines: List[str], max_gap_lines: int = 2) -> List[str]:
    """
    Convert section lines into rough chunks.
    We split on patterns that often indicate new entries.
    """
    chunks: List[str] = []
    current: List[str] = []
    for line in section_lines:
        # Heuristic: "•" and "-" bullets sometimes separate entries
        if line.strip().startswith(("-", "•")):
            if current:
                chunks.append("\n".join(current).strip())
                current = [line.strip("•- ").strip()]
            else:
                current = [line.strip("•- ").strip()]
            continue

        # If a new chunk likely starts with a year (start of a job/education entry)
        if YEAR_RE.search(line):
            if current:
                chunks.append("\n".join(current).strip())
                current = [line]
            else:
                current = [line]
            continue

        current.append(line)

    if current:
        chunks.append("\n".join(current).strip())

    # Remove too-short chunks
    return [c for c in chunks if len(c) >= 20]


def parse_education(education_text: str) -> List[Dict[str, Optional[str]]]:
    lines = _split_nonempty_lines(education_text)
    if not lines:
        return []
    chunks = _chunk_section(lines)
    items: List[Dict[str, Optional[str]]] = []
    for c in chunks:
        inst = _extract_institution(c)
        deg = _extract_degree(c)
        # Graduation usually corresponds to the latest year in range.
        year = _pick_last_year(c)
        # If there's no year, skip unless we clearly have a degree/institution.
        if not year and not (inst and deg):
            continue
        items.append(
            {
                "institution": inst,
                "degree": deg,
                "graduation_year": year,
            }
        )

    # Deduplicate similar entries by institution+degree+year
    seen = set()
    out: List[Dict[str, Optional[str]]] = []
    for it in items:
        key = (it.get("institution"), it.get("degree"), it.get("graduation_year"))
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


def parse_work_experience(work_text: str) -> List[Dict[str, Optional[str]]]:
    lines = _split_nonempty_lines(work_text)
    if not lines:
        return []

    # Month-year range, supports compact forms like "Jan2022–Aug2024".
    month = r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
    range_re = re.compile(
        rf"{month}\s*\d{{4}}\s*[–-]\s*(Present|Current|{month}\s*\d{{4}})",
        flags=re.IGNORECASE,
    )

    action_prefixes = (
        "built",
        "developed",
        "performed",
        "implemented",
        "learned",
        "designed",
        "enabled",
        "created",
        "architected",
        "owned",
        "enhanced",
        "automated",
        "developing",
        "building",
    )

    def _is_action_line(s: str) -> bool:
        low = s.lower().strip()
        return any(low.startswith(p) for p in action_prefixes)

    def _is_role_like(s: str) -> bool:
        low = s.lower()
        return any(k in low for k in POSITION_KEYWORDS) and len(re.findall(r"[A-Za-z]", s)) >= 4

    entries: List[Dict[str, Optional[str]]] = []
    current: Optional[Dict[str, Any]] = None
    recent_non_bullet: List[str] = []

    for raw in lines:
        line = _clean_line(raw)
        if not line:
            continue

        # Normalize bullet markers
        is_bullet = line.startswith(("•", "-"))
        if is_bullet:
            line = line.strip("•- ").strip()

        m = range_re.search(line)
        if m:
            # New employment header line found.
            if current:
                desc = "\n".join(current["desc"]).strip() if current["desc"] else None
                entries.append(
                    {
                        "company": current.get("company"),
                        "position": current.get("position"),
                        "description": desc,
                        "duration": current.get("duration"),
                    }
                )

            duration = m.group(0).strip()
            company_part = _clean_line(line[: m.start()]).strip(" ,|-")
            if not company_part:
                company_part = None

            # Pick closest prior role-like line as position.
            position = None
            for cand in reversed(recent_non_bullet[-3:]):
                if _is_role_like(cand):
                    position = cand
                    break

            current = {"company": company_part, "position": position, "duration": duration, "desc": []}
            recent_non_bullet = []
            continue

        # If we have an active job entry, attach role/details.
        if current is not None:
            if current.get("position") is None and _is_role_like(line) and not _is_action_line(line):
                current["position"] = line
                continue
            if _is_action_line(line) or is_bullet or len(line) > 35:
                current["desc"].append(line)
            continue

        # No active entry yet; cache lines for role lookup.
        if not is_bullet:
            recent_non_bullet.append(line)

    # Flush last entry
    if current:
        desc = "\n".join(current["desc"]).strip() if current["desc"] else None
        entries.append(
            {
                "company": current.get("company"),
                "position": current.get("position"),
                "description": desc,
                "duration": current.get("duration"),
            }
        )

    # Dedup + quality filter
    out: List[Dict[str, Optional[str]]] = []
    seen = set()
    for it in entries:
        if not (it.get("company") or it.get("position") or it.get("description")):
            continue
        key = (it.get("company"), it.get("position"), it.get("duration"))
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


def parse_skills(skills_text: str) -> List[str]:
    # Split on section labels + common delimiters.
    raw = skills_text.replace("\n", " ")
    raw = re.sub(
        r"\b(Languages|Machine Learning|Deep Learning|NLP|LLMs|Tools|Frameworks|Libraries)\s*:\s*",
        ",",
        raw,
        flags=re.IGNORECASE,
    )
    tokens = re.split(r"[,\;/\|•]+", raw)
    skills: List[str] = []
    for t in tokens:
        t = _clean_line(t)
        if not t:
            continue
        # Remove bullets/headers
        if t.lower() in {"skills", "technical skills", "tools", "core skills"}:
            continue
        # Keep reasonable skill tokens
        if len(t) < 2:
            continue
        # Split if token still contains "and"
        for part in re.split(r"\s+and\s+|\s*&\s*", t, flags=re.IGNORECASE):
            part = _clean_line(part)
            if not part:
                continue
            if len(part) >= 2 and part.lower() not in {"and"}:
                skills.append(part)

    # Dedup preserve order
    seen = set()
    out: List[str] = []
    for s in skills:
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def parse_additional_sections(sections: Dict[str, List[str]]) -> List[Dict[str, Any]]:
    additional: List[Dict[str, Any]] = []

    for key, lines in sections.items():
        if key in {"education", "work experience", "work", "experience", "professional experience", "skills", "technical skills", "core skills", "tools"}:
            continue

        title = key.title()
        items = []
        # Keep bullet-like lines and short lines
        for l in lines:
            ll = _clean_line(l)
            if not ll:
                continue
            if ll.startswith(("-", "•")):
                ll = ll.strip("•- ").strip()
            # Avoid very long paragraphs
            if len(ll) > 220:
                continue
            # Skip if it's clearly a date-only line
            if YEAR_RE.fullmatch(ll):
                continue
            items.append(ll)

        if items:
            additional.append({"title": title, "items": items})

    return additional


def extract_projects_fallback(text: str) -> List[str]:
    """
    Extract project lines when section detection misses/weakly captures Projects.
    """
    lines = _split_nonempty_lines(text)
    if not lines:
        return []

    # Find first line that mentions project/projects.
    start_idx = None
    for i, line in enumerate(lines):
        low = line.lower()
        if re.search(r"\bprojects?\b", low):
            start_idx = i
            break
    if start_idx is None:
        return []

    items: List[str] = []
    for j in range(start_idx + 1, min(len(lines), start_idx + 35)):
        line = lines[j]
        low = line.lower()
        # Stop when next major heading starts.
        if any(
            re.fullmatch(rf"{h}\s*:?", low)
            for h in (
                "education",
                "work experience",
                "experience",
                "skills",
                "technical skills",
                "certifications",
                "certificates",
                "summary",
                "profile",
            )
        ):
            break
        if len(line) < 4:
            continue
        if YEAR_RE.fullmatch(line):
            continue
        if line.startswith(("-", "•")):
            line = line.strip("•- ").strip()
        # Keep concise project bullets/details.
        if 4 <= len(line) <= 260:
            items.append(line)

    # Dedup preserve order
    seen = set()
    out: List[str] = []
    for item in items:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def parse_projects(project_items: List[str]) -> List[Dict[str, Any]]:
    """
    Convert flat project lines into structured project entries.
    Heuristic:
    - New project starts on lines containing a year (e.g., 2025/2024)
      or title-like short lines.
    - Tech stack lines often contain commas and known tools.
    """
    if not project_items:
        return []

    tool_tokens = {
        "python",
        "fastapi",
        "streamlit",
        "pytorch",
        "hugging face",
        "transformers",
        "lora",
        "rag",
        "docker",
        "ollama",
        "nlp",
        "genai",
        "gemini",
    }

    projects: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None

    def _start_new(name_line: str) -> Dict[str, Any]:
        year = _pick_last_year(name_line)
        duration = year if year else None
        return {"name": name_line, "duration": duration, "tech_stack": [], "description": ""}

    for line in project_items:
        line = _clean_line(line)
        if not line:
            continue

        is_new_header = bool(YEAR_RE.search(line)) and len(line) < 90
        if is_new_header:
            if current:
                current["description"] = current["description"].strip() or None
                projects.append(current)
            current = _start_new(line)
            continue

        if current is None:
            current = _start_new(line)
            continue

        # tech stack detection
        low = line.lower()
        comma_parts = [p.strip() for p in line.split(",") if p.strip()]
        if len(comma_parts) >= 2 and any(tok in low for tok in tool_tokens):
            for p in comma_parts:
                if p and p.lower() not in {x.lower() for x in current["tech_stack"]}:
                    current["tech_stack"].append(p)
            continue

        # description
        if current["description"]:
            current["description"] += "\n" + line
        else:
            current["description"] = line

    if current:
        current["description"] = current["description"].strip() or None
        projects.append(current)

    return projects


@dataclass
class HybridExtractResult:
    data: Dict[str, Any]


class HybridResumeExtractor:
    """
    Hybrid extraction technique inspired by rule-based + NER pipeline approaches:
    - Section detection via headings
    - Rule-based parsing for Education/Experience/Skills
    - Regex contact extraction
    - Optional transformer fallback handled outside this class
    """

    def extract(self, extracted_text: str) -> Dict[str, Any]:
        text = extracted_text or ""
        contact = _extract_contact_from_text(text)

        lines = _split_nonempty_lines(text)
        sections = _detect_sections(lines)

        # Map section keys to canonical categories
        education_key = None
        for k in ("education", "educational qualification", "educational qualifications"):
            if k in sections:
                education_key = k
                break

        work_key = None
        for k in ("work experience", "professional experience", "experience", "work"):
            if k in sections:
                work_key = k
                break

        skills_key = None
        for k in ("skills", "technical skills", "core skills", "tools"):
            if k in sections:
                skills_key = k
                break

        education_text = "\n".join(sections.get(education_key, [])) if education_key else ""
        work_text = "\n".join(sections.get(work_key, [])) if work_key else ""
        skills_text = "\n".join(sections.get(skills_key, [])) if skills_key else ""

        education = parse_education(education_text) if education_text else []
        work_experience = parse_work_experience(work_text) if work_text else []
        skills = parse_skills(skills_text) if skills_text else []

        additional_sections = parse_additional_sections(sections)

        # Projects fallback: if no "Projects" section was extracted, attempt recovery.
        has_projects = any(
            isinstance(s, dict) and str(s.get("title", "")).strip().lower() == "projects"
            for s in additional_sections
        )
        if not has_projects:
            project_items = extract_projects_fallback(text)
            if project_items:
                additional_sections.append({"title": "Projects", "items": project_items})

        # Build structured projects from Projects additional section when present.
        project_items: List[str] = []
        for sec in additional_sections:
            if str(sec.get("title", "")).strip().lower() == "projects":
                if isinstance(sec.get("items"), list):
                    project_items = [str(x) for x in sec.get("items", []) if str(x).strip()]
                break
        projects = parse_projects(project_items)

        return {
            "contact": contact,
            "education": education,
            "work_experience": work_experience,
            "skills": skills,
            "projects": projects,
            "additional_sections": additional_sections,
        }

