# Resume Parser Project Report

## 1. Project Overview

This repository implements a resume parsing web application using FastAPI. Users upload resume files in PDF, DOCX, or DOC format and receive structured JSON containing contact details, education, work experience, skills, and projects.

The project uses a hybrid extraction approach:

- text extraction from uploaded files
- rule-based parsing for core sections
- optional spaCy enrichment
- LLM-based refinement through Ollama, Groq, or Hugging Face

This design keeps the system usable even when the selected LLM provider is unavailable or produces incomplete output.

## 2. Main Goals

- Accept resume uploads through a browser UI or API
- Extract readable text from common resume file formats
- Convert unstructured text into structured resume fields
- Improve extraction quality with heuristics and optional NLP enrichment
- Support multiple LLM backends with environment-based configuration

## 3. Technology Stack

- Backend framework: FastAPI
- Server: Uvicorn
- Templates: Jinja2
- File parsing:
  - `pdfplumber` for PDF
  - `python-docx` for DOCX
  - `textract` as a best-effort optional DOC parser
- NLP:
  - rule-based parsing in custom Python code
  - optional spaCy enrichment
- LLM providers:
  - local Ollama
  - Groq API
  - Hugging Face Inference API

## 4. High-Level Architecture

The project flow is:

1. User uploads a resume through `/` or `/api/parse`
2. `app/services/text_extractor.py` converts the file into plain text
3. `app/services/hybrid_extractor.py` performs rule-based section and entity extraction
4. `app/services/spacy_enricher.py` optionally enriches the parsed result
5. The selected LLM extractor refines and completes the structured result
6. `app/main.py` normalizes the schema and returns the final response

This means the system is not purely LLM-driven. The rule-based parser is the first-stage extractor and the LLM acts as a refinement layer.

## 5. API Surface

### `GET /`

Serves the HTML upload interface.

### `GET /health`

Simple health check endpoint returning:

```json
{"status": "ok"}
```

### `POST /api/parse`

Primary API endpoint. Accepts a resume file and returns a `ResumeParseResult` object with:

- `filename`
- `content_type`
- `contact`
- `education`
- `work_experience`
- `skills`
- `projects`
- `transformer_model`
- `raw_text_characters`

### `POST /api/parse-debug`

Debug endpoint that returns:

- extracted text preview
- contact hints
- transformer generation preview
- hybrid pre-extracted data
- model-refined result
- final parsed result

## 6. Core Modules

### `app/main.py`

This is the main application entrypoint. It:

- loads environment variables from `.env`
- selects the LLM extractor with `_get_extractor()`
- creates FastAPI routes
- handles upload size limits and text truncation
- merges rule-based and model-based results
- normalizes the response schema

Provider selection logic:

- explicit `LLM_PROVIDER=hf|groq|ollama`
- otherwise auto-detects based on available environment variables
- defaults to Ollama when nothing is configured

### `app/services/text_extractor.py`

Responsible for raw text extraction from uploaded files.

- PDF parsing uses `pdfplumber`
- DOCX parsing uses `python-docx`
- DOC parsing uses `textract` if installed
- regexes extract email and phone hints

This module is the document parser layer, not the semantic resume parser.

### `app/services/hybrid_extractor.py`

This is the main semantic parser in the project.

It performs:

- section detection using common headings
- contact extraction with regex and top-of-document heuristics
- education parsing using degree and year patterns
- work experience parsing using duration and role heuristics
- skills parsing using delimiter and label normalization
- project extraction with fallback recovery

This module is the primary parser used before any LLM refinement.

### `app/services/spacy_enricher.py`

Optional enrichment layer enabled via `SPACY_ENABLED=1`.

It can:

- infer person and organization entities
- improve missing contact names
- enrich skills from known term lists
- fix mixed company and position values
- backfill education institutions from detected organizations

### `app/services/ollama_extractor.py`

Local LLM refinement using the Ollama chat API.

- default model constant is `llava:latest`
- base URL defaults to `http://localhost:11434`
- prompt includes both raw resume text and pre-extracted hints
- output is expected to be valid JSON only

### `app/services/groq_extractor.py`

Cloud LLM refinement using the Groq API.

- requires `GROQ_API_KEY`
- default model is `llama-3.1-8b-instant`
- includes defensive JSON extraction and key normalization

### `app/services/hf_extractor.py`

Cloud LLM refinement through Hugging Face hosted inference.

- requires `HF_TOKEN`
- optional model via `HF_MODEL`
- uses HTTP calls directly with `urllib`

### `app/services/transformer_extractor.py`

Legacy or alternative local transformer-based extractor using `transformers`.

It appears to be available in the repository but is not wired into `app/main.py` for the active application flow.

### `app/services/hf_ner_extractor.py`

Additional NER-related support code exists in the repository, but it is not part of the main request flow exposed by `app/main.py`.

## 7. Data Schema

The response schema is defined in `app/schemas.py`.

### Contact

- `name`
- `email`
- `phone`

### Education Item

- `institution`
- `degree`
- `graduation_year`

### Work Experience Item

- `company`
- `position`
- `description`
- `duration`

### Project Item

- `name`
- `duration`
- `tech_stack`
- `description`

## 8. Frontend

The frontend is minimal and server-rendered.

- template: `app/templates/index.html`
- styling: `app/static/style.css`
- client logic: `app/static/app.js`

Its job is primarily to upload files and display the extracted JSON result.

## 9. Environment Configuration

Key environment variables used by the application:

- `LLM_PROVIDER`
- `OLLAMA_MODEL`
- `OLLAMA_BASE_URL`
- `GROQ_API_KEY`
- `GROQ_MODEL`
- `HF_TOKEN`
- `HF_MODEL`
- `SPACY_ENABLED`
- `SPACY_MODEL`
- `MAX_UPLOAD_BYTES`
- `MAX_TEXT_CHARS`

The app loads configuration through `python-dotenv`, so values can be stored in `.env`.

## 10. Strengths

- Clear separation between text extraction, heuristic parsing, and model refinement
- Works with multiple LLM providers
- Has a fallback-first architecture instead of depending only on model output
- Includes a debug endpoint that helps inspect extraction stages
- Schema normalization reduces output inconsistency

## 11. Risks and Limitations

- Heuristic parsing may break on uncommon resume layouts
- DOC support depends on extra optional tooling through `textract`
- No automated tests are present in the visible repository
- Default Ollama model in code is `llava:latest`, while the README examples reference `llama3.2`; that mismatch can confuse setup
- Some text contains encoding artifacts such as `–` and `•`, which suggests source text normalization could be improved
- Legacy extractor modules remain in the codebase, which may create maintenance ambiguity

## 12. Suggested Improvements

- Add unit tests for text extraction, hybrid parsing, and schema normalization
- Add integration tests for `/api/parse`
- Align README defaults with the actual default Ollama model in code
- Standardize text cleanup to handle encoding artifacts before parsing
- Add logging around provider selection and LLM failures
- Consider moving inactive extractor implementations behind explicit feature flags or documenting them as experimental

## 13. File Inventory Summary

- `app/main.py`: API entrypoint and orchestration
- `app/schemas.py`: response schema definitions
- `app/services/text_extractor.py`: file-to-text extraction
- `app/services/hybrid_extractor.py`: primary rule-based parser
- `app/services/spacy_enricher.py`: optional enrichment
- `app/services/ollama_extractor.py`: Ollama refinement
- `app/services/groq_extractor.py`: Groq refinement
- `app/services/hf_extractor.py`: Hugging Face refinement
- `app/services/transformer_extractor.py`: alternative local transformer implementation
- `app/services/hf_ner_extractor.py`: additional NER support module
- `app/templates/index.html`: UI template
- `app/static/app.js`: frontend client logic
- `app/static/style.css`: frontend styling

## 14. Conclusion

This project is a practical hybrid resume parser built around FastAPI. Its most important design choice is the combination of deterministic heuristics and optional LLM refinement. That approach improves reliability compared with a pure model-driven parser and makes the application flexible across local and cloud inference providers.

For a small-to-medium engineering assignment or portfolio project, the architecture is sensible and easy to explain. The next step to improve production readiness would be stronger test coverage, tighter configuration consistency, and clearer separation between active and experimental extractor modules.
