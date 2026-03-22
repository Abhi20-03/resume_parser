## Resume Parser (FastAPI + Llama)

Upload a resume in `pdf`, `docx`, or `doc` format and get extracted resume information as JSON. Supports **local Ollama**, **Groq API**, and **Hugging Face hosted models**.

### Option 1: Local Ollama (recommended if you have Ollama)

1. Ensure Ollama is running with a model pulled (e.g. `ollama run llama3.2`).
2. Install dependencies: `pip install -r requirements.txt`
3. Start the server: `uvicorn app.main:app --port 8000`
4. Open `http://localhost:8000`

No API key needed. Default model: `llama3.2`. To use a different model:
```cmd
set OLLAMA_MODEL=llama3.2
```

### Option 2: Groq API (cloud, no local model)

1. Get a free API key at [console.groq.com](https://console.groq.com/).
2. Set it: `set GROQ_API_KEY=your_key_here`
3. Install and run as above.

### Option 3: Hugging Face hosted inference (large models)

1. Create an access token on Hugging Face.
2. Set:
   - `LLM_PROVIDER=hf`
   - `HF_TOKEN=your_token`
   - optional: `HF_MODEL=meta-llama/Meta-Llama-3.1-70B-Instruct`
3. Run the app.

### Notes

- **Provider selection**: `LLM_PROVIDER=hf|groq|ollama` to force one provider.
- **Auto detection**: Ollama -> HF -> Groq -> Ollama default if `LLM_PROVIDER` is not set.
- **Hybrid fallback**: If the LLM misses fields, a rule-based extractor fills them in.
- **Flow**: Hybrid extraction runs first, then LLM refines using the hybrid JSON hints.
- **Optional spaCy enrichment**: set `SPACY_ENABLED=1` (and install model: `python -m spacy download en_core_web_sm`) to enrich names/orgs/skills before LLM refinement.
