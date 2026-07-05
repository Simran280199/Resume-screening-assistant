# Candidate Dossier — AI Resume Screening Assistant

A RAG-powered resume screening tool built with LangChain, OpenAI, and Streamlit. Recruiters chat with an assistant to evaluate resumes against any job description, getting a structured match score, matching/missing skills, strengths, weaknesses, and a hiring recommendation — grounded strictly in the uploaded resume content.

## Features

- **Chat-based interface** — ask things like *"Evaluate all resumes"*, *"Compare Ananya and Rahul"*, *"What skills is Meera missing?"*, or *"Who's the best candidate?"*
- **Persistent Job Description library** — save JDs per role (Data Scientist, Software Engineer, Lawyer, Doctor, Judge, or any custom role), editable and reusable across sessions
- **RAG pipeline** — PDF loading → chunking → embeddings → FAISS vector search → structured LLM evaluation, so answers are grounded in actual resume text, not the model's memory
- **Deterministic scoring safeguards** — the hiring recommendation label is computed from the match score in code (not left to the LLM), and a self-consistency check with automatic retries catches cases where the model's score contradicts its own skill lists
- **Cost tracking** — every evaluation run reports real token usage and dollar cost

## Architecture

```
PDF resumes ──► PyPDFLoader ──► Text splitter ──► OpenAI embeddings ──► FAISS vector store
                                                                              │
Job description ─────────────────────────────────────────────────────────┐  │
                                                                           ▼  ▼
                                                                    Retriever (per-candidate)
                                                                           │
                                                                           ▼
                                                              Prompt template + gpt-4o-mini
                                                                    (structured output)
                                                                           │
                                                                           ▼
                                                          Pydantic-validated ResumeEvaluation
                                                                           │
                                                                           ▼
                                                              Streamlit chat UI (dossier cards)
```

## Project structure

```
├── app.py              # Streamlit chat UI - all layout, styling, and the command router
├── rag_pipeline.py      # Core RAG logic: PDF processing, embeddings, prompts, LLM calls
├── jd_library.py        # Persistent JD storage (JSON-backed, per-role)
├── requirements.txt     # Pinned dependencies
├── .gitignore           # Excludes .env, local venv, and generated data files
└── sample_data/         # Example resumes + job description for testing
```

## Setup (local)

1. Clone this repo and create a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # macOS/Linux
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the project root with your OpenAI API key:
   ```
   OPENAI_API_KEY=sk-your-key-here
   ```
4. Run the app:
   ```
   streamlit run app.py
   ```

## Deployment (Streamlit Community Cloud)

1. Push this repo to GitHub (the `.gitignore` already excludes `.env` — never commit your API key).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub, click **New app**.
3. Select this repository, branch `main`, main file `app.py`.
4. Under **Advanced settings → Secrets**, add:
   ```
   OPENAI_API_KEY = "sk-your-key-here"
   ```
5. Deploy. The app reads the key from `st.secrets` automatically when running in the cloud (see `app.py`), and from `.env` when running locally — no code changes needed between environments.

**Note on persistence:** Streamlit Cloud's filesystem is ephemeral. The default 5 job roles always load on startup, but any new roles or JD edits you make while the app is live will be lost on the next redeploy or restart. For permanent multi-user persistence, `jd_library.py` would need to point at an external database instead of a local JSON file — noted here as a known limitation, not implemented in this version.

## Model choice and cost

Uses `gpt-4o-mini` for evaluation and `text-embedding-3-small` for embeddings. An earlier version used the cheaper `gpt-4.1-nano`, but testing surfaced real consistency issues at that tier (occasional self-contradicting skill lists and match scores that didn't align with the model's own stated matching/missing skills) — documented in detail during development. `gpt-4o-mini` resolved these while remaining inexpensive: a typical evaluation costs a fraction of a cent.

## Known limitations

- **LLM numeric consistency**: a self-consistency check (`_score_is_inconsistent` in `rag_pipeline.py`) retries up to 2 times and falls back to a blended score if the match score still contradicts the skill lists after retries. This mitigates but doesn't eliminate the underlying unreliability of LLM-generated scores.
- **Command router uses keyword matching**, not full NLU — it recognizes "compare," "missing skill," "best candidate"/"recommend," and candidate name mentions (matched by individual words, tolerant of underscores/hyphens in filenames). Genuinely ambiguous or off-topic messages fall back to evaluating all uploaded resumes.
- **No persistent multi-user database** — JD library and vector stores are file-based and session-scoped, not designed for concurrent multi-user production use.

## Example test cases (from project requirements)

1. Evaluate a single resume for a given role → `evaluate <name>`
2. Compare two resumes for the same JD → `compare <name1> and <name2>`
3. Identify missing skills in a resume → `what skills is <name> missing?`
4. Recommend the best candidate among multiple → `who's the best candidate?`
5. Generate a hiring recommendation with justification → included in every evaluation card automatically
