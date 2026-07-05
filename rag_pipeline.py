"""
rag_pipeline.py
────────────────
Shared RAG logic used by both the notebook (for experimentation) and app.py
(the Streamlit UI). Keeping this in one file means we fix bugs once, not twice.
"""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

# ── Config ───────────────────────────────────────────────────────────
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0.0
MAX_OUTPUT_TOKENS = 2000
EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 6


# ── Schema ───────────────────────────────────────────────────────────
class ResumeEvaluation(BaseModel):
    candidate_name: str = Field(
        description="The candidate's full name as found in the resume. "
                    "If not found, use the filename."
    )
    match_score: int = Field(
        ge=0, le=100,
        description="Overall match score from 0-100 comparing resume to JD."
    )
    matching_skills: List[str] = Field(
        description="Skills/technologies explicitly required by the JD that "
                    "ARE present in the resume."
    )
    missing_skills: List[str] = Field(
        description="Skills/technologies explicitly required by the JD that "
                    "are NOT found anywhere in the resume."
    )
    candidate_summary: str = Field(
        description="A neutral 2-3 sentence summary of who this candidate is "
                    "professionally, based only on resume content."
    )
    strengths: List[str] = Field(
        description="3-5 specific strengths of this candidate relative to the JD."
    )
    weaknesses: List[str] = Field(
        description="3-5 specific gaps or weaknesses of this candidate relative "
                    "to the JD. Be honest, not diplomatic."
    )
    hiring_recommendation: Literal["Strong Hire", "Hire", "Maybe", "No Hire"] = Field(
        description="Base this strictly on match_score and skills gap, not tone. "
                    "Must be exactly one of the four allowed values - no extra "
                    "text, no explanation here."
    )
    justification: str = Field(
        description="2-3 sentences justifying the hiring_recommendation, "
                    "citing specific evidence from the resume. Put ALL "
                    "explanation here, NOT in hiring_recommendation."
    )


# ── PDF loading + chunking ───────────────────────────────────────────
def process_resume(file_path: str, candidate_name: str):
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    for chunk in chunks:
        chunk.metadata["candidate"] = candidate_name
    return chunks


# ── Embeddings + vector store ────────────────────────────────────────
def build_vector_store(resume_paths: dict) -> FAISS:
    """resume_paths: {candidate_name: pdf_file_path}"""
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    all_chunks = []
    for candidate_name, file_path in resume_paths.items():
        all_chunks.extend(process_resume(file_path, candidate_name))
    return FAISS.from_documents(all_chunks, embeddings)


def retrieve_relevant_chunks(vector_store, query: str, candidate_name: Optional[str] = None, k: int = TOP_K):
    if candidate_name:
        return vector_store.similarity_search(query, k=k, filter={"candidate": candidate_name})
    return vector_store.similarity_search(query, k=k)


# ── Prompt + LLM chain ───────────────────────────────────────────────
EVALUATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert technical recruiter with 15 years of experience "
     "screening resumes for technology roles. You are precise, evidence-based, "
     "and never invent information that isn't present in the provided resume "
     "excerpts. If something isn't mentioned in the resume context, treat it "
     "as ABSENT - do not assume the candidate has a skill just because it's "
     "common in the field. "
     "Search the ENTIRE resume context for evidence of each required skill - "
     "including inside job duty bullets and project descriptions, not just an "
     "explicit 'Skills' list. For example, if a bullet says 'built an NLP "
     "classifier', that counts as NLP experience even if the word 'NLP' doesn't "
     "appear in a skills section. Only mark a skill as missing if there is truly "
     "no related evidence anywhere in the provided text. "
     "Before finalizing your answer, cross-check your own lists for consistency: "
     "if a skill or technology appears in matching_skills or strengths, it must "
     "NOT also appear in missing_skills. These lists must never contradict "
     "each other."),
    ("human",
     "JOB DESCRIPTION:\n{job_description}\n\n"
     "RESUME EXCERPTS (retrieved from candidate '{candidate_name}'):\n"
     "{resume_context}\n\n"
     "Evaluate this candidate strictly against the job description above, "
     "using ONLY the resume excerpts provided. Be honest and specific - cite "
     "concrete evidence from the excerpts in your justification."),
])


def _format_chunks(chunks) -> str:
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        page = chunk.metadata.get("page", "?")
        parts.append(f"[Excerpt {i}, page {page}]\n{chunk.page_content}")
    return "\n\n".join(parts)


def get_evaluation_chain():
    """Built as a function (not a module-level global) so Streamlit's caching
    can control exactly when the LLM client gets created."""
    llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE, max_tokens=MAX_OUTPUT_TOKENS)
    structured_llm = llm.with_structured_output(ResumeEvaluation)
    return EVALUATION_PROMPT | structured_llm


def score_to_recommendation(score: int) -> str:
    """Maps match_score to a hiring_recommendation label DETERMINISTICALLY,
    in plain Python - not left up to the LLM's judgment call. Without this,
    two evaluations that land on the exact same score (e.g. both 85) can get
    different labels ('Hire' vs 'Maybe') purely because the LLM's per-call
    judgment on where the boundary sits isn't perfectly consistent, even at
    temperature 0. This guarantees the label is always a pure function of
    the score - same input, same output, every time.
    """
    if score >= 85:
        return "Strong Hire"
    if score >= 65:
        return "Hire"
    if score >= 40:
        return "Maybe"
    return "No Hire"


def _score_is_inconsistent(result: ResumeEvaluation) -> bool:
    """Sanity-checks match_score against the model's OWN matching/missing
    skill lists. Compares the actual score to what the raw skill ratio alone
    would suggest, and flags anything that deviates too far."""
    matched = len(result.matching_skills)
    missing = len(result.missing_skills)
    total = matched + missing
    if total == 0:
        return False
    expected_score = (matched / total) * 100
    return abs(result.match_score - expected_score) > 25


MAX_CONSISTENCY_RETRIES = 2  # up to 3 total LLM calls before we stop trusting it


def evaluate_resume(vector_store, evaluation_chain, job_description: str, candidate_name: str) -> ResumeEvaluation:
    chunks = retrieve_relevant_chunks(vector_store, query=job_description, candidate_name=candidate_name)
    resume_context = _format_chunks(chunks)
    inputs = {
        "job_description": job_description,
        "candidate_name": candidate_name,
        "resume_context": resume_context,
    }

    # WHY WE CATCH EXCEPTIONS HERE: an LLM call can occasionally return a
    # malformed/truncated response (e.g. hitting the token limit mid-JSON),
    # which raises a ValidationError when Pydantic tries to build
    # ResumeEvaluation from an incomplete result. Without this retry, that
    # single bad response would crash the entire Streamlit app with a raw
    # traceback visible to whoever's using it - unacceptable for anything
    # actually deployed. We retry a few times before giving up cleanly.
    result = None
    last_error = None
    for attempt in range(3):
        try:
            result = evaluation_chain.invoke(inputs)
            break
        except Exception as e:  # noqa: BLE001 - deliberately broad: any LLM/parsing failure should trigger a retry
            last_error = e
            continue

    if result is None:
        raise RuntimeError(
            f"Could not get a valid evaluation for '{candidate_name}' after 3 attempts. "
            f"This usually means the response is being truncated or malformed. "
            f"Last error: {last_error}"
        )

    # Retry up to MAX_CONSISTENCY_RETRIES times if the score contradicts the
    # model's own skill lists.
    attempts = 0
    while _score_is_inconsistent(result) and attempts < MAX_CONSISTENCY_RETRIES:
        try:
            result = evaluation_chain.invoke(inputs)
        except Exception:
            break  # keep the last valid result rather than crashing here
        attempts += 1

    # LAST RESORT: if it's STILL inconsistent after every retry, don't trust
    # the model's number at all - blend it toward the skill-ratio estimate so
    # the score can never end up wildly contradicting its own skill lists,
    # no matter how many times the model gets the number wrong.
    if _score_is_inconsistent(result):
        matched = len(result.matching_skills)
        missing = len(result.missing_skills)
        total = matched + missing
        if total > 0:
            expected_score = (matched / total) * 100
            result.match_score = round((result.match_score + expected_score) / 2)

    # Override whatever label the LLM picked with our deterministic mapping,
    # so the same score always yields the same recommendation.
    result.hiring_recommendation = score_to_recommendation(result.match_score)
    return result
