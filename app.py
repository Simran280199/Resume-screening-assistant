"""
app.py
──────
Chat-style Streamlit UI for the resume screening assistant. Same underlying
RAG logic as before (rag_pipeline.py, jd_library.py) - only the interaction
model changed, from "fill a form" to "type a request".

ARCHITECTURE NOTE:
This file adds exactly one new piece of logic beyond form-vs-chat plumbing:
a small COMMAND ROUTER (route_message) that reads the recruiter's plain-
English message and decides which rag_pipeline function to call. Everything
downstream of that decision is identical to the form version.
"""

import os
import tempfile
import html

from dotenv import load_dotenv
load_dotenv()

import streamlit as st

# WHY THIS BLOCK: locally, the API key comes from your .env file via
# load_dotenv() above. On Streamlit Community Cloud, there is no .env file -
# secrets are set through the app's dashboard instead, and surface inside
# your code as st.secrets, NOT as an environment variable automatically.
# LangChain's OpenAI client only looks at the OPENAI_API_KEY environment
# variable, so if we're running in the cloud, we copy it from st.secrets
# into os.environ ourselves.
if not os.getenv("OPENAI_API_KEY"):
    try:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    except (KeyError, FileNotFoundError):
        st.error(
            "OPENAI_API_KEY not found. Locally: add it to your .env file. "
            "On Streamlit Cloud: add it under your app's Settings -> Secrets."
        )
        st.stop()

from langchain_community.callbacks import get_openai_callback

import jd_library as jdlib
from rag_pipeline import build_vector_store, get_evaluation_chain, evaluate_resume


st.set_page_config(page_title="Candidate Dossier — Resume Screening", layout="wide")


# ── Design system (unchanged from before, extended for chat bubbles) ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
    --paper: #FAF8F3; --paper-dim: #F1EEE5; --ink: #1F2421; --ink-soft: #55584f;
    --line: #D9D3C7; --navy: #263042; --hire: #2F6F5E; --maybe: #B8862B; --nohire: #A23E32;
}

.stApp { background-color: var(--paper); }
.stApp, .stApp p, .stApp label, .stApp span { font-family: 'Inter', sans-serif; color: var(--ink); }
h1, h2, h3 { font-family: 'Fraunces', serif !important; color: var(--ink) !important; font-weight: 600 !important; }

.dossier-masthead { border-bottom: 2px solid var(--ink); padding-bottom: 14px; margin-bottom: 20px; }
.dossier-masthead .eyebrow {
    font-family: 'IBM Plex Mono', monospace; letter-spacing: 0.12em; text-transform: uppercase;
    font-size: 0.72rem; color: var(--ink-soft);
}
.dossier-masthead h1 { margin: 4px 0 0 0 !important; font-size: 2.1rem !important; }

[data-testid="stSidebar"] { background-color: var(--navy); }
[data-testid="stSidebar"] * { color: var(--paper) !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    font-family: 'Fraunces', serif !important; color: var(--paper) !important;
}
[data-testid="stSidebar"] .stTextArea textarea,
[data-testid="stSidebar"] .stTextInput input,
[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {
    background-color: #1B2331 !important; border: 1px solid #3C4A5F !important;
    color: var(--paper) !important; font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.85rem !important;
}
[data-testid="stSidebar"] .stButton button,
[data-testid="stSidebar"] .stButton button * {
    background-color: var(--paper); color: var(--navy) !important;
    border: none; border-radius: 2px; font-weight: 600;
}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
    background-color: #1B2331 !important; border: 1.5px dashed #3C4A5F !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button,
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button * {
    color: var(--navy) !important; background-color: var(--paper) !important;
}

/* Chat bubbles */
[data-testid="stChatMessage"] { background: transparent; }
[data-testid="stChatMessageContent"] {
    background: #FFFFFF; border: 1px solid var(--line); border-radius: 8px; padding: 4px 6px;
}

/* Dossier card (rendered inside assistant chat bubbles) */
.dossier-card {
    background: #FFFFFF; border: 1px solid var(--line); border-left: 5px solid var(--score-color, var(--ink));
    border-radius: 3px; padding: 18px 22px; margin: 6px 0;
}
.dossier-card .card-top { display: flex; justify-content: space-between; align-items: flex-start; }
.dossier-card .candidate-name { font-family: 'Fraunces', serif; font-size: 1.2rem; font-weight: 600; margin: 0; }
.dossier-card .score-block { font-family: 'IBM Plex Mono', monospace; text-align: right; }
.dossier-card .score-num { font-size: 1.8rem; font-weight: 600; line-height: 1; color: var(--score-color, var(--ink)); }
.dossier-card .score-label { font-size: 0.65rem; letter-spacing: 0.08em; text-transform: uppercase; color: var(--ink-soft); }

.stamp {
    display: inline-block; font-family: 'IBM Plex Mono', monospace; font-weight: 600;
    letter-spacing: 0.08em; text-transform: uppercase; font-size: 0.74rem; padding: 4px 12px;
    border: 2.5px solid var(--stamp-color, var(--ink)); color: var(--stamp-color, var(--ink));
    border-radius: 3px; transform: rotate(-3deg); margin-top: 6px;
}

.dossier-summary { color: var(--ink-soft); font-size: 0.9rem; margin: 12px 0 14px 0; font-style: italic; }
.chip-row { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 4px; }
.chip { font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; padding: 3px 10px; border-radius: 12px; border: 1px solid; }
.chip-match { color: var(--hire); border-color: var(--hire); background: #EEF5F2; }
.chip-missing { color: var(--nohire); border-color: var(--nohire); background: #F8EEEC; }
.dossier-card ul { margin: 4px 0 0 0; padding-left: 18px; font-size: 0.9rem; }
.dossier-card .col-label {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; letter-spacing: 0.08em;
    text-transform: uppercase; color: var(--ink-soft); margin-bottom: 6px; display: block;
}
.justification-block { border-left: 3px solid var(--line); padding-left: 12px; margin-top: 12px; font-size: 0.86rem; color: var(--ink-soft); }

.chip-name {
    display: inline-block; font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem;
    padding: 3px 10px; border-radius: 12px; border: 1px solid var(--line); background: var(--paper-dim);
    margin: 2px 4px 2px 0;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="dossier-masthead">
    <div class="eyebrow">RAG-powered candidate screening — chat mode</div>
    <h1>Candidate Dossier</h1>
</div>
""", unsafe_allow_html=True)


# ── Cached resources ─────────────────────────────────────────────────
@st.cache_resource
def get_chain():
    return get_evaluation_chain()


# ── Session state ────────────────────────────────────────────────────
# WHY THESE FOUR KEYS SPECIFICALLY:
#   messages        -> the visible chat transcript (list of {role, content})
#   vector_store     -> built once per unique set of uploaded resumes
#   evaluated_cache  -> {candidate_name: ResumeEvaluation}, so re-asking about
#                        someone already evaluated costs zero extra API calls
#   uploaded_names   -> lets us detect when the uploaded file SET changes,
#                        so we know when to rebuild the vector store / cache
for key, default in [
    ("messages", []),
    ("vector_store", None),
    ("evaluated_cache", {}),
    ("uploaded_names", []),
    ("last_jd_fingerprint", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ── Sidebar: Case File (role/JD) + Candidates (uploads) ──────────────
st.sidebar.markdown("### Case File")
st.sidebar.caption("Select or edit the role you're hiring for.")

roles = jdlib.list_roles()
selected_role = st.sidebar.selectbox("Role", roles, label_visibility="collapsed")
current_jd_text = jdlib.get_jd(selected_role)
edited_jd_text = st.sidebar.text_area(
    "Job description", value=current_jd_text, height=200, label_visibility="collapsed"
)
if st.sidebar.button("Save changes to this role"):
    jdlib.add_or_update_jd(selected_role, edited_jd_text)
    st.sidebar.success(f"Updated '{selected_role}'.")

# WHY THIS MATTERS: without this check, evaluated_cache only gets cleared
# when the uploaded FILE set changes - so switching roles or editing the JD
# text while keeping the same resumes uploaded would silently serve back
# stale evaluations computed against the OLD job description.
jd_fingerprint = f"{selected_role}::{hash(edited_jd_text)}"
if st.session_state.get("last_jd_fingerprint") != jd_fingerprint:
    st.session_state["evaluated_cache"] = {}
    st.session_state["last_jd_fingerprint"] = jd_fingerprint

with st.sidebar.expander("+ Add a new role"):
    new_role_name = st.text_input("Role name", key="new_role_name")
    new_role_jd = st.text_area("Job description", key="new_role_jd")
    if st.button("Add role"):
        if new_role_name.strip() and new_role_jd.strip():
            jdlib.add_or_update_jd(new_role_name.strip(), new_role_jd.strip())
            st.success(f"Added '{new_role_name}'. Select it above.")
        else:
            st.warning("Provide both a role name and a job description.")

st.sidebar.markdown("### Candidates")
uploaded_files = st.sidebar.file_uploader(
    "Upload resume PDFs", type=["pdf"], accept_multiple_files=True, label_visibility="collapsed"
)

if uploaded_files:
    current_names = sorted(f.name for f in uploaded_files)
    # Only rebuild the (expensive) vector store if the SET of uploaded files
    # actually changed - avoids re-embedding on every chat message rerun.
    if current_names != st.session_state["uploaded_names"]:
        with tempfile.TemporaryDirectory() as tmp_dir:
            resume_paths = {}
            for uploaded_file in uploaded_files:
                candidate_name = os.path.splitext(uploaded_file.name)[0]
                file_path = os.path.join(tmp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                resume_paths[candidate_name] = file_path

            with st.spinner("Embedding resumes..."):
                st.session_state["vector_store"] = build_vector_store(resume_paths)

            st.session_state["uploaded_names"] = current_names
            st.session_state["evaluated_cache"] = {}  # new resumes -> stale cache
            st.session_state["candidate_names"] = [
                os.path.splitext(n)[0] for n in current_names
            ]

    st.sidebar.markdown("".join(
        f'<span class="chip-name">{html.escape(n)}</span>'
        for n in st.session_state.get("candidate_names", [])
    ), unsafe_allow_html=True)
else:
    st.session_state["vector_store"] = None
    st.session_state["uploaded_names"] = []
    st.session_state["candidate_names"] = []


# ── Rendering helpers (same visual system as the form version) ───────
def _clean_html(s: str) -> str:
    """Streamlit's markdown renderer treats lines with 4+ leading spaces as
    a preformatted code block, showing raw tags as text instead of rendering
    them as HTML. Our f-strings pick up Python's own indentation, so we
    strip leading whitespace from every line before handing HTML to
    st.markdown()."""
    return "\n".join(line.strip() for line in s.strip("\n").splitlines())


STAMP_COLORS = {"Strong Hire": "var(--hire)", "Hire": "var(--hire)", "Maybe": "var(--maybe)", "No Hire": "var(--nohire)"}


def render_missing_skills_only(result) -> str:
    """A compact view for 'what skills is X missing' queries - just the
    name, score, and missing skills, not the full dossier card."""
    return _clean_html(f"""
    <div style="border-left: 3px solid var(--nohire); padding: 8px 16px; margin: 6px 0; background: #FFFFFF;">
        <span class="col-label">{html.escape(result.candidate_name)} — {result.match_score}/100</span>
        {render_chips(result.missing_skills, "chip-missing")}
    </div>
    """)


def score_color(score: int) -> str:
    if score >= 70:
        return "var(--hire)"
    if score >= 40:
        return "var(--maybe)"
    return "var(--nohire)"


def render_chips(items, css_class):
    if not items:
        return '<span style="color: var(--ink-soft); font-size: 0.85rem;">None noted</span>'
    return '<div class="chip-row">' + "".join(
        f'<span class="chip {css_class}">{html.escape(s)}</span>' for s in items
    ) + "</div>"


def render_list(items):
    if not items:
        return '<span style="color: var(--ink-soft); font-size: 0.85rem;">None noted</span>'
    return "<ul>" + "".join(f"<li>{html.escape(s)}</li>" for s in items) + "</ul>"


def render_card(result) -> str:
    s_color = score_color(result.match_score)
    stamp_color = STAMP_COLORS.get(result.hiring_recommendation, "var(--ink)")
    return _clean_html(f"""
    <div class="dossier-card" style="--score-color: {s_color};">
        <div class="card-top">
            <div>
                <p class="candidate-name">{html.escape(result.candidate_name)}</p>
                <span class="stamp" style="--stamp-color: {stamp_color};">{html.escape(result.hiring_recommendation)}</span>
            </div>
            <div class="score-block">
                <div class="score-num">{result.match_score}</div>
                <div class="score-label">match / 100</div>
            </div>
        </div>
        <p class="dossier-summary">{html.escape(result.candidate_summary)}</p>
        <div style="display: flex; gap: 28px; flex-wrap: wrap;">
            <div style="flex: 1; min-width: 220px;">
                <span class="col-label">Matching skills</span>
                {render_chips(result.matching_skills, "chip-match")}
            </div>
            <div style="flex: 1; min-width: 220px;">
                <span class="col-label">Missing skills</span>
                {render_chips(result.missing_skills, "chip-missing")}
            </div>
        </div>
        <div style="display: flex; gap: 28px; flex-wrap: wrap; margin-top: 14px;">
            <div style="flex: 1; min-width: 220px;">
                <span class="col-label">Strengths</span>
                {render_list(result.strengths)}
            </div>
            <div style="flex: 1; min-width: 220px;">
                <span class="col-label">Weaknesses</span>
                {render_list(result.weaknesses)}
            </div>
        </div>
        <div class="justification-block">{html.escape(result.justification)}</div>
    </div>
    """)


def render_comparison_table(evaluations) -> str:
    ranked = sorted(evaluations, key=lambda r: r.match_score, reverse=True)
    rows = "".join(
        f"<tr><td style='padding:4px 10px;'>{html.escape(r.candidate_name)}</td>"
        f"<td style='padding:4px 10px; font-family: IBM Plex Mono, monospace;'>{r.match_score}</td>"
        f"<td style='padding:4px 10px;'>{html.escape(r.hiring_recommendation)}</td></tr>"
        for r in ranked
    )
    return _clean_html(f"""
    <table style="width:100%; border-collapse: collapse; font-size: 0.88rem;">
        <thead><tr style="border-bottom: 1px solid var(--line);">
            <th style="text-align:left; padding:4px 10px;">Candidate</th>
            <th style="text-align:left; padding:4px 10px;">Score</th>
            <th style="text-align:left; padding:4px 10px;">Recommendation</th>
        </tr></thead>
        <tbody>{rows}</tbody>
    </table>
    """)


# ── Ensure evaluations exist (cache-aware, only calls the LLM when needed) ─
def ensure_evaluated(names):
    """Evaluates any candidate in `names` not already in the cache. This is
    the piece that makes chat re-queries free: asking about the same
    candidate twice in one session only calls the LLM once."""
    chain = get_chain()
    to_run = [n for n in names if n not in st.session_state["evaluated_cache"]]
    if not to_run:
        return 0.0, 0

    total_cost, total_tokens = 0.0, 0
    with get_openai_callback() as cb:
        for name in to_run:
            result = evaluate_resume(
                st.session_state["vector_store"], chain, edited_jd_text, name
            )
            st.session_state["evaluated_cache"][name] = result
        total_cost, total_tokens = cb.total_cost, cb.total_tokens
    return total_cost, total_tokens


import re


def _normalize(s: str) -> str:
    """Turns 'Simran_Rani_Resume_2026' or 'niharika-nath' into space-separated
    lowercase words, so filename punctuation doesn't break name matching."""
    return re.sub(r"[_\-]+", " ", s).lower()


def find_mentioned_candidates(user_text: str, candidates: list) -> list:
    """Matches candidates by WORD overlap, not exact substring - so 'Simran'
    alone matches 'Simran_Rani_Resume_2026', and 'niharika' matches
    'Niharika Nath DS' even though the message uses different punctuation
    or omits the rest of the name."""
    text_words = set(_normalize(user_text).split())
    mentioned = []
    for c in candidates:
        candidate_words = [w for w in _normalize(c).split() if len(w) > 2]
        if any(w in text_words for w in candidate_words):
            mentioned.append(c)
    return mentioned


# ── Command router: plain-English message -> which action to take ────
def route_message(user_text: str) -> str:
    """Reads the recruiter's message and decides which rag_pipeline action
    to run, returning HTML/markdown to display as the assistant's reply.
    This is the ONE new piece of logic the chat interface needed - every
    downstream function call is identical to the form-based version."""
    candidates = st.session_state.get("candidate_names", [])
    if not candidates:
        return "Please upload at least one resume PDF in the sidebar first, then ask me to evaluate it."

    text = user_text.lower()
    mentioned = find_mentioned_candidates(user_text, candidates)

    if "compare" in text:
        targets = mentioned if mentioned else candidates
        if len(targets) < 2:
            return "I need at least two candidates to compare - upload another resume, or ask me to evaluate just one."
        ensure_evaluated(targets)
        evals = [st.session_state["evaluated_cache"][n] for n in targets]
        return "**Comparison:**\n\n" + render_comparison_table(evals)

    if "missing" in text and "skill" in text:
        targets = mentioned if mentioned else candidates
        ensure_evaluated(targets)
        return "**Missing skills:**\n\n" + "".join(
            render_missing_skills_only(st.session_state["evaluated_cache"][n]) for n in targets
        )

    if any(kw in text for kw in ["best candidate", "who should", "recommend", "rank"]):
        ensure_evaluated(candidates)
        evals = sorted(
            (st.session_state["evaluated_cache"][n] for n in candidates),
            key=lambda r: r.match_score, reverse=True,
        )
        best = evals[0]
        return (
            "**Ranking (best to worst match):**\n\n" + render_comparison_table(evals) +
            _clean_html(
                f'<div style="border-left:4px solid var(--hire); padding:10px 14px; margin-top:10px; background:#FFFFFF;">'
                f'<span class="col-label">Recommended</span>'
                f'<p style="font-family: Fraunces, serif; font-size:1.05rem; margin:4px 0;">{html.escape(best.candidate_name)}</p>'
                f'<p style="color: var(--ink-soft); font-size:0.88rem; margin:0;">{html.escape(best.justification)}</p></div>'
            )
        )

    if mentioned:
        ensure_evaluated(mentioned)
        return "".join(render_card(st.session_state["evaluated_cache"][n]) for n in mentioned)

    # Default: no specific candidate named, no known command matched ->
    # evaluate everyone uploaded so far against the current JD.
    ensure_evaluated(candidates)
    return "".join(render_card(st.session_state["evaluated_cache"][n]) for n in candidates)


# ── Chat transcript ──────────────────────────────────────────────────
if not st.session_state["messages"]:
    st.session_state["messages"].append({
        "role": "assistant",
        "content": (
            f"I'm ready to screen candidates for **{selected_role}**. "
            "Upload resumes in the sidebar, then ask me things like:\n\n"
            "- *\"Evaluate all resumes\"*\n"
            "- *\"Compare Ananya and Rahul\"*\n"
            "- *\"What skills is Meera missing?\"*\n"
            "- *\"Who's the best candidate?\"*"
        ),
    })

for message in st.session_state["messages"]:
    avatar = "🗂️" if message["role"] == "assistant" else None
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"], unsafe_allow_html=True)

user_input = st.chat_input("Ask about a candidate, e.g. 'Compare Ananya and Rahul'...")

if user_input:
    st.session_state["messages"].append({"role": "user", "content": html.escape(user_input)})
    with st.chat_message("user"):
        st.markdown(html.escape(user_input))

    with st.chat_message("assistant", avatar="🗂️"):
        with st.spinner("Reviewing the case file..."):
            reply = route_message(user_input)
        st.markdown(reply, unsafe_allow_html=True)

    st.session_state["messages"].append({"role": "assistant", "content": reply})
