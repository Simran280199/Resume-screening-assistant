"""
jd_library.py
──────────────
Manages a persistent library of Job Descriptions, organized by role.
Lets a recruiter pick a saved role (e.g. "Data Scientist") instead of
retyping the JD every single time, and lets them add new roles as needed.

WHY A JSON FILE:
This is read constantly but written rarely, by a single user, with a small
number of entries. A JSON file is simple, human-readable, and needs zero
setup - a real database would be over-engineering for this scale.
"""

import json
import os

JD_LIBRARY_PATH = "jd_library.json"

DEFAULT_JDS = {
    "Data Scientist": (
        "We are looking for a Data Scientist to build machine learning models, "
        "communicate insights to stakeholders, and help deploy models into "
        "production. Required: Python, SQL, scikit-learn, statistics, deep "
        "learning (TensorFlow/PyTorch), cloud deployment (AWS/GCP/Azure), "
        "data visualization (Tableau/Power BI), MLOps (Docker, MLflow)."
    ),
    "Software Engineer": (
        "We are looking for a Software Engineer to design, build, and maintain "
        "scalable backend services. Required: strong proficiency in at least "
        "one of Java/Python/Go, REST API design, microservices architecture, "
        "SQL databases, Git, CI/CD, containerization (Docker/Kubernetes), "
        "cloud platforms (AWS/GCP/Azure)."
    ),
    "Lawyer": (
        "We are looking for an Associate Lawyer to support litigation and "
        "contract review. Required: Juris Doctor (JD) degree, bar admission, "
        "2+ years of experience in corporate or litigation practice, strong "
        "legal research and writing skills, contract drafting experience, "
        "excellent client communication."
    ),
    "Doctor": (
        "We are looking for a Physician for our internal medicine department. "
        "Required: MBBS/MD degree, valid medical license, residency completion "
        "in internal medicine, 3+ years of clinical experience, strong "
        "diagnostic skills, patient communication, and familiarity with "
        "electronic health record (EHR) systems."
    ),
    "Judge": (
        "We are seeking a candidate for a Judicial Officer position. Required: "
        "law degree with bar admission, minimum 10 years of legal practice or "
        "prior judicial experience, strong track record of impartial legal "
        "reasoning, published judgments or legal opinions, excellent written "
        "and verbal communication."
    ),
}


def load_library() -> dict:
    """Loads the JD library from disk. If the file doesn't exist yet (first
    run), creates it pre-populated with DEFAULT_JDS."""
    if not os.path.exists(JD_LIBRARY_PATH):
        save_library(DEFAULT_JDS)
        return dict(DEFAULT_JDS)

    with open(JD_LIBRARY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_library(library: dict) -> None:
    """Writes the full library dict back to disk as JSON."""
    with open(JD_LIBRARY_PATH, "w", encoding="utf-8") as f:
        json.dump(library, f, indent=2, ensure_ascii=False)


def add_or_update_jd(role_name: str, jd_text: str) -> None:
    """Adds a new role or overwrites an existing one, then persists to disk."""
    library = load_library()
    library[role_name] = jd_text
    save_library(library)


def delete_jd(role_name: str) -> None:
    """Removes a role from the library, then persists to disk."""
    library = load_library()
    library.pop(role_name, None)
    save_library(library)


def list_roles() -> list:
    """Returns just the role names, e.g. for populating a dropdown."""
    return list(load_library().keys())


def get_jd(role_name: str) -> str:
    """Returns the JD text for one role."""
    return load_library().get(role_name, "")
