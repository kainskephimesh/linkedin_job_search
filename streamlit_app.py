"""
LinkedIn Job Scraper — Streamlit UI

A thin UI layer over mcp_server.py's scraping tools and mcp_client.py's LLM
filter-expansion helper. Runs the same Playwright + Azure OpenAI pipeline used
by the CLI (mcp_client.py), just driven by widgets instead of input() prompts.
"""
import asyncio
import json
import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import mcp_server
from mcp_client import expand_filters_with_llm
from resume_utils import extract_text_from_resume, parse_resume_with_llm, score_jobs_against_resume

st.set_page_config(page_title="LinkedIn Job Scraper", page_icon="🔎", layout="wide")

st.title("🔎 LinkedIn Job Scraper")
st.caption(
    "Scrapes the LinkedIn Jobs tab (last 24h) for roles matching your filters, "
    "with LLM-assisted role expansion, location correction, and field extraction."
)

if not os.path.exists(mcp_server.COOKIE_PATH):
    st.error(
        "No LinkedIn session found. Run `python manual_login.py` once from a terminal "
        "to save a session, then reload this page."
    )
    st.stop()

for key, default in [
    ("roles_input", "AI Engineer, Data Scientist"),
    ("locations_input", "Remote, India"),
    ("max_exp_years", 3),
    ("resume_text", None),
    ("resume_profile", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

with st.sidebar:
    st.header("📄 Search by resume")
    resume_file = st.file_uploader("Upload your resume", type=["pdf", "docx", "txt"])
    if resume_file is not None and st.button("Parse resume & autofill filters", use_container_width=True):
        with st.spinner("Reading resume and extracting your profile..."):
            try:
                resume_text = extract_text_from_resume(resume_file.getvalue(), resume_file.name)
                if len(resume_text.strip()) < 50:
                    st.error("Couldn't extract enough text from this file — try a different format.")
                else:
                    profile = asyncio.run(parse_resume_with_llm(resume_text))
                    st.session_state.resume_text = resume_text
                    st.session_state.resume_profile = profile
                    st.session_state.roles_input = ", ".join(profile.get("roles", []))
                    st.session_state.locations_input = ", ".join(profile.get("locations", []))
                    st.session_state.max_exp_years = int(profile.get("max_exp_years", 3))
                    st.rerun()
            except Exception as e:
                st.error(f"Resume parsing failed: {e}")

    if st.session_state.resume_profile:
        p = st.session_state.resume_profile
        st.success(p.get("summary", "Resume parsed."))
        st.caption("Skills: " + ", ".join(p.get("skills", [])))
        if st.button("Clear resume", use_container_width=True):
            st.session_state.resume_text = None
            st.session_state.resume_profile = None
            st.rerun()

    st.divider()
    st.header("Search filters")
    roles_input = st.text_input("Job title(s), comma-separated", key="roles_input")
    locations_input = st.text_input("Location(s), comma-separated", key="locations_input")
    max_exp_years = st.slider("Max years of experience", 0, 20, key="max_exp_years")
    work_type = st.selectbox("Workplace type", ["Any", "Remote", "Hybrid", "On-site"])
    search_mode = st.selectbox("Search Mode", ["jobs", "feed", "both"], help="jobs: Official Jobs tab. feed: User posts. both: Fallback to feed if jobs is empty.")
    max_posts = st.slider("Max results", 5, 100, 30, step=5)
    expand_with_llm = st.checkbox(
        "Expand role aliases & correct location typos with LLM", value=True
    )
    score_with_resume = st.checkbox(
        "Score & sort results by fit to my resume",
        value=st.session_state.resume_text is not None,
        disabled=st.session_state.resume_text is None,
        help="Upload and parse a resume above to enable this.",
    )
    show_browser = st.checkbox(
        "Show the browser window while scraping (unchecked = headless)", value=False
    )

    st.divider()
    if os.path.exists(mcp_server.SEEN_JOBS_PATH):
        with open(mcp_server.SEEN_JOBS_PATH, encoding="utf-8") as f:
            seen_count = len(json.load(f))
        st.caption(f"📦 {seen_count} previously-seen jobs cached (won't be re-shown).")
        if st.button("Clear seen-jobs cache"):
            os.remove(mcp_server.SEEN_JOBS_PATH)
            st.success("Cache cleared — next search can re-surface old jobs.")
            st.rerun()

    run_button = st.button("🚀 Search jobs", type="primary", use_container_width=True)

if "results" not in st.session_state:
    st.session_state.results = None
if "raw_message" not in st.session_state:
    st.session_state.raw_message = None


def run_search(roles, locations, max_exp_years, work_type, max_posts, expand, headless, resume_text, search_mode):
    async def _run():
        filters = {
            "roles": roles,
            "locations": locations,
            "max_exp_years": max_exp_years,
            "work_type": work_type,
            "search_mode": search_mode,
        }
        if expand:
            filters = await expand_filters_with_llm(filters)

        results = None
        raw = ""

        if search_mode in ["jobs", "both"]:
            raw = await mcp_server.scrape_linkedin_jobs_tab(
                roles=filters["roles"],
                locations=filters["locations"],
                max_exp_years=filters["max_exp_years"],
                work_type=filters["work_type"],
                max_posts=max_posts,
                headless=headless,
            )
            results = json.loads(raw) if raw.strip().startswith("[") else None

        if not results and search_mode in ["feed", "both"]:
            raw = await mcp_server.search_linkedin_job_posts(
                roles=filters["roles"],
                locations=filters["locations"],
                max_exp_years=filters["max_exp_years"],
                max_posts=max_posts,
                headless=headless,
            )
            results = json.loads(raw) if raw.strip().startswith("[") else None

        results = results or []
        if results and resume_text:
            results = await score_jobs_against_resume(resume_text, results)

        return filters, results, raw

    return asyncio.run(_run())


if run_button:
    spinner_msg = "Scraping LinkedIn, running LLM extraction"
    if score_with_resume:
        spinner_msg += ", and scoring against your resume"
    spinner_msg += "... this can take a couple of minutes."

    with st.spinner(spinner_msg):
        try:
            filters, results, raw_message = run_search(
                roles_input, locations_input, max_exp_years, work_type,
                max_posts, expand_with_llm, headless=not show_browser,
                resume_text=st.session_state.resume_text if score_with_resume else None,
                search_mode=search_mode,
            )
            st.session_state.results = results
            st.session_state.filters = filters
            st.session_state.raw_message = raw_message if not results else None
        except Exception as e:
            st.error(f"Search failed: {e}")

if st.session_state.get("filters"):
    f = st.session_state.filters
    st.info(
        f"**Roles searched:** {f['roles']}  \n"
        f"**Locations searched:** {f['locations']}  \n"
        f"**Max experience:** {f['max_exp_years']} yrs · **Workplace:** {f['work_type']} · **Mode:** {f.get('search_mode', 'jobs')}"
    )

if st.session_state.results:
    results = st.session_state.results
    has_fit_score = any("fit_score" in r for r in results)
    st.success(
        f"Found {len(results)} matching job post(s)"
        + (", sorted by fit to your resume." if has_fit_score else ".")
    )

    df = pd.DataFrame(results)
    display_cols = [
        c for c in (["fit_score"] if has_fit_score else [])
        + ["company", "role", "location", "experience", "emails", "phones", "posted"]
        if c in df.columns
    ]
    
    # Dynamically show/hide link columns based on search mode
    search_mode = st.session_state.filters.get("search_mode", "jobs")
    if "post_url" in df.columns: display_cols.append("post_url")
    
    if search_mode != "feed":
        if "apply_link" in df.columns: display_cols.append("apply_link")
        if "poster_link" in df.columns: display_cols.append("poster_link")
        
    st.dataframe(df[display_cols], use_container_width=True, hide_index=True)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download results as CSV",
        data=csv_bytes,
        file_name="linkedin_jobs.csv",
        mime="text/csv",
    )

    st.subheader("Detailed view")
    for row in results:
        score_prefix = f"[{row['fit_score']}%] " if "fit_score" in row else ""
        title = f"{score_prefix}{row.get('company', 'N/A')} — {row.get('role', 'N/A')}"
        with st.expander(title):
            if "fit_score" in row:
                st.markdown(f"**Fit score:** {row['fit_score']}/100 — {row.get('fit_reason', '')}")
            st.markdown(f"**Location:** {row.get('location', 'N/A')}")
            st.markdown(f"**Experience:** {row.get('experience', 'N/A')}")
            st.markdown(f"**Posted:** {row.get('posted', 'N/A')}")
            st.markdown(f"**Emails:** {row.get('emails', 'N/A')}")
            st.markdown(f"**Phones:** {row.get('phones', 'N/A')}")
            st.markdown(f"**WhatsApp:** {row.get('whatsapp', 'N/A')}")
            if row.get("post_url", "N/A") != "N/A":
                st.markdown(f"**Original Post:** {row['post_url']}")
            if row.get("apply_link", "N/A") != "N/A":
                st.markdown(f"**Apply:** {row['apply_link']}")
            if row.get("poster_link", "N/A") != "N/A":
                st.markdown(f"**Job Poster:** {row['poster_link']}")
            st.caption(row.get("snippet", ""))
elif st.session_state.raw_message:
    st.warning("No results found.")
    st.code(st.session_state.raw_message[:1000])
