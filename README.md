# LinkedIn AI/ML Job Scraper — MCP + Azure OpenAI

An AI agent that scrapes LinkedIn's Jobs tab for roles matching filters you
choose (title, location, experience, workplace type), uses an LLM to expand
your search into relevant aliases, extract structured data from each posting,
and — optionally — score every result against your resume. Available as both
a CLI agent and a Streamlit web UI.

**Read-only by design**: the AI is restricted to a fixed allowlist of tools
and can never post, like, comment, message, or apply on your behalf.

---

## Features

- **Two ways to search LinkedIn**: the actual Jobs tab (`scrape_linkedin_jobs_tab`,
  primary) and post/content search (`search_linkedin_job_posts`, fallback if the
  Jobs tab returns nothing).
- **LLM-assisted filter expansion**: typing `devops` automatically searches
  `DevOps Engineer, Site Reliability Engineer, SRE, Cloud Engineer, ...` as well.
  Typos in locations (`banglore` → `Bangalore`) get corrected automatically.
- **LLM structured extraction**: each raw scraped post/listing is parsed by the
  LLM into company, role, emails, phones, WhatsApp, location, experience,
  posted-time, and apply link — instead of brittle regex/CSS guessing.
- **Resume-based search**: upload a resume (PDF/DOCX/TXT) and the LLM extracts
  your target roles, locations, experience level, and skills to auto-fill the
  search — then scores and ranks every result by fit to your resume.
- **Persistent dedup cache**: previously-seen jobs (`.li_session/seen_jobs.json`)
  are never re-shown across runs, so repeated searches only surface what's new.
- **Human-like scraping**: randomized scroll speed/pauses, mouse movement, and
  wheel-based scrolling (LinkedIn's feed scrolls a nested container, not the
  window) to reduce the chance of being flagged as a bot.
- **File-based logging**: every step (filters used, cards found, skip reasons,
  LLM calls) is logged to `logs/` for debugging zero-result runs.

---

## Architecture

```
                       ┌───────────────────────────┐
                       │   manual_login.py         │   (run once)
                       │   → .li_session/cookies.json
                       └───────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
┌───────────────┐          ┌────────────────┐          ┌─────────────────┐
│ mcp_client.py │  stdio   │ mcp_server.py  │          │ streamlit_app.py │
│ (CLI agent)   │◄────────►│ (MCP tools)    │◄─────────│ (Web UI)         │
└───────────────┘   MCP    └────────────────┘  direct  └─────────────────┘
        │                     │        │           call        │
        │                     │        │                       │
        ▼                     ▼        ▼                       ▼
  Azure OpenAI          Playwright   Azure OpenAI         resume_utils.py
  (tool-calling loop)   (Chromium)   (field extraction,    (resume parsing +
                                      filter expansion)     fit scoring)
```

- **`manual_login.py`** — one-time interactive login; saves LinkedIn session
  cookies to `.li_session/cookies.json` so nothing else needs your password.
- **`mcp_server.py`** — the MCP server. Exposes three read-only tools over
  stdio (see table below). Owns all Playwright browser automation and the
  LLM-based post/job field extraction.
- **`mcp_client.py`** — CLI entry point. Prompts you for filters, expands them
  via LLM (`expand_filters_with_llm`), then drives an Azure OpenAI
  tool-calling loop that decides which MCP tools to call and in what order.
- **`streamlit_app.py`** — web UI. Calls the same `mcp_server.py` tool
  functions directly (no MCP/agent loop needed for a single linear flow),
  adds resume upload, a results table, CSV export, and fit-score sorting.
- **`resume_utils.py`** — resume text extraction (PDF/DOCX/TXT) and two LLM
  calls: profile extraction (roles/locations/experience/skills) and per-job
  fit scoring (parallelized across all results).

### MCP Server Tools (READ-ONLY)

| Tool | Description |
|------|-------------|
| `scrape_linkedin_jobs_tab` | Search LinkedIn's Jobs tab directly (primary source — structured listings) |
| `search_linkedin_job_posts` | Fallback: search LinkedIn post/content search for hiring posts |
| `save_results_to_file` | Write a formatted `.txt` report (summary table + detailed view) |

All three accept `roles` / `locations` as comma-separated strings and
`max_exp_years` as an integer. Any tool call outside this list is blocked by
the client's allowlist, even if the model tries.

---

## Setup

### 1. Install dependencies

This project uses a `uv`-managed virtual environment (`.venv`):

```bash
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
playwright install chromium
```

(If you're not using `uv`, a plain `pip install -r requirements.txt` in your
own virtualenv works too.)

### 2. Configure secrets

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

```
AZURE_OPENAI_API_KEY=your_azure_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource.cognitiveservices.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-5.4
AZURE_OPENAI_API_VERSION=2024-12-01-preview

LINKEDIN_EMAIL=your_linkedin_email@example.com
LINKEDIN_PASSWORD=your_linkedin_password
```

`.env` is gitignored — never commit it. Both `mcp_server.py` and
`mcp_client.py` load it automatically via `python-dotenv`.

### 3. Save a LinkedIn session

Run once, log in manually in the browser window that opens (handles
CAPTCHA/OTP), then press Enter in the terminal once you're on your feed:

```bash
python manual_login.py
```

This saves cookies to `.li_session/cookies.json`, which every other script
reuses — no further logins needed until the session expires.

---

## Usage

### Option A — CLI agent

```bash
python mcp_client.py
```

You'll be prompted for job title(s), location(s), max experience, and
workplace type. The agent expands your filters via LLM, searches the Jobs
tab (falling back to post search if empty), and writes results to
`linkedin_jobs.txt`.

### Option B — Streamlit UI

```bash
python -m streamlit run streamlit_app.py
```

- Enter filters manually in the sidebar, **or** upload a resume and click
  "Parse resume & autofill filters" to auto-populate them from your profile.
- Optionally check "Score & sort results by fit to my resume" to rank results
  by an LLM-computed fit score (0–100) instead of scrape order.
- Results appear as a sortable table with a CSV download button, plus an
  expandable detailed view per job (emails, phones, WhatsApp, apply link).

---

## Output

### `linkedin_jobs.txt` (CLI mode)

```
===============================================================================================
LinkedIn AI/ML Job Posts — Last 24 Hours
Generated  : 2026-07-03 14:14:06
Total      : 12 posts found
===============================================================================================

Company                        | Role                         | Emails                              | Location             | Experience
---------------------------------------------------------------------------------------------------------------------------------------
Tata Consultancy Services      | AWS DevOps                   | himaja.madala@tcs.com               | Mumbai - Olympus     | 4 - 6 years
...

===============================================================================================
DETAILED VIEW
===============================================================================================

[1] Tata Consultancy Services — AWS DevOps
    Emails     : himaja.madala@tcs.com
    Phones     : N/A
    WhatsApp   : N/A
    Location   : Mumbai - Olympus
    Experience : 4 - 6 years
    Posted     : 2 hours ago
    Post URL   : https://www.linkedin.com/jobs/view/4436487422/
    Apply Link : Easy Apply on LinkedIn: https://www.linkedin.com/jobs/view/4436487422/
    Snippet    : ...
```

### Logs

Every run writes detailed step-by-step logs to `logs/mcp_server.log` and
`logs/mcp_client.log` — filters used, cards found per selector strategy, why
each card was skipped (short/duplicate/role mismatch/location mismatch), and
LLM extraction results. Check these first if a run returns 0 results.

### Debug artifacts

`debug_feed.png` / `debug_feed.html` are saved on each feed scrape so you can
see exactly what LinkedIn rendered if something looks wrong.

---

## Project files

| File | Purpose |
|------|---------|
| `mcp_server.py` | MCP server: Playwright scraping tools + LLM extraction |
| `mcp_client.py` | CLI agent: filter prompts, LLM filter expansion, tool-calling loop |
| `streamlit_app.py` | Web UI over the same tools, with resume upload |
| `resume_utils.py` | Resume parsing + LLM job-fit scoring |
| `manual_login.py` | One-time interactive LinkedIn login → saves session cookies |
| `.env` / `.env.example` | Secrets (gitignored) / template |
| `requirements.txt` | Python dependencies |
| `.li_session/` | Saved cookies + seen-jobs dedup cache (gitignored) |
| `logs/` | Per-run debug logs (gitignored) |

---

## Notes & limitations

- **READ-ONLY by design**: `mcp_client.py`'s `allowed_tools` set blocks any
  tool call outside the three listed above, regardless of what the LLM
  attempts. This project only reads/reports — it never posts, applies, or
  messages on LinkedIn.
- LinkedIn's markup changes frequently (hashed/obfuscated CSS classes). If
  scraping suddenly returns 0 results, check `logs/mcp_server.log` first —
  it logs exactly which selector strategy matched (or didn't).
  Text-based extraction (author/timestamp parsing) is used instead of CSS
  selectors specifically to be more resilient to this.
  Selector fixes may still be needed after a LinkedIn redesign.
  LinkedIn may also show a CAPTCHA on first login from a new IP — re-run
  `manual_login.py` with a visible browser to solve it.
- Respect LinkedIn's Terms of Service — use this for personal job research
  only. Avoid running excessively frequently or with high `max_posts`/scroll
  counts, which increases the chance of triggering anti-automation defenses.
