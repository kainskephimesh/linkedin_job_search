

"""
LinkedIn Job Scraper — Azure OpenAI MCP Client
"""

import asyncio
import json
import logging
import os
import sys

from dotenv import load_dotenv
from openai import AzureOpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

LOG_DIR = "logs"
LOG_PATH = os.path.join(LOG_DIR, "mcp_client.log")
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("mcp-client")
logger.setLevel(logging.DEBUG)
logger.propagate = False
_file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
_file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(_file_handler)


def _log(msg: str):
    logger.info(msg)


AZURE_ENDPOINT   = os.environ["AZURE_OPENAI_ENDPOINT"]
AZURE_DEPLOYMENT = os.environ["AZURE_OPENAI_DEPLOYMENT"]
AZURE_API_KEY    = os.environ["AZURE_OPENAI_API_KEY"]
AZURE_API_VER    = os.environ["AZURE_OPENAI_API_VERSION"]

# ── LinkedIn credentials (set via env for safety) ──────────────────────────────


LI_EMAIL    = os.environ.get("LINKEDIN_EMAIL", "")
LI_PASSWORD = os.environ.get("LINKEDIN_PASSWORD", "")

az_client = AzureOpenAI(
    api_version=AZURE_API_VER,
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_API_KEY,
)

SYSTEM_PROMPT = """You are a LinkedIn job research assistant.
Your ONLY permitted actions are:
  1. Search the LinkedIn Jobs tab to find job posts matching the user's filters.
  2. As a fallback, search LinkedIn posts by keyword.
  3. Save results to a .txt file.

IMPORTANT TOOL ORDER:
  - ALWAYS call scrape_linkedin_jobs_tab FIRST, passing the user's roles/locations/max_exp_years.
  - Only call search_linkedin_job_posts if the Jobs tab search returns 0 results, passing the same filters.
  - Always finish by calling save_results_to_file.

You must NOT post, like, comment, send messages, or perform any write action on LinkedIn.
Always confirm completion with a summary table showing Company | Role | Email | Location."""


def prompt_for_filters() -> dict:
    print("\nEnter search filters (press Enter to use the default shown).\n")

    roles = input("Job title(s), comma-separated [AI Engineer, Data Scientist]: ").strip()
    roles = roles or "AI Engineer, Data Scientist"

    locations = input("Location(s), comma-separated [Remote, India]: ").strip()
    locations = locations or "Remote, India"

    max_exp_raw = input("Max years of experience [3]: ").strip()
    try:
        max_exp_years = int(max_exp_raw) if max_exp_raw else 3
    except ValueError:
        print("  Not a number, defaulting to 3.")
        max_exp_years = 3

    work_type = input("Workplace type (Any, Remote, Hybrid, On-site) [Any]: ").strip()
    work_type = work_type or "Any"

    return {"roles": roles, "locations": locations, "max_exp_years": max_exp_years, "work_type": work_type}


_EXPAND_FILTERS_SYSTEM = """\
You expand and correct job-search filters typed by a user for a LinkedIn job search.

Return ONLY a valid JSON object with exactly these fields:
{
  "roles": ["expanded list of job titles, max 8"],
  "locations": ["corrected/expanded list of locations, max 6"]
}

Rules:
- Roles: keep the user's original term(s) and add common real-world synonyms/aliases/adjacent
  titles actually used in job postings. Example: "devops" -> ["DevOps Engineer", "Site Reliability Engineer",
  "SRE", "Cloud Engineer", "Platform Engineer", "Infrastructure Engineer", "DevOps"].
  Do not add unrelated roles (e.g. don't expand "devops" into "Data Scientist").
- Locations: fix spelling/casing mistakes to the real city/region/country name (e.g. "banglore" -> "Bangalore",
  "puna" -> "Pune", "hydrabad" -> "Hyderabad"). Keep "Remote" as-is if present. Do not invent locations
  that weren't implied by the input.
"""


async def expand_filters_with_llm(filters: dict) -> dict:
    """Expand role aliases and correct location typos via the LLM. Falls back to the
    original filters unchanged if the call fails, so a bad/slow API call never blocks the run."""

    def _call() -> str:
        resp = az_client.chat.completions.create(
            model=AZURE_DEPLOYMENT,
            messages=[
                {"role": "system", "content": _EXPAND_FILTERS_SYSTEM},
                {"role": "user", "content": f"roles: {filters['roles']}\nlocations: {filters['locations']}"},
            ],
            response_format={"type": "json_object"},
            max_completion_tokens=400,
            temperature=0,
        )
        return resp.choices[0].message.content

    try:
        raw = await asyncio.to_thread(_call)
        expanded = json.loads(raw)
        roles = expanded.get("roles") or []
        locations = expanded.get("locations") or []
        if not roles or not locations:
            raise ValueError("LLM returned empty roles/locations")

        new_filters = dict(filters)
        new_filters["roles"] = ", ".join(roles)
        new_filters["locations"] = ", ".join(locations)

        print(f"🔎 Expanded roles: {new_filters['roles']}")
        print(f"📍 Corrected/expanded locations: {new_filters['locations']}")
        _log(f"Filter expansion — roles: {filters['roles']!r} -> {new_filters['roles']!r} | "
             f"locations: {filters['locations']!r} -> {new_filters['locations']!r}")
        return new_filters
    except Exception as e:
        print(f"  ⚠ Filter expansion failed ({e}), using filters as typed.")
        _log(f"Filter expansion failed: {e}")
        return filters


def mcp_tool_to_openai(tool) -> dict:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.inputSchema or {"type": "object", "properties": {}},
        },
    }


async def run_agent():
    print("=" * 60)
    print("  LinkedIn AI/ML Job Scraper — Azure OpenAI + MCP")
    print("=" * 60)
    _log("=" * 60)
    _log("Client starting.")

    filters = prompt_for_filters()
    _log(f"Filters chosen: {filters}")

    filters = await expand_filters_with_llm(filters)
    _log(f"Filters after LLM expansion: {filters}")

    server_params = StdioServerParameters(
        command=sys.executable,
        args=["mcp_server.py"],
        env=None,
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("✅ Connected to MCP server.\n")
            _log("Connected to MCP server.")

            tools_response = await session.list_tools()
            mcp_tools = tools_response.tools
            openai_tools = [mcp_tool_to_openai(t) for t in mcp_tools]

            print(f"📦 Tools available: {[t.name for t in mcp_tools]}\n")
            _log(f"Tools available: {[t.name for t in mcp_tools]}")

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Please start the job search by checking the LinkedIn Jobs tab first.\n"
                        "Call scrape_linkedin_jobs_tab with "
                        f"roles=\"{filters['roles']}\", locations=\"{filters['locations']}\", "
                        f"max_exp_years={filters['max_exp_years']}, work_type=\"{filters['work_type']}\".\n"
                        "If the jobs tab returns 0 results, fall back to search_linkedin_job_posts "
                        "with the same parameters.\n"
                        "Save all results to 'linkedin_jobs.txt'."
                    ),
                },
            ]

            allowed_tools = {
                "scrape_linkedin_jobs_tab",
                "search_linkedin_job_posts",
                "save_results_to_file",
            }

            max_iterations = 10
            iteration = 0

            while iteration < max_iterations:
                iteration += 1
                print(f"\n🔄 Agent iteration {iteration}...")
                _log(f"Agent iteration {iteration} starting.")

                # Force first call to use feed scroll
                if iteration == 1:
                    tool_choice = {
                        "type": "function",
                        "function": {"name": "scrape_linkedin_jobs_tab"}
                    }
                else:
                    tool_choice = "auto"

                response = az_client.chat.completions.create(
                    model=AZURE_DEPLOYMENT,
                    messages=messages,
                    tools=openai_tools,
                    tool_choice=tool_choice,
                    max_completion_tokens=4096,
                )

                msg = response.choices[0].message
                messages.append({
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": msg.tool_calls
                })

                if not msg.tool_calls:
                    print("\n✅ Agent finished.\n")
                    print("─" * 60)
                    print(msg.content)
                    _log(f"Agent finished at iteration {iteration}. Final message: {msg.content}")
                    break

                for tc in msg.tool_calls:
                    fn_name = tc.function.name
                    fn_args = json.loads(tc.function.arguments or "{}")

                    print(f"  🛠  Calling tool: {fn_name}({fn_args})")
                    _log(f"Calling tool: {fn_name}({fn_args})")

                    if fn_name not in allowed_tools:
                        tool_result = f"❌ Tool '{fn_name}' is not permitted."
                        _log(f"Blocked disallowed tool call: {fn_name}")
                    else:
                        try:
                            result = await session.call_tool(fn_name, fn_args)
                            tool_result = result.content[0].text if result.content else "No output."
                        except Exception as e:
                            tool_result = f"❌ Tool error: {e}"
                            _log(f"Tool '{fn_name}' raised an error: {e}")

                    preview = tool_result[:200].replace("\n", " ")
                    print(f"  📤 Result preview: {preview}...")
                    _log(f"Result from {fn_name}: {tool_result[:2000]}")

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tool_result,
                    })

            else:
                print("⚠  Max iterations reached.")
                _log("Max iterations reached without the agent finishing.")

    print("\n🎉 Done! Check 'linkedin_jobs.txt' for results.")
    print("📸 Check 'debug_feed.png' if 0 results — it shows what LinkedIn rendered.")
    _log("Client run finished.")


if __name__ == "__main__":
    asyncio.run(run_agent())