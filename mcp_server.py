"""
LinkedIn Job MCP Server — No login tool, uses pre-saved cookies from manual_login.py
All debug output goes to stderr and logs/mcp_server.log (stdout is reserved for MCP JSON-RPC).
"""

import asyncio
import json
import logging
import re
import os
import sys
import random
from datetime import datetime
from dotenv import load_dotenv
from openai import AzureOpenAI
from mcp.server.fastmcp import FastMCP
from playwright.async_api import async_playwright, Page

load_dotenv()

mcp = FastMCP("linkedin-job-scraper")

# ── Azure OpenAI client (for LLM-powered post parsing) ────────────────────────
AZURE_ENDPOINT   = os.environ["AZURE_OPENAI_ENDPOINT"]
AZURE_DEPLOYMENT = os.environ["AZURE_OPENAI_DEPLOYMENT"]
AZURE_API_KEY    = os.environ["AZURE_OPENAI_API_KEY"]
AZURE_API_VER    = os.environ["AZURE_OPENAI_API_VERSION"]

_az_client = AzureOpenAI(
    api_version=AZURE_API_VER,
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_API_KEY,
)

COOKIE_PATH = ".li_session/cookies.json"
SEEN_JOBS_PATH = ".li_session/seen_jobs.json"
LOG_DIR = "logs"
LOG_PATH = os.path.join(LOG_DIR, "mcp_server.log")

os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("linkedin-job-scraper")
logger.setLevel(logging.DEBUG)
logger.propagate = False

_file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
_file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(_file_handler)

_stderr_handler = logging.StreamHandler(sys.stderr)
_stderr_handler.setFormatter(logging.Formatter("[SERVER] %(message)s"))
logger.addHandler(_stderr_handler)

DEFAULT_ROLES = "AI Engineer, Gen AI Engineer, Data Scientist, Machine Learning Engineer"
DEFAULT_LOCATIONS = "Jaipur, Remote, India"
DEFAULT_MAX_EXP_YEARS = 3

GENERIC_HIRING_KEYWORDS = [
    "we are hiring", "we're hiring", "looking for", "job opening",
    "opportunity", "join our team", "immediate joiner", "hiring",
]


def _log(msg: str):
    logger.info(msg)


def _parse_list(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def _matches_role(text: str, roles: list[str]) -> bool:
    keywords = [r.lower() for r in roles]
    return any(k in text.lower() for k in keywords)


def _matches_location(text: str, locations: list[str]) -> bool:
    return any(loc.lower() in text.lower() for loc in locations)


def _matches_experience(text: str, max_exp_years: int) -> bool:
    t = text.lower()
    if any(w in t for w in ["fresher", "0-3", "0 to 3", "entry level",
                             "junior", "graduate", "intern", "trainee"]):
        return True
    for lo, hi in re.findall(r"(\d+)\s*(?:to|-)\s*(\d+)\s*year", t):
        if int(lo) <= max_exp_years:
            return True
    for y in re.findall(r"(\d+)\s*year", t):
        if int(y) <= max_exp_years:
            return True
    return False


def _extract_email(text: str) -> str:
    emails = re.findall(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", text)
    return emails[0] if emails else "N/A"


def _extract_location(text: str, locations: list[str]) -> str:
    for loc in locations:
        if loc.lower() in text.lower():
            return loc
    return locations[0] if locations else "N/A"


_JOB_ID_RE = re.compile(r"/jobs/view/(\d+)")


def _dedup_keys(full_text: str, post_url: str) -> list[str]:
    """Return every key a post could be deduplicated under. A post is considered
    a duplicate if it matches on EITHER key — this catches the same job showing up
    via different search queries even when only one of the two signals is available."""
    keys = []
    match = _JOB_ID_RE.search(post_url or "")
    if match:
        keys.append(f"job:{match.group(1)}")
    normalized = re.sub(r"\s+", " ", full_text[:150]).strip().lower()
    keys.append(f"text:{normalized}")
    return keys


def _load_seen_jobs() -> set:
    """Jobs already surfaced in a previous run — persisted across runs so the
    same posting doesn't get re-scraped, re-classified by the LLM, and re-shown."""
    try:
        with open(SEEN_JOBS_PATH, encoding="utf-8") as f:
            return set(json.load(f))
    except (FileNotFoundError, json.JSONDecodeError):
        return set()


def _save_seen_jobs(seen: set):
    os.makedirs(os.path.dirname(SEEN_JOBS_PATH), exist_ok=True)
    with open(SEEN_JOBS_PATH, "w", encoding="utf-8") as f:
        json.dump(sorted(seen), f)


# LinkedIn ships obfuscated/hashed CSS class names that change frequently, so
# author/timestamp are parsed from the card's visible text lines instead of
# CSS selectors — this survives markup changes that break class-based selectors.
_BOILERPLATE_LINE_RE = re.compile(
    r"^(feed post|suggested|promoted|sponsored|ad)$"
    r"|(likes this|reposted this|commented on this|celebrates this|"
    r"supports this|loves this|finds this (interesting|funny))$",
    re.IGNORECASE,
)
_FOLLOWER_COUNT_RE = re.compile(r"^[\d,]+\s+followers?$", re.IGNORECASE)
_TIME_LINE_RE = re.compile(
    r"^(\d+)\s*(h|hr|hrs|d|day|days|w|wk|wks|mo|mos|y|yr|yrs)\b", re.IGNORECASE
)


def _extract_author_from_text(full_text: str) -> str:
    for line in full_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if _BOILERPLATE_LINE_RE.search(line) or _FOLLOWER_COUNT_RE.match(line):
            continue
        return line
    return "Unknown"


def _extract_time_from_text(full_text: str) -> str:
    for line in full_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        match = _TIME_LINE_RE.match(line)
        if match:
            return f"{match.group(1)}{match.group(2)}"
        if line.lower() in ("just now", "now"):
            return line
    return "recent"


async def _human_scroll(page: Page, scroll_pause_min: float = 1.5, scroll_pause_max: float = 3.5):
    """Scroll using mouse.wheel() — the only reliable method on LinkedIn.

    LinkedIn's feed renders inside a nested scrollable div, NOT the window.
    window.scrollBy() and keyboard keys have no effect unless the exact container
    element has focus, which is unreliable. page.mouse.wheel() dispatches a native
    WheelEvent to whatever element sits under the cursor, bypassing focus entirely.
    """
    scroll_amount = random.randint(400, 900)

    # Position the cursor in the centre of the viewport so the wheel
    # event lands on the main feed container (not a sidebar widget)
    viewport = page.viewport_size or {"width": 1280, "height": 800}
    cx = (viewport["width"] // 2) + random.randint(-60, 60)
    cy = (viewport["height"] // 2) + random.randint(-60, 60)
    await page.mouse.move(cx, cy)

    # Primary: dispatch wheel event directly to element under cursor
    await page.mouse.wheel(0, scroll_amount)
    await asyncio.sleep(0.3)

    # Secondary belt-and-braces: also nudge window in case layout scrolls both
    await page.evaluate(f"window.scrollBy({{top: {scroll_amount // 2}, behavior: 'smooth'}})")

    pause = random.uniform(scroll_pause_min, scroll_pause_max)
    _log(f"    Wheel scroll {scroll_amount}px at ({cx},{cy}), pausing {pause:.1f}s")
    await asyncio.sleep(pause)

    # Occasionally scroll back up a little (like re-reading)
    if random.random() < 0.2:
        back_scroll = random.randint(60, 200)
        _log(f"    Scrolling back up {back_scroll}px (re-reading simulation)")
        await page.mouse.wheel(0, -back_scroll)
        await asyncio.sleep(random.uniform(0.5, 1.2))
        await page.mouse.wheel(0, back_scroll + random.randint(20, 80))
        await asyncio.sleep(random.uniform(0.4, 0.9))


async def _move_mouse_randomly(page: Page):
    """Move mouse to a random position to simulate human presence."""
    x = random.randint(200, 1000)
    y = random.randint(200, 600)
    await page.mouse.move(x, y, steps=random.randint(5, 15))


async def _expand_see_more(page: Page):
    """Click 'see more' buttons to expand truncated posts."""
    try:
        see_more_buttons = await page.query_selector_all(
            "button.feed-shared-inline-show-more-text__see-more-less-toggle, "
            "button[aria-label*='see more'], "
            "span.see-more"
        )
        if see_more_buttons:
            _log(f"    Expanding {min(len(see_more_buttons), 5)} 'see more' button(s)")
        for btn in see_more_buttons[:5]:
            try:
                await btn.click()
                await asyncio.sleep(random.uniform(0.3, 0.8))
            except Exception:
                pass
    except Exception:
        pass


async def _get_cards(page: Page):
    """Try multiple selector strategies to find post cards.

    LinkedIn now ships hashed/obfuscated CSS class names (e.g. "_1e09f84a")
    that change on every deploy, so class-based selectors below are legacy
    fallbacks. The reliable current markup exposes each feed/search post as
    a `div[role="listitem"]` — verified against both the home feed and
    content search pages.
    """
    # Strategy 0: role=listitem (current LinkedIn markup, as of 2026)
    cards = await page.query_selector_all("div[role='listitem']")
    if cards:
        _log(f"  Selector strategy 0 (role=listitem): {len(cards)} cards")
        return cards

    # Strategy 1: data-urn with activity
    cards = await page.query_selector_all("div[data-urn*='activity']")
    if cards:
        _log(f"  Selector strategy 1 (data-urn activity): {len(cards)} cards")
        return cards

    # Strategy 2: data-id with urn:li:activity
    cards = await page.query_selector_all("div[data-id*='urn:li:activity']")
    if cards:
        _log(f"  Selector strategy 2 (data-id urn): {len(cards)} cards")
        return cards

    # Strategy 3: feed-shared-update-v2
    cards = await page.query_selector_all("div.feed-shared-update-v2")
    if cards:
        _log(f"  Selector strategy 3 (feed-shared-update-v2): {len(cards)} cards")
        return cards

    # Strategy 4: occludable-update
    cards = await page.query_selector_all("div.occludable-update")
    if cards:
        _log(f"  Selector strategy 4 (occludable-update): {len(cards)} cards")
        return cards

    # Strategy 5: fie-impression-container
    cards = await page.query_selector_all("div.fie-impression-container")
    if cards:
        _log(f"  Selector strategy 5 (fie-impression-container): {len(cards)} cards")
        return cards

    # Strategy 6: artdeco list items
    cards = await page.query_selector_all("li.artdeco-list__item")
    if cards:
        _log(f"  Selector strategy 6 (artdeco-list__item): {len(cards)} cards")
        return cards

    # Strategy 7: broad fallback — any large text block in feed
    cards = await page.query_selector_all(
        "div[class*='update-components-text'], "
        "div[class*='feed-shared-text']"
    )
    if cards:
        _log(f"  Selector strategy 7 (update-components-text): {len(cards)} cards")
        return cards

    _log("  No cards found with any selector strategy.")
    return []


_LLM_EXTRACT_SYSTEM = """\
You are a LinkedIn post data extractor. Given raw text scraped from a LinkedIn post, \
extract job-related information and return ONLY a valid JSON object with these exact fields:
{
  "is_job_post": <true if this is a hiring/recruitment/job-opening post, false for opinion/news/congratulatory/repost posts>,
  "company": "<hiring company name, or poster name if company not mentioned>",
  "role": "<exact job role or title being hired for>",
  "emails": ["<all email addresses found in the post>"],
  "phones": ["<all phone numbers found, as-is including country code>"],
  "whatsapp": "<wa.me link if present, else null>",
  "location": "<job location e.g. Jaipur / Remote / Hybrid / India>",
  "experience": "<experience requirement e.g. 0-2 years / fresher / 3+ years, else null>",
  "posted": "<relative posting time if visible in text e.g. 3h, 1d, else null>",
  "apply_link": "<direct job application URL if present, else null>"
}
Rules: extract ALL emails and phones (not just the first). Use null for missing string fields and [] for empty arrays.\
"""


async def _llm_extract_fields(full_text: str) -> dict:
    """Use Azure OpenAI to extract structured job data from raw LinkedIn post text.
    Falls back gracefully — caller should catch exceptions.
    """

    def _call() -> str:
        resp = _az_client.chat.completions.create(
            model=AZURE_DEPLOYMENT,
            messages=[
                {"role": "system", "content": _LLM_EXTRACT_SYSTEM},
                {"role": "user", "content": full_text[:3000]},
            ],
            response_format={"type": "json_object"},
            max_completion_tokens=512,
            temperature=0,
        )
        return resp.choices[0].message.content

    raw = await asyncio.to_thread(_call)
    return json.loads(raw)


@mcp.tool()
async def scrape_linkedin_jobs_tab(
    roles: str = DEFAULT_ROLES,
    locations: str = DEFAULT_LOCATIONS,
    max_exp_years: int = DEFAULT_MAX_EXP_YEARS,
    work_type: str = "Any",
    max_posts: int = 50,
    headless: bool = False,
) -> str:
    """
    Search the LinkedIn Jobs tab directly and collect job posts matching
    the given filters. `roles` and `locations` are comma-separated lists
    (e.g. "AI Engineer, Data Scientist" / "Remote, Jaipur").
    Requires cookies saved by manual_login.py. READ-ONLY.
    """
    role_list = _parse_list(roles)
    location_list = _parse_list(locations)
    _log(f"Filters — roles: {role_list} | locations: {location_list} | max_exp_years: {max_exp_years} | work_type: {work_type}")

    if not os.path.exists(COOKIE_PATH):
        return (
            "❌ No session found. Run manual_login.py first:\n"
            "   python manual_login.py\n"
        )

    with open(COOKIE_PATH) as f:
        cookies = json.load(f)
    _log(f"Loaded {len(cookies)} cookies from {COOKIE_PATH}")

    _log("Launching Chromium...")
    pw = await async_playwright().start()
    browser = await pw.chromium.launch(
        headless=headless,
        slow_mo=50,
        args=["--no-sandbox", "--disable-blink-features=AutomationControlled"],
    )
    context = await browser.new_context(
        user_agent=(
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        viewport={"width": 1280, "height": 800},
        locale="en-US",
        timezone_id="Asia/Kolkata",
    )
    await context.add_cookies(cookies)
    _log("Browser context created and cookies applied.")
    page = await context.new_page()

    raw_matches = []  # [{"full_text": ..., "post_url": ...}]
    seen = _load_seen_jobs()
    previously_seen_count = len(seen)
    _log(f"Loaded {previously_seen_count} previously-seen job keys from {SEEN_JOBS_PATH}")

    LINKEDIN_JOBS_URL = (
        "https://www.linkedin.com/jobs/search/?keywords={query}&location={location}&f_TPR=r86400"
    )

    stats = {
        "cards_seen": 0, "skipped_short": 0, "skipped_duplicate": 0,
        "skipped_role": 0, "skipped_location_exp": 0,
        "skipped_llm": 0, "matched": 0,
    }

    try:
        # ── Setup Deep URL Filters ─────────────────────────────────────────────
        f_e_filter = ""
        if max_exp_years <= 1:
            f_e_filter = "1"
        elif max_exp_years <= 3:
            f_e_filter = "1%2C2%2C3" # Intern, Entry, Associate
        elif max_exp_years <= 5:
            f_e_filter = "1%2C2%2C3%2C4" # Up to Mid-Senior
        else:
            f_e_filter = "1%2C2%2C3%2C4%2C5%2C6"

        f_wt_filter = ""
        wt_lower = work_type.lower()
        if "remote" in wt_lower:
            f_wt_filter = "2"
        elif "hybrid" in wt_lower:
            f_wt_filter = "3"
        elif "site" in wt_lower:
            f_wt_filter = "1"

        # ── PHASE 1: Browse Jobs tab and collect raw matching posts ────────────────
        _log("Phase 1 — Browsing Jobs tab and collecting raw matches (no LLM yet)...")

        for role in role_list:
            for location in location_list:
                if len(raw_matches) >= max_posts:
                    break

                url = LINKEDIN_JOBS_URL.format(
                    query=role.replace(" ", "%20"),
                    location=location.replace(" ", "%20")
                )
                if f_e_filter:
                    url += f"&f_E={f_e_filter}"
                if f_wt_filter:
                    url += f"&f_WT={f_wt_filter}"

                _log(f"Searching Jobs for: {role} in {location}")

                await page.goto(url, wait_until="domcontentloaded", timeout=40_000)
                await page.bring_to_front()
                await asyncio.sleep(random.uniform(1.5, 3.0))

                # Force the search to execute if LinkedIn only filled the text fields
                try:
                    search_input = await page.query_selector("input.jobs-search-box__keyboard-text-input, input[aria-label='Search by title, skill, or company'], input.jobs-search-global-typeahead__input")
                    if search_input:
                        await search_input.click()
                        await page.keyboard.press("Enter")
                        _log("Pressed Enter in search box to force search execution.")
                        await asyncio.sleep(random.uniform(2, 3))
                except Exception as e:
                    _log(f"Could not press Enter in search box: {e}")

                # Wait for the job list to load
                try:
                    await page.wait_for_selector(
                        ".jobs-search-results-list, .scaffold-layout__list",
                        timeout=12_000,
                    )
                    _log("Jobs list detected.")
                except Exception:
                    _log("Jobs list selector timed out — continuing anyway.")

                await asyncio.sleep(random.uniform(2, 4))

                current_url = page.url
                if any(x in current_url for x in ["login", "authwall", "checkpoint"]):
                    await context.close()
                    await browser.close()
                    await pw.stop()
                    return "❌ LinkedIn session expired. Re-run: python manual_login.py"

                # Scroll the left panel a few times to load job cards
                for _ in range(4):
                    viewport = page.viewport_size or {"width": 1280, "height": 800}
                    await page.mouse.move(viewport["width"] // 4, viewport["height"] // 2)
                    await page.mouse.wheel(0, 800)
                    await asyncio.sleep(random.uniform(1.5, 3.0))

                # Extract job cards
                cards = await page.query_selector_all("li.jobs-search-results__list-item, div.job-card-container")
                _log(f"Job cards found: {len(cards)}")

                for card in cards:
                    if len(raw_matches) >= max_posts:
                        break
                    try:
                        stats["cards_seen"] += 1
                        
                        full_text = ""
                        try:
                            # Try to intercept the network response for the job details.
                            # LinkedIn frequently has adjacent cards' details already
                            # prefetched/cached, so this response never arrives — keep the
                            # timeout short (most real responses land in a few hundred ms)
                            # so a miss doesn't stall every single card by 2 full seconds.
                            async with page.expect_response(
                                lambda r: ("voyager/api/graphql" in r.url or "voyager/api/jobs/jobPostings" in r.url) and r.request.method != "OPTIONS" and r.status == 200,
                                timeout=600
                            ) as response_info:
                                await card.scroll_into_view_if_needed()
                                await asyncio.sleep(random.uniform(0.1, 0.2))
                                await card.click()

                            resp = await response_info.value
                            resp_text = await resp.text()
                            
                            # Parse JSON and find the longest string, which is typically the job description
                            try:
                                data = json.loads(resp_text)
                                def _find_longest_string(obj):
                                    longest = ""
                                    if isinstance(obj, dict):
                                        for k, v in obj.items():
                                            if isinstance(v, str) and k in ["text", "description"] and len(v) > len(longest):
                                                longest = v
                                            res = _find_longest_string(v)
                                            if len(res) > len(longest):
                                                longest = res
                                    elif isinstance(obj, list):
                                        for item in obj:
                                            res = _find_longest_string(item)
                                            if len(res) > len(longest):
                                                longest = res
                                    elif isinstance(obj, str):
                                        if len(obj) > len(longest):
                                            longest = obj
                                    return longest
                                
                                longest_str = _find_longest_string(data)
                                if len(longest_str) > 200:
                                    full_text = longest_str
                                    _log("  ⚡ Intercepted job details from network API!")
                            except Exception:
                                pass
                                
                        except Exception as intercept_err:
                            pass # Fallback below

                        if not full_text or len(full_text) < 100:
                            # The card was already clicked above (as part of the intercept
                            # attempt) — no need to click again, just give the right pane
                            # a moment to render and read it directly.
                            _log("  ⚠ Network intercept missed (likely cached), reading details pane directly.")
                            await asyncio.sleep(random.uniform(1.0, 1.8))

                            # Extract structured data from the right pane
                            details_pane = await page.query_selector(".jobs-search__job-details--container, .jobs-description__container")
                            if details_pane:
                                full_text = (await details_pane.inner_text()).strip()

                        if not full_text or len(full_text) < 40:
                            stats["skipped_short"] += 1
                            continue

                        # Extract URL from the title link
                        post_url = page.url
                        link_el = await card.query_selector("a.job-card-list__title, a.job-card-container__link")
                        if link_el:
                            href = await link_el.get_attribute("href")
                            if href:
                                post_url = ("https://www.linkedin.com" + href if href.startswith("/") else href).split("?")[0]

                        dedup_keys = _dedup_keys(full_text, post_url)
                        if any(k in seen for k in dedup_keys):
                            stats["skipped_duplicate"] += 1
                            continue
                        seen.update(dedup_keys)

                        if not _matches_role(full_text, role_list):
                            stats["skipped_role"] += 1
                            if stats["skipped_role"] <= 3:
                                _log(f"  ⏭ role mismatch, snippet: {full_text[:100].replace(chr(10), ' ')!r}")
                            continue
                        if not (_matches_location(full_text, location_list) or _matches_experience(full_text, max_exp_years)):
                            stats["skipped_location_exp"] += 1
                            if stats["skipped_location_exp"] <= 3:
                                _log(f"  ⏭ location/exp mismatch, snippet: {full_text[:100].replace(chr(10), ' ')!r}")
                            continue

                        # Try to extract the direct Apply Link from the UI button
                        apply_link = post_url
                        try:
                            apply_btn = await page.query_selector(".jobs-apply-button")
                            if apply_btn:
                                tag_name = await apply_btn.evaluate("el => el.tagName.toLowerCase()")
                                if tag_name == 'a':
                                    href = await apply_btn.get_attribute("href")
                                    if href:
                                        apply_link = ("https://www.linkedin.com" + href if href.startswith("/") else href).split("?")[0]
                                else:
                                    btn_text = (await apply_btn.inner_text()).lower()
                                    if "easy apply" in btn_text:
                                        apply_link = "Easy Apply on LinkedIn: " + post_url
                        except Exception:
                            pass

                        # Try to extract the Job Poster (Hiring Team) profile link
                        poster_link = "N/A"
                        try:
                            poster_el = await page.query_selector(".hirer-profile-card a[href*='/in/'], .jobs-poster a[href*='/in/'], .jobs-advocate-profile a[href*='/in/']")
                            if not poster_el:
                                poster_el = await page.query_selector(".jobs-search__job-details--container a[href*='/in/']")
                            
                            if poster_el:
                                href = await poster_el.get_attribute("href")
                                if href:
                                    poster_link = ("https://www.linkedin.com" + href if href.startswith("/") else href).split("?")[0]
                        except Exception:
                            pass

                        raw_matches.append({
                            "full_text": full_text,
                            "post_url": post_url,
                            "apply_link": apply_link,
                            "poster_link": poster_link
                        })
                        _log(f"  📌 Collected Job: {full_text[:80].replace(chr(10), ' ')!r}")

                    except Exception as ex:
                        _log(f"  Job card error: {ex}")
                        continue

        _log(f"Phase 1 complete — raw matches: {len(raw_matches)}, closing browser.")
        await context.close()
        await browser.close()
        await pw.stop()
        _save_seen_jobs(seen)
        _log(f"Saved {len(seen)} seen-job keys ({len(seen) - previously_seen_count} new) to {SEEN_JOBS_PATH}")

        # ── PHASE 2: LLM extraction (browser is closed, no blocking visible) ──
        _log("Phase 2 — Running LLM extraction on collected jobs...")
        results = []

        for raw in raw_matches:
            full_text = raw["full_text"]
            post_url  = raw["post_url"]
            ui_apply_link = raw.get("apply_link", post_url)
            poster_link = raw.get("poster_link", "N/A")
            try:
                extracted = await _llm_extract_fields(full_text)
                if not extracted.get("is_job_post", True):
                    stats["skipped_llm"] += 1
                    _log("  ⏭ LLM classified as non-job post, skipping.")
                    continue
                emails_list = extracted.get("emails") or []
                phones_list = extracted.get("phones") or []
                company     = extracted.get("company") or _extract_author_from_text(full_text)
                role_found  = extracted.get("role") or "AI/ML Role"
                location    = extracted.get("location") or _extract_location(full_text, location_list)
                whatsapp    = extracted.get("whatsapp") or "N/A"
                experience  = extracted.get("experience") or "N/A"
                apply_link  = extracted.get("apply_link") or ui_apply_link
                posted      = extracted.get("posted") or _extract_time_from_text(full_text)
                _log(
                    f"  🤖 LLM: {company} | {role_found} | "
                    f"emails={emails_list} | phones={phones_list}"
                )
            except Exception as llm_err:
                _log(f"  ⚠ LLM extraction failed ({llm_err}), falling back to regex.")
                company     = _extract_author_from_text(full_text)
                role_found  = "AI/ML Role"
                for role in role_list:
                    if role.lower() in full_text.lower():
                        role_found = role
                        break
                emails_list = []
                phones_list = []
                location    = _extract_location(full_text, location_list)
                whatsapp    = "N/A"
                experience  = "N/A"
                apply_link  = ui_apply_link
                posted      = _extract_time_from_text(full_text)

            stats["matched"] += 1
            results.append({
                "company":    company,
                "role":       role_found,
                "emails":     ", ".join(emails_list) if emails_list else _extract_email(full_text),
                "phones":     ", ".join(phones_list) if phones_list else "N/A",
                "whatsapp":   whatsapp,
                "location":   location,
                "experience": experience,
                "posted":     posted,
                "post_url":   post_url,
                "apply_link": apply_link,
                "poster_link": poster_link,
                "snippet":    full_text[:400].replace("\n", " ").strip(),
            })
            _log(f"  ✅ Matched: {company} — {role_found}")

        _log(
            f"Jobs summary — cards seen: {stats['cards_seen']}, "
            f"skipped (short): {stats['skipped_short']}, "
            f"skipped (duplicate): {stats['skipped_duplicate']}, "
            f"skipped (role mismatch): {stats['skipped_role']}, "
            f"skipped (location/exp mismatch): {stats['skipped_location_exp']}, "
            f"skipped (LLM non-job): {stats['skipped_llm']}, "
            f"matched: {stats['matched']}"
        )
        if stats["cards_seen"] == 0:
            _log("  ⚠ No jobs were found at all — LinkedIn's markup may have changed or the jobs didn't load.")
        elif stats["matched"] == 0:
            _log("  ⚠ Jobs were found but none matched your filters — try broader roles/locations.")

        _log(f"Total results from jobs tab: {len(results)}")
        return json.dumps(results, indent=2, ensure_ascii=False)

    except Exception as e:
        try:
            await context.close()
            await browser.close()
            await pw.stop()
        except Exception:
            pass
        return f"❌ Jobs tab scrape error: {e}"


@mcp.tool()
async def search_linkedin_job_posts(
    roles: str = DEFAULT_ROLES,
    locations: str = DEFAULT_LOCATIONS,
    max_exp_years: int = DEFAULT_MAX_EXP_YEARS,
    max_posts: int = 50,
    headless: bool = False,
) -> str:
    """
    Search LinkedIn posts for hiring activity matching the given filters
    (last 24h). `roles` and `locations` are comma-separated lists
    (e.g. "AI Engineer, Data Scientist" / "Remote, Jaipur").
    Requires cookies saved by manual_login.py. READ-ONLY.
    """
    role_list = _parse_list(roles)
    location_list = _parse_list(locations)
    _log(f"Filters — roles: {role_list} | locations: {location_list} | max_exp_years: {max_exp_years}")

    if not os.path.exists(COOKIE_PATH):
        return "❌ No session found. Run manual_login.py first."

    with open(COOKIE_PATH) as f:
        cookies = json.load(f)
    _log(f"Loaded {len(cookies)} cookies from {COOKIE_PATH}")

    _log("Launching Chromium...")
    pw = await async_playwright().start()
    browser = await pw.chromium.launch(
        headless=headless,
        slow_mo=80,
        args=["--no-sandbox", "--disable-blink-features=AutomationControlled"],
    )
    context = await browser.new_context(
        user_agent=(
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        viewport={"width": 1280, "height": 800},
        locale="en-US",
        timezone_id="Asia/Kolkata",
    )
    await context.add_cookies(cookies)
    _log("Browser context created and cookies applied.")
    page = await context.new_page()

    raw_matches = []  # [{"full_text": ..., "post_url": ...}]
    seen = _load_seen_jobs()
    previously_seen_count = len(seen)
    _log(f"Loaded {previously_seen_count} previously-seen job keys from {SEEN_JOBS_PATH}")

    LINKEDIN_SEARCH_URL = (
        "https://www.linkedin.com/search/results/content/"
        "?keywords={query}&datePosted=past-24h&sortBy=date_posted"
    )

    search_queries = [
        f"{role} hiring {location}"
        for role in role_list
        for location in location_list
    ][:8] or [DEFAULT_ROLES.split(",")[0].strip()]

    stats = {
        "cards_seen": 0, "skipped_short": 0, "skipped_duplicate": 0,
        "skipped_role": 0, "skipped_location_exp": 0,
        "skipped_llm": 0, "matched": 0,
    }

    try:
        # ── PHASE 1: Browse search pages and collect raw matching posts ────────
        _log("Phase 1 — Browsing search results and collecting raw matches (no LLM yet)...")
        for query in search_queries:
            if len(raw_matches) >= max_posts:
                break

            url = LINKEDIN_SEARCH_URL.format(query=query.replace(" ", "%20"))
            _log(f"Searching: {query}")

            await page.goto(url, wait_until="domcontentloaded", timeout=40_000)
            await asyncio.sleep(4)

            current_url = page.url
            if any(x in current_url for x in ["login", "authwall", "checkpoint"]):
                await context.close()
                await browser.close()
                await pw.stop()
                return "❌ LinkedIn session expired. Re-run: python manual_login.py"

            # Scroll to load more results
            for _ in range(5):
                await page.evaluate("window.scrollBy(0, window.innerHeight * 2)")
                await asyncio.sleep(random.uniform(1.5, 3.0))

            cards = await _get_cards(page)
            _log(f"Cards found: {len(cards)}")

            for card in cards:
                if len(raw_matches) >= max_posts:
                    break
                try:
                    stats["cards_seen"] += 1
                    full_text = (await card.inner_text()).strip()
                    if not full_text or len(full_text) < 40:
                        stats["skipped_short"] += 1
                        continue

                    # Post URL extraction (Robust 3-stage fallback)
                    post_url = "N/A"
                    try:
                        # Strategy 1: data-urn attribute
                        urn = await card.evaluate('''el => {
                            let node = el.querySelector('[data-urn^="urn:li:activity:"], [data-urn^="urn:li:ugcPost:"], [data-urn^="urn:li:share:"]');
                            if (node) return node.getAttribute('data-urn');
                            // Also check the element itself
                            if (el.getAttribute('data-urn')) return el.getAttribute('data-urn');
                            return null;
                        }''')
                        if urn and "urn:li:" in urn:
                            post_url = f"https://www.linkedin.com/feed/update/{urn}/"
                        else:
                            # Strategy 2: All hrefs in the card
                            all_hrefs = await card.evaluate('''el => {
                                return Array.from(el.querySelectorAll('a')).map(a => a.href);
                            }''')
                            for href in all_hrefs:
                                if href and any(x in href for x in ["urn:li:activity:", "urn:li:ugcPost:", "urn:li:share:", "/posts/", "/feed/update/"]):
                                    post_url = href.split("?")[0]
                                    break
                            
                            # Strategy 3: Regex the raw HTML for the URN
                            if post_url == "N/A":
                                html = await card.inner_html()
                                import re
                                # Catch standard and URL-encoded URNs of any type (activity, ugcPost, share, etc.)
                                match = re.search(r'(urn:li:[a-zA-Z]+:\d+)', html)
                                if not match:
                                    match = re.search(r'(urn%3Ali%3A[a-zA-Z]+%3A\d+)', html)
                                    
                                if match:
                                    raw_urn = match.group(1).replace("%3A", ":")
                                    post_url = f"https://www.linkedin.com/feed/update/{raw_urn}/"
                    except Exception as e:
                        _log(f"Error getting post_url: {e}")

                    dedup_keys = _dedup_keys(full_text, post_url)
                    if any(k in seen for k in dedup_keys):
                        stats["skipped_duplicate"] += 1
                        continue
                    seen.update(dedup_keys)

                    if not _matches_role(full_text, role_list):
                        stats["skipped_role"] += 1
                        if stats["skipped_role"] <= 3:
                            _log(f"  ⏭ role mismatch, snippet: {full_text[:100].replace(chr(10), ' ')!r}")
                        continue
                    if not (_matches_location(full_text, location_list) or _matches_experience(full_text, max_exp_years)):
                        stats["skipped_location_exp"] += 1
                        if stats["skipped_location_exp"] <= 3:
                            _log(f"  ⏭ location/exp mismatch, snippet: {full_text[:100].replace(chr(10), ' ')!r}")
                        continue

                    raw_matches.append({"full_text": full_text, "post_url": post_url})
                    _log(f"  📌 Collected: {full_text[:80].replace(chr(10), ' ')!r}")

                except Exception as ex:
                    _log(f"Card error: {ex}")
                    continue

        _log(f"Phase 1 complete — raw matches: {len(raw_matches)}, closing browser.")
        await context.close()
        await browser.close()
        await pw.stop()
        _save_seen_jobs(seen)
        _log(f"Saved {len(seen)} seen-job keys ({len(seen) - previously_seen_count} new) to {SEEN_JOBS_PATH}")

        # ── PHASE 2: LLM extraction (browser closed, no UI blocking) ──────────
        _log("Phase 2 — Running LLM extraction on collected posts...")
        results = []

        for raw in raw_matches:
            full_text = raw["full_text"]
            post_url  = raw["post_url"]
            try:
                extracted = await _llm_extract_fields(full_text)
                if not extracted.get("is_job_post", True):
                    stats["skipped_llm"] += 1
                    _log("  ⏭ LLM classified as non-job post, skipping.")
                    continue
                emails_list = extracted.get("emails") or []
                phones_list = extracted.get("phones") or []
                company     = extracted.get("company") or _extract_author_from_text(full_text)
                role_found  = extracted.get("role") or "AI/ML Role"
                location    = extracted.get("location") or _extract_location(full_text, location_list)
                whatsapp    = extracted.get("whatsapp") or "N/A"
                experience  = extracted.get("experience") or "N/A"
                apply_link  = extracted.get("apply_link") or post_url
                posted      = extracted.get("posted") or _extract_time_from_text(full_text)
                _log(
                    f"  🤖 LLM: {company} | {role_found} | "
                    f"emails={emails_list} | phones={phones_list}"
                )
            except Exception as llm_err:
                _log(f"  ⚠ LLM extraction failed ({llm_err}), falling back to regex.")
                company     = _extract_author_from_text(full_text)
                role_found  = "AI/ML Role"
                for role in role_list:
                    if role.lower() in full_text.lower():
                        role_found = role
                        break
                emails_list = []
                phones_list = []
                location    = _extract_location(full_text, location_list)
                whatsapp    = "N/A"
                experience  = "N/A"
                apply_link  = post_url
                posted      = _extract_time_from_text(full_text)

            stats["matched"] += 1
            results.append({
                "company":    company,
                "role":       role_found,
                "emails":     ", ".join(emails_list) if emails_list else _extract_email(full_text),
                "phones":     ", ".join(phones_list) if phones_list else "N/A",
                "whatsapp":   whatsapp,
                "location":   location,
                "experience": experience,
                "posted":     posted,
                "post_url":   post_url,
                "apply_link": apply_link,
                "snippet":    full_text[:400].replace("\n", " ").strip(),
            })
            _log(f"  ✅ Matched: {company} — {role_found}")

        _log(
            f"Search summary — cards seen: {stats['cards_seen']}, "
            f"skipped (short): {stats['skipped_short']}, "
            f"skipped (duplicate): {stats['skipped_duplicate']}, "
            f"skipped (role mismatch): {stats['skipped_role']}, "
            f"skipped (location/exp mismatch): {stats['skipped_location_exp']}, "
            f"skipped (LLM non-job): {stats['skipped_llm']}, "
            f"matched: {stats['matched']}"
        )
        if stats["cards_seen"] == 0:
            _log("  ⚠ No cards were found at all across any query — check debug output, "
                 "LinkedIn's markup may have changed or results didn't load.")
        elif stats["matched"] == 0:
            _log("  ⚠ Cards were found but none matched your filters — try broader roles/locations.")

        _log(f"Total results: {len(results)}")
        return json.dumps(results, indent=2, ensure_ascii=False)

    except Exception as e:
        try:
            await context.close()
            await browser.close()
            await pw.stop()
        except Exception:
            pass
        return f"❌ Search error: {e}"


@mcp.tool()
async def save_results_to_file(results_json: str, output_path: str = "linkedin_jobs.txt") -> str:
    """Save job results to a formatted .txt file. Local disk only."""
    _log(f"Saving results to '{output_path}'...")
    try:
        posts = json.loads(results_json)
    except json.JSONDecodeError:
        _log("❌ save_results_to_file received invalid JSON.")
        return "❌ Invalid JSON."

    lines = [
        "=" * 135,
        "LinkedIn AI/ML Job Posts — Last 24 Hours",
        f"Generated  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total      : {len(posts)} posts found",
        "=" * 135,
        "",
        f"{'Company':<25} | {'Role':<25} | {'Location':<15} | {'Apply Link':<30} | {'Job Poster (HR)'}",
        "-" * 135,
    ]
    for p in posts:
        lines.append(
            f"{p.get('company','N/A')[:25]:<25} | "
            f"{p.get('role','N/A')[:25]:<25} | "
            f"{p.get('location','N/A')[:15]:<15} | "
            f"{p.get('apply_link', p.get('post_url', 'N/A'))[:30]:<30} | "
            f"{p.get('poster_link', 'N/A')}"
        )

    lines += ["", "=" * 95, "DETAILED VIEW", "=" * 95]
    for i, p in enumerate(posts, 1):
        lines += [
            f"\n[{i}] {p.get('company','N/A')} — {p.get('role','N/A')}",
            f"    Emails     : {p.get('emails','N/A')}",
            f"    Phones     : {p.get('phones','N/A')}",
            f"    WhatsApp   : {p.get('whatsapp','N/A')}",
            f"    Location   : {p.get('location','N/A')}",
            f"    Experience : {p.get('experience','N/A')}",
            f"    Posted     : {p.get('posted','N/A')}",
            f"    Post URL   : {p.get('post_url','N/A')}",
            f"    Apply Link : {p.get('apply_link','N/A')}",
            f"    Job Poster : {p.get('poster_link','N/A')}",
            f"    Snippet    : {p.get('snippet','N/A')}",
        ]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    _log(f"✅ Saved {len(posts)} posts to '{output_path}'")
    return f"✅ Saved {len(posts)} posts to '{output_path}'"


if __name__ == "__main__":
    _log("=" * 60)
    _log(f"MCP server starting. Logs: {LOG_PATH}")
    _log("=" * 60)
    mcp.run(transport="stdio")