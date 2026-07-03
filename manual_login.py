"""
Run this ONCE manually to log in and solve any LinkedIn verification.
After you complete login in the browser window, press Enter in the terminal.
This saves valid cookies that mcp_server.py will use for all searches.
"""
import asyncio, json, os
from playwright.async_api import async_playwright

COOKIE_PATH = ".li_session/cookies.json"

async def main():
    pw = await async_playwright().start()
    browser = await pw.chromium.launch(
        headless=False,
        slow_mo=100,
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
    page = await context.new_page()
    await page.goto("https://www.linkedin.com/login")

    print("\n" + "="*60)
    print("  Browser is open.")
    print("  1. Log in with your LinkedIn credentials")
    print("  2. Complete any verification (OTP, CAPTCHA, etc.)")
    print("  3. Wait until you see your LinkedIn FEED")
    print("  4. Come back here and press ENTER")
    print("="*60)
    input("\nPress ENTER once you are on the LinkedIn feed page: ")

    cookies = await context.cookies()
    os.makedirs(".li_session", exist_ok=True)
    with open(COOKIE_PATH, "w") as f:
        json.dump(cookies, f)

    print(f"\n✅ Saved {len(cookies)} cookies to {COOKIE_PATH}")
    print("You can now run: python mcp_client.py")

    await context.close()
    await browser.close()
    await pw.stop()

asyncio.run(main())