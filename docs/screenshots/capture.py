"""Regenerates the screenshots used in the README. Not part of the app.

Requires a running server (`python serve/app.py`, or any host:port serving
serve/app.py) and Playwright hooked up to the system Chrome install:

    pip install playwright
    playwright install chrome
    APP_URL=http://localhost:8000 python docs/screenshots/capture.py
"""
import os

from playwright.sync_api import sync_playwright

URL = os.environ.get("APP_URL", "http://localhost:8000")
OUT = "docs/screenshots"


def crop_to_content(page, path, extra_bottom=24):
    box = page.locator("main").bounding_box()
    height = int(box["y"] + box["height"] + extra_bottom)
    page.screenshot(path=path, clip={"x": 0, "y": 0, "width": 1280, "height": height})


with sync_playwright() as p:
    browser = p.chromium.launch(channel="chrome")
    page = browser.new_page(viewport={"width": 1280, "height": 1000})

    # 1. Default search, relevance filter ON -- the core "before/after" story
    page.goto(URL)
    page.wait_for_selector(".card")
    page.click("#filtered-out-toggle")
    page.wait_for_timeout(200)
    crop_to_content(page, f"{OUT}/search_filtered.png")

    # 1b. "cleaning" query -- shows the semantic-match badge on items the
    # keyword matcher alone could never find (e.g. "Paper Towels")
    page.fill("#query", "cleaning")
    page.click("#search-btn")
    page.wait_for_selector(".card")
    page.wait_for_timeout(300)
    crop_to_content(page, f"{OUT}/semantic_retrieval.png")
    page.fill("#query", "salt")
    page.click("#search-btn")
    page.wait_for_selector(".card")
    page.wait_for_timeout(300)

    # 2. Same query, relevance filter OFF -- raw keyword retrieval
    page.click("#filter-toggle")
    page.wait_for_timeout(300)
    crop_to_content(page, f"{OUT}/search_unfiltered.png")

    # 3. "Why?" panel expanded, then ask-the-local-teacher comparison filled in
    page.click("#filter-toggle")  # back on
    page.wait_for_timeout(300)
    page.click(".card .card-why")
    page.wait_for_timeout(200)
    page.click(".card .teacher-btn")
    page.wait_for_selector(".card .teacher-answer strong", timeout=20000)
    page.wait_for_timeout(200)
    crop_to_content(page, f"{OUT}/teacher_comparison.png")

    # 4. Persona selected -- personalization re-ranking + match% badges
    page.select_option("#persona-select", "home_chef_carlos")
    page.wait_for_selector(".card .tag.personalized")
    page.wait_for_timeout(300)
    crop_to_content(page, f"{OUT}/personalization.png")

    browser.close()

print("done")
