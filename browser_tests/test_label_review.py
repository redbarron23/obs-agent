"""Run the production review UI in Chromium with fictional data and private temporary state."""

import json
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import pytest
from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "output" / "playwright"
REVIEWER = "browser-review-test"


@pytest.fixture
def app(tmp_path):
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    env = {
        **os.environ,
        "OBS_AGENT_DEMO": "0",
        "OBS_AGENT_DECISIONS": str(tmp_path / "decisions.db"),
        "OBS_AGENT_REVIEW_DIR": str(tmp_path / "reviews"),
        "OBS_AGENT_REVIEWER": "",
        "OBS_AGENT_RAG_URL": "",
        "OBS_AGENT_MLFLOW": "0",
    }
    env.pop("MLFLOW_TRACKING_URI", None)
    with (ARTIFACTS / "streamlit.log").open("w") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                "app.py",
                "--server.headless=true",
                "--server.address=127.0.0.1",
                f"--server.port={port}",
                "--browser.gatherUsageStats=false",
            ],
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 30
            while True:
                assert process.poll() is None, (
                    "Streamlit exited; see output/playwright/streamlit.log"
                )
                try:
                    with urlopen(url + "/_stcore/health", timeout=1) as response:
                        if response.status == 200:
                            break
                except (URLError, TimeoutError):
                    pass
                assert time.monotonic() < deadline, "Streamlit did not become healthy in 30 seconds"
                time.sleep(0.1)
            yield url
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)


def review_page(page, url, *, reopen=False):
    page.goto(url)
    page.get_by_test_id("stSidebar").get_by_role(
        "link", name="Import / reopen inventory", exact=True
    ).click()
    if reopen:
        page.get_by_role("button", name="Open saved review", exact=True).click()
    else:
        page.get_by_test_id("stFileUploader").filter(
            has_text="Sanitized Azure/GCP inventory CSV"
        ).locator('input[type="file"]').set_input_files(
            ROOT / "fixtures" / "review_persistence_inventory.csv"
        )
        page.get_by_role("button", name="Save and open review", exact=True).click()
    page.get_by_role("link", name="Open Label review", exact=True).click()
    page.get_by_role("textbox", name="Reviewer name or ID", exact=True).fill(REVIEWER)
    page.get_by_role("textbox", name="Reviewer name or ID", exact=True).press("Enter")


def cell(page, resource, column):
    search = page.get_by_role("textbox", name="Resource contains", exact=True)
    search.fill(resource)
    search.press("Enter")
    editor = page.locator('[class*="st-key-editor-"]').get_by_test_id("stDataFrame")
    # Glide exposes its canvas cells through an accessibility table, including column indexes.
    row = editor.get_by_role("row", include_hidden=True).filter(
        has=page.get_by_role("gridcell", name=resource, exact=True, include_hidden=True)
    )
    expect(row).to_have_count(1)
    header = editor.get_by_role("columnheader", name=column, exact=True, include_hidden=True)
    expect(header).to_have_count(1)
    index = header.get_attribute("aria-colindex")
    return row.locator(f'[role="gridcell"][aria-colindex="{index}"]')


def dropdown(page, resource, column):
    target = cell(page, resource, column)
    target.focus()
    expect(target).to_have_attribute("aria-selected", "true")
    target.press("Enter")
    editor = page.get_by_test_id("portal").get_by_role("combobox")
    expect(editor).to_be_visible()
    return editor


def choose(page, resource, column, value):
    dropdown(page, resource, column).fill(value)
    page.get_by_role("option", name=value, exact=True).click()
    expect(cell(page, resource, column)).to_have_text(value)


def save(page):
    page.get_by_role("button", name=re.compile("Save decisions")).click()
    expect(page.get_by_test_id("stToast")).to_contain_text("Saved 1 decision(s)")


def download_plan(page):
    with page.expect_download() as pending:
        page.get_by_role("button", name=re.compile("Reviewed change plan")).click()
    download = pending.value
    assert download.suggested_filename == "approved-label-change-plan.json"
    return json.loads(Path(download.path()).read_text())


def test_review_decisions_reload_and_export(app):
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        context = browser.new_context(viewport={"width": 1920, "height": 1080})
        context.tracing.start(screenshots=True, snapshots=True)
        page = context.new_page()
        page.set_default_timeout(15000)
        try:
            review_page(page, app)
            choose(page, "gcp-demo-003", "Decision", "accept")
            save(page)
            choose(page, "az-demo-001", "Decision", "reject")
            save(page)
            page.get_by_role("radio", name="env", exact=True).check()
            choose(page, "gcp-demo-003", "Decision", "override")
            invalid = dropdown(page, "gcp-demo-003", "Override value")
            invalid.fill("not-an-allowed-environment")
            expect(page.get_by_role("option")).to_have_count(0)
            invalid.press("Enter")
            page.keyboard.press("Escape")
            page.get_by_role("button", name=re.compile("Save decisions")).click()
            expect(
                page.get_by_test_id("stAlert").filter(has_text="not a valid env")
            ).to_be_visible()
            invalid_plan = download_plan(page)
            assert "not-an-allowed-environment" not in json.dumps(invalid_plan)
            assert all("env" not in change["tags"] for change in invalid_plan["changes"])
            choose(page, "gcp-demo-003", "Override value", "prod")
            save(page)
            original_plan = download_plan(page)
            page.screenshot(path=str(ARTIFACTS / "saved-review.png"), full_page=True)
            context.tracing.stop(path=str(ARTIFACTS / "editing-trace.zip"))
            context.close()

            context = browser.new_context(viewport={"width": 1920, "height": 1080})
            context.tracing.start(screenshots=True, snapshots=True)
            page = context.new_page()
            page.set_default_timeout(15000)
            review_page(page, app, reopen=True)
            expect(cell(page, "gcp-demo-003", "Decision")).to_have_text("accept")
            expect(cell(page, "az-demo-001", "Decision")).to_have_text("reject")
            page.get_by_role("radio", name="env", exact=True).check()
            expect(cell(page, "gcp-demo-003", "Decision")).to_have_text("override")
            expect(cell(page, "gcp-demo-003", "Override value")).to_have_text("prod")
            plan = download_plan(page)
            for result in (original_plan, plan):
                assert result["reviewed_only"] is True
                assert len(result["changes"]) == 1
                change = result["changes"][0]
                assert change["resource_id"] == "gcp-demo-003"
                assert change["tags"]["team"] == {
                    "previous": None,
                    "new": "data-platform",
                    "how": "accept",
                }
                assert change["tags"]["env"] == {"previous": None, "new": "prod", "how": "override"}
                assert len(result["approvals"]) == 2
                assert {a["label"] for a in result["approvals"]} == {"team", "env"}
                for approval in result["approvals"]:
                    assert approval["resource_id"] == "gcp-demo-003"
                    assert approval["reviewer"] == REVIEWER
                    for key in ("inventory_id", "model_version"):
                        assert approval[key] == result[key]
                del result["generated_at"]
            assert plan == original_plan, "reopened review exported different approval evidence"
        finally:
            page.screenshot(path=str(ARTIFACTS / "last-page.png"), full_page=True)
            (ARTIFACTS / "last-page.txt").write_text(page.locator("body").aria_snapshot())
            context.tracing.stop(path=str(ARTIFACTS / "trace.zip"))
            context.close()
            browser.close()
