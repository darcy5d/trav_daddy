#!/usr/bin/env python3
"""Drive the indias_dad Flask UI with Playwright and capture evidence.

Usage (always with the repo's venv311 interpreter):
    venv311/bin/python .cursor/skills/verify-indias-dad/scripts/drive.py \
        [--base http://127.0.0.1:5001] [--run-id NAME] [--headed] \
        [--timeout-ms 30000] [--allow-post /api/path ...] STEP [STEP ...]

Steps (executed in order, each recorded in result.json):
    goto:/path                  navigate (relative to --base)
    wait:SELECTOR               wait until visible
    waitgone:SELECTOR           wait until hidden/detached
    waittext:SELECTOR~REGEX     wait until innerText matches REGEX
    click:SELECTOR
    select:SELECTOR=VALUE       select <option> by value, falling back to label
    fill:SELECTOR=VALUE
    text:SELECTOR               record innerText (first match)
    count:SELECTOR              record number of matches
    expect:SELECTOR~REGEX       assert innerText matches REGEX
    expectcount:SELECTOR>=N     assert at least N matches
    dialogs:accept|dismiss      how to answer later alert/confirm (default dismiss)
    shot:NAME                   full-page screenshot NAME.png
    sleep:MS

Safety: every non-GET/HEAD/OPTIONS request the page makes is aborted unless
its path is in SAFE_POSTS (pure compute / read-only market lookups) or passed
via --allow-post. --allow-post is refused against the live port 5001.

Evidence goes to logs/verify/<run-id>/ (gitignored): result.json,
network.jsonl, console.log, *.png. Exit 0 = all steps passed, 1 = a step
failed, 2 = usage error.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parents[4]
EVIDENCE_ROOT = REPO_ROOT / "logs" / "verify"
LIVE_PORT = 5001

SAFE_POSTS = {
    "/api/simulate",
    "/api/simulate-stream",
    "/api/players/by-ids",
    "/api/team/optimize",
    "/api/integrations/polymarket/compare/batch",
    "/api/integrations/polymarket/compare/multi",
    "/api/integrations/polymarket/compare/multi/batch",
}
SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", default=f"http://127.0.0.1:{LIVE_PORT}")
    p.add_argument("--run-id", default=None)
    p.add_argument("--headed", action="store_true")
    p.add_argument("--timeout-ms", type=int, default=30000)
    p.add_argument("--allow-post", action="append", default=[])
    p.add_argument("steps", nargs="+")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    base = args.base.rstrip("/")
    port = urlparse(base).port
    if args.allow_post and port == LIVE_PORT:
        print("refusing --allow-post against the live instance on :5001; use a scratch instance", file=sys.stderr)
        return 2
    allowed_posts = SAFE_POSTS | set(args.allow_post)

    run_id = args.run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    out = EVIDENCE_ROOT / run_id
    n = 2
    while out.exists():
        out = EVIDENCE_ROOT / f"{run_id}-{n}"
        n += 1
    run_id = out.name
    out.mkdir(parents=True)
    net_f = (out / "network.jsonl").open("a")
    con_f = (out / "console.log").open("a")

    from playwright.sync_api import sync_playwright

    result = {
        "run_id": run_id,
        "base": base,
        "started": datetime.now().isoformat(timespec="seconds"),
        "steps": [],
        "blocked_requests": [],
        "page_errors": [],
        "passed": False,
    }
    dialog_mode = {"value": "dismiss"}

    def log_net(entry: dict) -> None:
        net_f.write(json.dumps(entry) + "\n")
        net_f.flush()

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=not args.headed)
        ctx = browser.new_context(viewport={"width": 1440, "height": 1000})
        page = ctx.new_page()
        page.set_default_timeout(args.timeout_ms)

        def guard(route, request):
            path = urlparse(request.url).path
            if request.method in SAFE_METHODS or path in allowed_posts:
                route.continue_()
                return
            entry = {"method": request.method, "url": request.url, "blocked": True}
            result["blocked_requests"].append(entry)
            log_net(entry)
            route.abort()

        page.route("**/*", guard)
        page.on("response", lambda r: log_net({"method": r.request.method, "url": r.url, "status": r.status}))
        page.on("console", lambda m: con_f.write(f"[{m.type}] {m.text}\n"))
        page.on("pageerror", lambda e: (result["page_errors"].append(str(e)), con_f.write(f"[pageerror] {e}\n")))

        def on_dialog(d):
            con_f.write(f"[dialog:{d.type}] {d.message} -> {dialog_mode['value']}\n")
            d.accept() if dialog_mode["value"] == "accept" else d.dismiss()

        page.on("dialog", on_dialog)

        ok = True
        for raw in args.steps:
            kind, _, arg = raw.partition(":")
            rec = {"step": raw, "ok": True}
            t0 = time.time()
            try:
                if kind == "goto":
                    resp = page.goto(base + arg if arg.startswith("/") else arg, wait_until="domcontentloaded")
                    rec["status"] = resp.status if resp else None
                    if resp and resp.status >= 400:
                        raise AssertionError(f"HTTP {resp.status}")
                elif kind == "wait":
                    page.locator(arg).first.wait_for(state="visible")
                elif kind == "waitgone":
                    page.locator(arg).first.wait_for(state="hidden")
                elif kind == "waittext":
                    sel, _, rx = arg.partition("~")
                    deadline = time.time() + args.timeout_ms / 1000
                    while True:
                        txt = page.locator(sel).first.inner_text()
                        if re.search(rx, txt, re.S):
                            rec["value"] = txt[:500]
                            break
                        if time.time() > deadline:
                            raise AssertionError(f"text of {sel!r} never matched /{rx}/; last={txt[:200]!r}")
                        page.wait_for_timeout(250)
                elif kind == "click":
                    page.locator(arg).first.click()
                elif kind == "select":
                    sel, _, val = arg.partition("=")
                    loc = page.locator(sel).first
                    try:
                        loc.select_option(value=val, timeout=3000)
                    except Exception:
                        loc.select_option(label=val)
                elif kind == "fill":
                    sel, _, val = arg.partition("=")
                    page.locator(sel).first.fill(val)
                elif kind == "text":
                    rec["value"] = page.locator(arg).first.inner_text()[:2000]
                elif kind == "count":
                    rec["value"] = page.locator(arg).count()
                elif kind == "expect":
                    sel, _, rx = arg.partition("~")
                    txt = page.locator(sel).first.inner_text()
                    rec["value"] = txt[:500]
                    if not re.search(rx, txt, re.S):
                        raise AssertionError(f"text of {sel!r} does not match /{rx}/")
                elif kind == "expectcount":
                    sel, _, n = arg.rpartition(">=")
                    c = page.locator(sel).count()
                    rec["value"] = c
                    if c < int(n):
                        raise AssertionError(f"{sel!r} count {c} < {n}")
                elif kind == "dialogs":
                    dialog_mode["value"] = arg
                elif kind == "shot":
                    path = out / f"{arg}.png"
                    page.screenshot(path=str(path), full_page=True)
                    rec["file"] = str(path)
                elif kind == "sleep":
                    page.wait_for_timeout(int(arg))
                else:
                    raise ValueError(f"unknown step kind {kind!r}")
            except Exception as e:
                ok = False
                rec["ok"] = False
                rec["error"] = f"{type(e).__name__}: {e}"[:1000]
                fail = out / "FAILED.png"
                try:
                    page.screenshot(path=str(fail), full_page=True)
                    rec["file"] = str(fail)
                except Exception:
                    pass
            rec["ms"] = int((time.time() - t0) * 1000)
            result["steps"].append(rec)
            print(("PASS " if rec["ok"] else "FAIL ") + raw + (f"  -> {rec.get('value')!r}" if "value" in rec else "")
                  + (f"  !! {rec['error']}" if not rec["ok"] else ""))
            if not ok:
                break

        browser.close()

    result["passed"] = ok
    result["finished"] = datetime.now().isoformat(timespec="seconds")
    (out / "result.json").write_text(json.dumps(result, indent=2))
    net_f.close()
    con_f.close()
    print(f"{'PASSED' if ok else 'FAILED'}  blocked={len(result['blocked_requests'])}  "
          f"page_errors={len(result['page_errors'])}  evidence={out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
