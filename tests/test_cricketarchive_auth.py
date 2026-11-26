"""Saved CricketArchive browser sessions, without the network.

Headless login never receives pigeon_access. The weekly job must reuse or
refresh that cookie, and must say so when it has expired.
"""

import importlib.util
import json
import os
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from src.api.cricketarchive import auth


def _cookie(name, value, expires):
    return {
        "name": name,
        "value": value,
        "domain": ".cricketarchive.com",
        "path": "/",
        "expires": expires,
        "httpOnly": False,
        "secure": True,
        "sameSite": "Lax",
    }


def _load_topup():
    path = Path(__file__).resolve().parents[1] / "scripts" / "ca_topup.py"
    spec = importlib.util.spec_from_file_location("ca_topup_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestChooseSessionCookies(unittest.TestCase):
    def test_reuses_saved_cookie_when_browser_jar_is_empty(self):
        saved = [_cookie("pigeon_access", "saved", time.time() + 3600)]
        chosen, reason = auth.choose_session_cookies(saved, [], now=time.time())
        self.assertEqual(reason, "reused")
        self.assertEqual(chosen, saved)

    def test_refreshes_when_saved_cookie_has_no_expiry_and_browser_does(self):
        saved = [_cookie("pigeon_access", "same", -1)]
        imported = [_cookie("pigeon_access", "same", time.time() + 86400)]
        chosen, reason = auth.choose_session_cookies(saved, imported, now=time.time())
        self.assertEqual(reason, "refreshed")
        self.assertEqual(chosen, imported)

    def test_expired_saved_cookie_is_replaced_by_browser(self):
        now = time.time()
        saved = [_cookie("pigeon_access", "old", now - 10)]
        imported = [_cookie("pigeon_access", "new", now + 3600)]
        chosen, reason = auth.choose_session_cookies(saved, imported, now=now)
        self.assertEqual(reason, "refreshed")
        self.assertEqual(chosen[0]["value"], "new")

    def test_expired_with_nothing_newer_is_expired(self):
        now = time.time()
        saved = [_cookie("pigeon_access", "old", now - 10)]
        chosen, reason = auth.choose_session_cookies(saved, [], now=now)
        self.assertIsNone(chosen)
        self.assertEqual(reason, "expired")

    def test_missing_when_neither_side_has_the_archive_cookie(self):
        chosen, reason = auth.choose_session_cookies([], [], now=time.time())
        self.assertIsNone(chosen)
        self.assertEqual(reason, "missing")


class TestRefreshDue(unittest.TestCase):
    def test_missing_or_expired_is_due(self):
        now = time.time()
        self.assertTrue(auth.refresh_due(None, now=now))
        self.assertTrue(auth.refresh_due(_cookie("pigeon_access", "x", now - 10), now=now))

    def test_inside_margin_is_due_and_outside_is_not(self):
        now = time.time()
        soon = _cookie("pigeon_access", "x", now + 47 * 3600)
        later = _cookie("pigeon_access", "x", now + 49 * 3600)
        self.assertTrue(auth.refresh_due(soon, now=now))
        self.assertFalse(auth.refresh_due(later, now=now))


class TestEnsureSession(unittest.TestCase):
    def _cfg(self, state: Path):
        return {
            "auth_state_path": state,
            "user_agent": "ua",
            "auth_max_age_hours": 168,
        }

    def test_old_file_is_reused_and_headless_login_is_not_called(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "auth_state.json"
            future = time.time() + 3600
            state.write_text(json.dumps({
                "cookies": [_cookie("pigeon_access", "still-good", future)],
                "origins": [],
            }))
            old = time.time() - 200 * 3600
            os.utime(state, (old, old))

            def boom(*_a, **_k):
                raise AssertionError("headless login must not run")

            with mock.patch.object(auth, "login_playwright", boom), \
                 mock.patch.object(auth, "_cfg", lambda: self._cfg(state)), \
                 mock.patch.object(auth, "cursor_browser_cookie_db", lambda: Path(tmp) / "absent"):
                session = auth.ensure_session()
            self.assertEqual(session.cookies.get("pigeon_access"), "still-good")

    def test_expired_cookie_raises_without_headless_login(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "auth_state.json"
            state.write_text(json.dumps({
                "cookies": [_cookie("pigeon_access", "dead", time.time() - 60)],
                "origins": [],
            }))

            def boom(*_a, **_k):
                raise AssertionError("headless login must not run")

            with mock.patch.object(auth, "login_playwright", boom), \
                 mock.patch.object(auth, "_cfg", lambda: self._cfg(state)), \
                 mock.patch.object(auth, "cursor_browser_cookie_db", lambda: Path(tmp) / "absent"):
                with self.assertRaises(RuntimeError) as caught:
                    auth.ensure_session()
            self.assertIn("expired", str(caught.exception))
            # The dead snapshot is left in place; login must not replace it.
            self.assertIn("dead", state.read_text())

    def test_browser_jar_refreshes_saved_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "auth_state.json"
            state.write_text(json.dumps({"cookies": [], "origins": []}))
            db = Path(tmp) / "Cookies"
            expires_utc = int((time.time() + 7 * 86400 + auth._CHROMIUM_EPOCH_OFFSET) * 1_000_000)
            conn = sqlite3.connect(db)
            conn.execute(
                """
                CREATE TABLE cookies (
                    name TEXT, value TEXT, host_key TEXT, path TEXT,
                    expires_utc INTEGER, is_secure INTEGER, is_httponly INTEGER,
                    samesite INTEGER
                )
                """
            )
            conn.execute(
                "INSERT INTO cookies VALUES (?,?,?,?,?,?,?,?)",
                ("pigeon_access", "from-browser", ".cricketarchive.com", "/",
                 expires_utc, 1, 0, 1),
            )
            conn.execute(
                "INSERT INTO cookies VALUES (?,?,?,?,?,?,?,?)",
                ("pigeon_access", "v10-encrypted", ".cricketarchive.com", "/",
                 expires_utc, 1, 0, 1),
            )
            conn.commit()
            conn.close()

            with mock.patch.object(auth, "_cfg", lambda: self._cfg(state)), \
                 mock.patch.object(auth, "cursor_browser_cookie_db", lambda: db), \
                 mock.patch.object(auth, "login_playwright", lambda *_a, **_k: state):
                session = auth.ensure_session()
            self.assertEqual(session.cookies.get("pigeon_access"), "from-browser")
            saved = json.loads(state.read_text())
            names = [c["name"] for c in saved["cookies"]]
            self.assertEqual(names, ["pigeon_access"])
            self.assertGreater(saved["cookies"][0]["expires"], time.time())
            self.assertEqual(state.stat().st_mode & 0o777, 0o600)


class TestLoginDoesNotDropArchiveCookie(unittest.TestCase):
    def test_commit_keeps_unexpired_archive_cookie(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "auth_state.json"
            fresh = time.time() + 3600
            state.write_text(json.dumps({
                "cookies": [_cookie("pigeon_access", "keep-me", fresh)],
                "origins": [],
            }))
            login_tmp = state.with_suffix(".login.tmp")
            login_tmp.write_text(json.dumps({
                "cookies": [_cookie("ea_id", "only-account", fresh)],
                "origins": [],
            }))
            auth._commit_login_state(state, login_tmp)
            saved = json.loads(state.read_text())
            self.assertEqual(saved["cookies"][0]["value"], "keep-me")
            self.assertFalse(login_tmp.exists())


class TestOtherTopupPids(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.topup = _load_topup()

    def test_ignores_this_process_its_parent_and_the_shell_line(self):
        ps = "\n".join([
            "10 1 /bin/bash -lc python scripts/ca_topup.py --harvest",
            "11 10 ./venv311/bin/python scripts/ca_topup.py --harvest",
            "12 11 ./venv311/bin/python scripts/ca_harvest_parallel.py --list x",
            "99 1 ./venv311/bin/python scripts/ca_topup.py --harvest",
        ])
        self.assertEqual(self.topup.other_topup_pids(ps, my_pid=11, parent_pid=10), [99])


if __name__ == "__main__":
    unittest.main()
