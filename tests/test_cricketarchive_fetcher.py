"""The polite fetcher must not lose its login when a connection resets.

The archive closes idle keep-alive connections mid-run. The retry path used to
build a bare requests.Session, which dropped pigeon_access, so every retry went
out anonymous and came back HTTP 403. That took out a live harvest at combo 47
of 428 and would have taken out the weekly job the same way.
"""

import unittest
from unittest import mock

import requests

from src.api.cricketarchive.fetcher import NotAuthenticatedError, PoliteFetcher

CFG = {
    "cache_dir": "/tmp/ca-fetch-test",
    "base_url": "https://cricketarchive.com",
    "max_requests_per_day": 10_000,
    "request_timeout_sec": 5,
    "min_delay_sec": 0,
    "max_delay_sec": 0,
    "max_retries": 1,
    "backoff_base_sec": 0,
    "user_agent": "test",
}


def _session_with_archive_cookie():
    s = requests.Session()
    s.cookies.set("pigeon_access", "secret-value", domain="cricketarchive.com", path="/")
    return s


class TestCookieSurvivesReset(unittest.TestCase):
    def setUp(self):
        self.fetcher = PoliteFetcher(_session_with_archive_cookie(), config=CFG)
        self.fetcher._robots_allows = lambda url: True
        self.fetcher._throttle = lambda: None
        self.fetcher._check_and_bump_counter = lambda: None

    def test_reset_session_keeps_the_archive_cookie(self):
        self.fetcher._reset_session()
        self.assertEqual(
            self.fetcher.session.cookies.get("pigeon_access"), "secret-value"
        )

    def test_connection_error_retry_is_still_authenticated(self):
        seen = []
        fetcher = self.fetcher

        def fake_get(session_self, url, **_kwargs):
            seen.append(session_self.cookies.get("pigeon_access"))
            if len(seen) == 1:
                raise requests.exceptions.ConnectionError("RemoteDisconnected")
            resp = mock.Mock()
            resp.status_code = 200
            resp.text = "<html>real scorecard</html>"
            resp.raise_for_status = lambda: None
            return resp

        with mock.patch.object(requests.Session, "get", fake_get), mock.patch("time.sleep"):
            html = fetcher.get("https://cricketarchive.com/x", force=True)

        self.assertIn("real scorecard", html)
        self.assertEqual(seen, ["secret-value", "secret-value"])

    def test_exhausted_403_raises_not_authenticated(self):
        def fake_get(session_self, url, **_kwargs):
            resp = mock.Mock()
            resp.status_code = 403
            resp.text = "paywall"
            resp.raise_for_status = lambda: None
            return resp

        with mock.patch.object(requests.Session, "get", fake_get), mock.patch("time.sleep"):
            with self.assertRaises(NotAuthenticatedError) as caught:
                self.fetcher.get("https://cricketarchive.com/x", force=True)
        self.assertIn("403", str(caught.exception))
        self.assertIn("Cursor browser", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
