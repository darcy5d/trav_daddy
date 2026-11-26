"""The TOP-UP COMPLETE banner must report the finish time, not the start time.

`main()` captures a start timestamp to name the run (and its default output
file). That same value used to be printed in the final banner, so a run that
began 00:24 and finished 09:14 announced "TOP-UP COMPLETE 2026-11-08 00:24",
which read as an 8h50m failure instead of a successful overnight run.

These tests pin the small helper that builds the banner so the regression cannot
come back without running the whole script (which would hit the network).
"""

import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.ca_topup import format_elapsed, render_summary_lines  # noqa: E402


def _banner(started: datetime, finished: datetime) -> list[str]:
    return render_summary_lines(
        started=started,
        finished=finished,
        new_urls=["https://example/1.html", "https://example/2.html"],
        existing_count=100,
        out_path=Path("/tmp/list.txt"),
    )


def test_banner_reports_finish_not_start_time():
    started = datetime(2026, 11, 8, 0, 24)
    finished = datetime(2026, 11, 8, 9, 14)
    line = _banner(started, finished)[1]
    assert line.startswith("TOP-UP COMPLETE")
    assert "2026-11-08 09:14" in line, line
    # The start time is still shown, but labelled as the start.
    assert "started 2026-11-08 00:24" in line, line


def test_banner_reports_elapsed_duration():
    started = datetime(2026, 11, 8, 0, 24)
    finished = datetime(2026, 11, 8, 9, 14)
    line = _banner(started, finished)[1]
    assert "8h 50m 00s" in line, line


def test_banner_counts_are_unchanged():
    lines = _banner(datetime(2026, 1, 1, 10, 0), datetime(2026, 1, 1, 10, 5))
    body = "\n".join(lines)
    assert "New scorecards found : 2" in body
    assert "DB before            : 100" in body
    assert "DB after (est.)      : 102" in body
    assert "URL list             : /tmp/list.txt" in body


def test_banner_is_a_boxed_block():
    lines = _banner(datetime(2026, 1, 1, 10, 0), datetime(2026, 1, 1, 10, 5))
    assert lines[0] == "=" * 72
    assert lines[-1] == "=" * 72
    stamp = re.compile(r"^TOP-UP COMPLETE\s+\d{4}-\d{2}-\d{2} \d{2}:\d{2}")
    assert stamp.match(lines[1]), lines[1]


def test_elapsed_formatting():
    base = datetime(2026, 11, 8, 0, 0)
    assert format_elapsed(base, datetime(2026, 11, 8, 0, 0, 42)) == "42s"
    assert format_elapsed(base, datetime(2026, 11, 8, 0, 1, 0)) == "1m 00s"
    assert format_elapsed(base, datetime(2026, 11, 8, 0, 5, 7)) == "5m 07s"
    assert format_elapsed(base, datetime(2026, 11, 8, 2, 3, 4)) == "2h 03m 04s"


def test_elapsed_never_goes_negative():
    # A clock step backwards must not print a nonsense duration.
    started = datetime(2026, 11, 8, 9, 0)
    finished = datetime(2026, 11, 8, 8, 0)
    assert format_elapsed(started, finished) == "0s"
