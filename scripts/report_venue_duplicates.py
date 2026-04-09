#!/usr/bin/env python3
"""
Generate a fuzzy duplicate report for venue names.

Usage:
    python scripts/report_venue_duplicates.py
"""

import csv
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data.database import get_connection
from src.data.venue_normalizer import venue_similarity


def _norm(s: str) -> str:
    return (s or "").strip().lower()


def find_duplicate_candidates(threshold: float = 0.88):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(venues)")
    columns = {row["name"] for row in cur.fetchall()}
    has_state = "state" in columns

    if has_state:
        cur.execute("SELECT venue_id, name, city, country, state FROM venues ORDER BY country, state, city, name")
    else:
        cur.execute("SELECT venue_id, name, city, country, '' AS state FROM venues ORDER BY country, city, name")
    rows = cur.fetchall()
    conn.close()

    by_bucket = {}
    for r in rows:
        key = (_norm(r["country"]), _norm(r["state"]), _norm(r["city"]))
        by_bucket.setdefault(key, []).append(r)

    candidates = []
    for bucket_rows in by_bucket.values():
        n = len(bucket_rows)
        if n < 2:
            continue
        for i in range(n):
            for j in range(i + 1, n):
                a = bucket_rows[i]
                b = bucket_rows[j]
                score = venue_similarity(a["name"], b["name"])
                if score >= threshold:
                    candidates.append(
                        {
                            "score": round(score, 4),
                            "venue_id_1": a["venue_id"],
                            "name_1": a["name"],
                            "venue_id_2": b["venue_id"],
                            "name_2": b["name"],
                            "city": a["city"] or b["city"] or "",
                            "state": a["state"] or b["state"] or "",
                            "country": a["country"] or b["country"] or "",
                        }
                    )

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates


def main():
    candidates = find_duplicate_candidates()
    out_dir = project_root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "venue_duplicate_candidates.csv"

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "score",
                "venue_id_1",
                "name_1",
                "venue_id_2",
                "name_2",
                "city",
                "state",
                "country",
            ],
        )
        writer.writeheader()
        writer.writerows(candidates)

    print(f"Wrote {len(candidates)} candidates to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
