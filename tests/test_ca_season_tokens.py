"""Pin the season tokens `_season_tokens()` generates.

Discovery tries several season spellings per year because a country's T20
cricket can live under the calendar year ("2026") or the split season
("2025-26"). Getting the set wrong in the *narrow* direction is the dangerous
one: a token that is never requested is a list that is never read, and the loss
is silent.

The full-year spellings ("2024-2025", "2025-2026", "2026-2027") were pruned on
8 Nov 2026 after measuring that they never return cards. These tests pin the
resulting set so a future edit cannot quietly widen it back (slow, harmless) or
narrow it further (fast, lossy) without a deliberate test change.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.ca_topup import (  # noqa: E402
    ALL_COUNTRIES,
    TOKENS_BY_YEAR,
    _match_list_url,
    _season_tokens,
)

# Measured 8 Nov 2026: each of these returned cards for at least one country.
YIELDING_TOKENS = {
    2025: ["2025", "2024-25", "2025-26"],
    2026: ["2026", "2025-26", "2026-27"],
}

# Measured 8 Nov 2026: never returned a card for any of 106 countries, and the
# archive's season index does not publish this spelling at all.
PRUNED_TOKENS = ["2024-2025", "2025-2026", "2026-2027"]


def test_token_set_is_pinned_for_both_years():
    assert _season_tokens(2025) == YIELDING_TOKENS[2025]
    assert _season_tokens(2026) == YIELDING_TOKENS[2026]


def test_tokens_by_year_matches_the_pinned_set():
    assert TOKENS_BY_YEAR == YIELDING_TOKENS


def test_pruned_tokens_are_gone():
    for year, tokens in YIELDING_TOKENS.items():
        for pruned in PRUNED_TOKENS:
            assert pruned not in _season_tokens(year), (
                f"{pruned} was measured dead and pruned; asking for it again "
                f"costs a guaranteed 404 on every country"
            )


def test_kept_tokens_include_every_yielding_spelling():
    # The dangerous direction: a token that yields for even one country must
    # stay in the set. 2026-27 is the one most likely to be pruned by mistake.
    assert "2026-27" in _season_tokens(2026)
    assert "2024-25" in _season_tokens(2025)


def test_split_season_is_shared_by_adjacent_years():
    # 2025-26 is both the "next" season of 2025 and the "previous" season of
    # 2026, so it must be requested for both. Dropping it from either year would
    # leave that year's split season half-read.
    assert "2025-26" in _season_tokens(2025)
    assert "2025-26" in _season_tokens(2026)


def test_no_duplicate_tokens_within_a_year():
    for year in (2025, 2026):
        tokens = _season_tokens(year)
        assert len(tokens) == len(set(tokens)), (year, tokens)


def test_token_count_shrank_but_did_not_collapse():
    # 5 -> 3 for the covered years. If this ever drops to 1, discovery has been
    # narrowed to a single spelling and entire seasons will be missed.
    for year in (2025, 2026):
        assert len(_season_tokens(year)) == 3, _season_tokens(year)


def test_unmeasured_year_still_gets_the_full_spread():
    # The prune was only measured for 2025 and 2026. Other years must keep every
    # spelling, including the full-year form, until someone measures them.
    older = _season_tokens(2020)
    assert len(older) == 5, older
    assert "2019-2020" in older
    assert "2020-2021" in older


def test_tokens_are_url_safe_and_match_the_list_url_shape():
    for year in (2025, 2026):
        for token in _season_tokens(year):
            url = _match_list_url("IND", token, women=False)
            assert url.endswith(f"/IND/{token}_tt_Match_List.html"), url
            assert " " not in url


def test_discovery_still_probes_every_remaining_token():
    # Guard against someone "optimising" the loop into stopping at the first hit.
    from tests.test_ca_topup_empty_warning import FakeFetcher
    from scripts.ca_topup import discover

    fetcher = FakeFetcher({("IND", 2026, False): 2})
    discover(["IND"], [2026], False, fetcher)
    probed = {u.rsplit("/", 1)[1].split("_")[0] for u in fetcher.requests}
    assert probed == set(YIELDING_TOKENS[2026]), probed


def test_token_set_is_far_smaller_than_the_old_five_spread():
    # Documents the payoff: 3 requests instead of 5 per (country, year, gender).
    for year in (2025, 2026):
        assert len(_season_tokens(year)) < 5


def test_country_list_is_untouched_by_the_prune():
    # A token prune must not change which countries are probed.
    assert "ZAMBIA" in ALL_COUNTRIES
