from datetime import date, datetime, timezone

from src.integrations.polymarket import espn_fixture_venues as efv


def _ev(eid, t1, t2, start, ground, city, country, cls="Twenty20", et="T20",
        league="Major League Cricket", gid="1"):
    return {"espn_id": eid, "league": league, "start": start, "name": f"{t1} v {t2}",
            "teams": [t1, t2], "ground_id": gid, "ground": ground, "city": city,
            "country": country, "class": cls, "event_type": et}


VENUES = [
    efv.VenueRow(1, "Oakland Coliseum,Oakland", "Oakland", "USA", 6),
    efv.VenueRow(2, "Grand Prairie Stadium, Dallas", "Dallas", "USA", 40),
    efv.VenueRow(3, "Sabina Park, Kingston", "Kingston", "Jamaica", 2),
    efv.VenueRow(4, "Sabina Park, Kingston, Jamaica", "Kingston", "Jamaica", 20),
    efv.VenueRow(5, "County Ground, Taunton", "Taunton", "England", 30),
    efv.VenueRow(6, "County Ground, Hove", "Hove", "England", 30),
]

EVENTS = [
    _ev("a", "Los Angeles Knight Riders", "San Francisco Unicorns", "2026-07-15T21:30Z",
        "Oakland Coliseum, Oakland", "Oakland", "United States of America", gid="10"),
    _ev("b", "MI New York", "Washington Freedom", "2026-07-16T01:30Z",
        "Oakland Coliseum, Oakland", "Oakland", "United States of America", gid="10"),
    _ev("c", "Sussex", "Somerset", "2026-07-15T17:30Z", "County Ground, Hove", "Hove", "England",
        league="Vitality Blast", gid="20"),
    _ev("d", "Sussex Women", "Somerset Women", "2026-07-15T13:00Z", "County Ground, Hove", "Hove",
        "England", cls="Women's Twenty20", league="Vitality Blast Women", gid="20"),
    _ev("e", "India Under-19s", "Sri Lanka Under-19s", "2026-07-15T04:30Z", "Galle International Stadium",
        "Galle", "Sri Lanka", cls="Youth ODI", et="ODI", league="Youth tour"),
    _ev("f", "Sussex", "Somerset", "2026-07-15T10:00Z", "County Ground, Hove", "Hove", "England",
        cls="First-class", et="Test", league="County Championship", gid="20"),
]

KO = datetime(2026, 7, 15, 21, 30, tzinfo=timezone.utc)


def test_team_score_handles_suffixes_and_aliases():
    assert efv.team_score("St. Lucia Kings", "St Lucia Kings") == 1.0
    assert efv.team_score("Sussex Sharks", "Sussex") == 0.9
    assert efv.team_score("USA", "United States of America") == 1.0
    assert efv.team_score("Biratnagar Kings", "Biratnagar Kings (NPL)") == 1.0
    assert efv.team_score("Sussex", "Surrey") == 0.0
    assert efv.team_score("Kathmandu Gurkhas", "Kathmandu Gorkhas (NPL)") == 0.9
    assert efv.team_score("Sydney Sixers", "Sydney Thunder") == 0.0
    assert efv.team_score("Durban Super Giants", "Lucknow Super Giants") == 0.0


def test_match_picks_event_in_either_orientation():
    em = efv.match_fixture("San Francisco Unicorns", "Los Angeles Knight Riders", "male", EVENTS, kickoff=KO)
    assert em.event["espn_id"] == "a"
    assert em.hours_off == 0.0


def test_gender_and_format_filters():
    men = efv.match_fixture("Sussex Sharks", "Somerset", "male", EVENTS, slug_date=date(2026, 7, 15))
    women = efv.match_fixture("Sussex", "Somerset", "female", EVENTS, slug_date=date(2026, 7, 15))
    assert men.event["espn_id"] == "c"  # not the first-class fixture between the same teams
    assert women.event["espn_id"] == "d"


def test_youth_fixture_is_never_matched():
    em = efv.match_fixture("India", "Sri Lanka", "male", EVENTS, slug_date=date(2026, 7, 15))
    assert em.event is None


def test_kickoff_outside_window_is_rejected():
    far = datetime(2026, 7, 18, 21, 30, tzinfo=timezone.utc)
    assert efv.match_fixture("Los Angeles Knight Riders", "San Francisco Unicorns", "male",
                             EVENTS, kickoff=far).event is None


def test_ground_mapping_respects_city_and_prefers_recent_duplicate():
    vid, s, how = efv.map_ground_to_venue("County Ground, Hove", "Hove", "England", VENUES)
    assert (vid, how) == (6, "name")
    vid, _, _ = efv.map_ground_to_venue("Sabina Park, Kingston, Jamaica", "Kingston", "Jamaica", VENUES)
    assert vid == 4
    vid, _, _ = efv.map_ground_to_venue("Oakland Coliseum, Oakland", "Oakland",
                                        "United States of America", VENUES)
    assert vid == 1


def test_ground_id_map_wins_over_names():
    vid, s, how = efv.map_ground_to_venue("Somewhere else", "X", "England", VENUES,
                                          ground_id="20", ground_map={"20": 5})
    assert (vid, s, how) == (5, 1.0, "ground_id_map")


def test_unknown_ground_is_no_opinion():
    vid, _, how = efv.map_ground_to_venue("Brand New Oval", "Nowhere", "England", VENUES)
    assert vid is None and how == "no_name_match"


def test_resolve_grades_high_only_when_everything_agrees():
    vm = efv.resolve("Los Angeles Knight Riders", "San Francisco Unicorns", "male", EVENTS, VENUES, kickoff=KO)
    assert vm.venue_id == 1 and vm.confidence == "high"
    late = datetime(2026, 7, 16, 9, 30, tzinfo=timezone.utc)
    vm = efv.resolve("Los Angeles Knight Riders", "San Francisco Unicorns", "male", EVENTS, VENUES, kickoff=late)
    assert vm.confidence != "high"
    vm = efv.resolve("Nobody", "Else", "male", EVENTS, VENUES, kickoff=KO)
    assert vm.venue_id is None and vm.confidence == "none"


def test_identified_fixture_at_unknown_ground_is_flagged_new_ground():
    evs = [_ev("n", "Conaree", "Nevis", "2026-07-15T21:30Z", "Conaree Sports Club, Basseterre",
               "Basseterre", "St Kitts", gid="99")]
    vm = efv.resolve("Conaree", "Nevis", "male", evs, VENUES, kickoff=KO)
    assert vm.venue_id is None and vm.method == "new_ground" and vm.confidence == "high"


def test_fetch_failure_returns_empty(monkeypatch):
    class Boom:
        def get(self, *a, **k):
            raise OSError("down")

    class Counting(Boom):
        calls = 0

        def get(self, *a, **k):
            Counting.calls += 1
            return super().get(*a, **k)

    monkeypatch.setattr(efv.time, "sleep", lambda s: None)
    monkeypatch.setattr(efv, "_MEMO", {})
    assert efv.fetch_scorepanel(date(2026, 7, 15), session=Counting(), retries=2) == []
    first = Counting.calls
    assert efv.fetch_scorepanel(date(2026, 7, 15), session=Counting(), retries=2) == []
    assert Counting.calls == first, "a failed date is not refetched inside the backoff window"
