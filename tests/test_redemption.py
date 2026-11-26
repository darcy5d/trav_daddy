from src.integrations.polymarket import redemption


class FakeClient:
    def get_data_api_positions(self, **_kwargs):
        return [
            {
                "asset": "winner-token",
                "conditionId": "0x" + "a" * 64,
                "redeemable": True,
                "currentValue": 12.5,
                "size": "12.5",
                "outcome": "Yes",
                "negativeRisk": False,
                "title": "Winner",
            },
            {
                "asset": "pending-token",
                "conditionId": "0x" + "b" * 64,
                "redeemable": False,
                "currentValue": 20,
                "size": "20",
            },
            {
                "asset": "missing-condition",
                "redeemable": True,
                "currentValue": 20,
            },
        ]


def test_discovery_only_returns_api_confirmed_redeemable_positions(monkeypatch):
    monkeypatch.setitem(redemption.BETTING_CONFIG, "auto_redeem_min_value_usdc", 0.01)

    candidates = redemption.discover_redeemable_positions(FakeClient())

    assert len(candidates) == 1
    assert candidates[0]["token_id"] == "winner-token"
    assert candidates[0]["condition_id"] == "0x" + "a" * 64
    assert candidates[0]["yes_amount"] == "12.5"
    assert candidates[0]["no_amount"] == "0"


def test_discovery_deduplicates_by_condition(monkeypatch):
    monkeypatch.setitem(redemption.BETTING_CONFIG, "auto_redeem_min_value_usdc", 0.01)

    class DuplicateClient:
        def get_data_api_positions(self, **_kwargs):
            return [
                {"asset": "small", "conditionId": "0x" + "c" * 64, "redeemable": True, "currentValue": 1},
                {"asset": "large", "conditionId": "0x" + "c" * 64, "redeemable": True, "currentValue": 2},
            ]

    candidates = redemption.discover_redeemable_positions(DuplicateClient())

    assert len(candidates) == 1
    assert candidates[0]["token_id"] == "large"
