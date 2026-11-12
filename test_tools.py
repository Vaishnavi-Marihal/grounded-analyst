import checker
import tools
import pytest

SAMPLE = {
    "building_id": "B",
    "date": "2017-12-25",
    "interval_level": 0.9,
    "hours": [
        {"timestamp": "2017-12-25T00:00:00", "forecast_kwh": 10.0, "lower_kwh": 8.0,
         "upper_kwh": 14.0, "baseline_lag168_kwh": 12.0, "actual_kwh": 11.0},
        {"timestamp": "2017-12-25T01:00:00", "forecast_kwh": 20.0, "lower_kwh": 18.0,
         "upper_kwh": 22.0, "baseline_lag168_kwh": 19.0, "actual_kwh": 21.0},
        {"timestamp": "2017-12-25T02:00:00", "forecast_kwh": 30.0, "lower_kwh": 26.0,
         "upper_kwh": 32.0, "baseline_lag168_kwh": 28.0, "actual_kwh": None},
    ],
}


def fake_get(path, params=None):
    return SAMPLE


def test_day_summary_totals(monkeypatch):
    monkeypatch.setattr(tools, "_get", fake_get)
    s = tools.get_day_forecast("B", "2017-12-25")
    assert s["total_forecast_kwh"] == 60.0
    assert s["total_last_week_kwh"] == 59.0
    assert s["interval_level_pct"] == 90


def test_day_summary_peak_and_width(monkeypatch):
    monkeypatch.setattr(tools, "_get", fake_get)
    s = tools.get_day_forecast("B", "2017-12-25")
    assert s["peak_forecast_kwh"] == 30.0
    assert s["peak_hour"] == "02:00"
    assert s["mean_interval_width_kwh"] == 5.3


def test_missing_actual_gives_no_total(monkeypatch):
    monkeypatch.setattr(tools, "_get", fake_get)
    s = tools.get_day_forecast("B", "2017-12-25")
    assert "total_actual_kwh" not in s


def test_summary_numbers_feed_the_checker(monkeypatch):
    monkeypatch.setattr(tools, "_get", fake_get)
    s = tools.get_day_forecast("B", "2017-12-25")
    numbers = checker.collect_numbers(s)
    assert checker.find_ungrounded("Total forecast is 60 kWh.", numbers) == []
    assert checker.find_ungrounded("Total forecast is 75 kWh.", numbers) == ["75"]


def test_calculate_subtract_and_percent():
    assert tools.calculate("subtract", 2168.0, 2088.9)["result"] == 79.1
    assert tools.calculate("percent_difference", 2168.0, 2088.9)["result"] == 3.8


def test_calculate_rejects_zero_and_unknown():
    assert "error" in tools.calculate("percent_difference", 5.0, 0)
    assert "error" in tools.calculate("power", 2.0, 3.0)

def test_get_retries_after_connection_error(monkeypatch):
    calls = {"n": 0}

    class Resp:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"ok": 1}

    def flaky(url, params=None, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise tools.requests.ConnectionError("boom")
        return Resp()

    monkeypatch.setattr(tools.requests, "get", flaky)
    monkeypatch.setattr(tools.time, "sleep", lambda s: None)
    assert tools._get("/x") == {"ok": 1}
    assert calls["n"] == 2

class FakeResponse:
    def __init__(self, status, body):
        self.status_code = status
        self._body = body

    def json(self):
        return self._body


def test_record_then_replay(monkeypatch, tmp_path):
    monkeypatch.setattr(tools, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(tools.requests, "get",
                        lambda url, params=None, timeout=None: FakeResponse(200, {"v": 7}))
    monkeypatch.setenv("FORECAST_MODE", "record")
    assert tools._get("/a", {"d": "x"}) == {"v": 7}

    def no_network(*args, **kwargs):
        raise AssertionError("network used")

    monkeypatch.setattr(tools.requests, "get", no_network)
    monkeypatch.setenv("FORECAST_MODE", "replay")
    assert tools._get("/a", {"d": "x"}) == {"v": 7}


def test_error_status_is_recorded_and_replayed(monkeypatch, tmp_path):
    monkeypatch.setattr(tools, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(tools.requests, "get",
                        lambda url, params=None, timeout=None: FakeResponse(422, {"detail": "bad date"}))
    monkeypatch.setenv("FORECAST_MODE", "record")
    with pytest.raises(tools.requests.HTTPError):
        tools._get("/b")
    monkeypatch.setenv("FORECAST_MODE", "replay")
    with pytest.raises(tools.requests.HTTPError, match="422"):
        tools._get("/b")