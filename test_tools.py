import checker
import tools

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
    
    assert s["total_lag168_kwh"] == 59.0
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