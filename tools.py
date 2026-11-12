import os

import requests

BASE_URL = os.environ.get("FORECAST_API_URL", "http://13.51.206.202:8080/api")


def _get(path, params=None):
    response = requests.get(BASE_URL + path, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def list_buildings():
    return _get("/buildings")


def get_building_metrics(building_id):
    return _get(f"/buildings/{building_id}/metrics")


def get_metrics_summary():
    return _get("/metrics/summary")


def get_day_forecast(building_id, date):
    data = _get(f"/buildings/{building_id}/forecast", {"date": date})
    hours = data["hours"]
    forecast = [h["forecast_kwh"] for h in hours]
    lower = [h["lower_kwh"] for h in hours]
    upper = [h["upper_kwh"] for h in hours]
    peak = max(hours, key=lambda h: h["forecast_kwh"])
    summary = {
        "building_id": data["building_id"],
        "date": data["date"],
        "interval_level_pct": int(round(data["interval_level"] * 100)),
        "total_forecast_kwh": round(sum(forecast), 1),
        "mean_interval_width_kwh": round(
            sum(u - l for u, l in zip(upper, lower)) / len(hours), 1
        ),
        "peak_forecast_kwh": round(peak["forecast_kwh"], 1),
        "peak_hour": peak["timestamp"][11:16],
    }
    last_week = [h["baseline_lag168_kwh"] for h in hours]
    if all(v is not None for v in last_week):
        summary["total_last_week_kwh"] = round(sum(last_week), 1)
    actual = [h["actual_kwh"] for h in hours]
    if all(v is not None for v in actual):
        summary["total_actual_kwh"] = round(sum(actual), 1)
    return summary


def calculate(operation, a, b):
    if operation == "subtract":
        value = a - b
    elif operation == "percent_difference":
        if b == 0:
            return {"error": "b must not be zero"}
        value = (a - b) / b * 100
    else:
        return {"error": f"unknown operation: {operation}"}
    return {"operation": operation, "result": round(value, 1)}