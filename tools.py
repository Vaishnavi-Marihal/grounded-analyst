import hashlib
import json
import os
import pathlib
import time

import requests

BASE_URL = os.environ.get("FORECAST_API_URL", "http://13.51.206.202:8080/api")
CACHE_DIR = pathlib.Path(os.environ.get("FORECAST_CACHE_DIR", "cache"))


def _cache_file(path, params):
    key = json.dumps([path, sorted((params or {}).items())])
    return CACHE_DIR / (hashlib.sha1(key.encode()).hexdigest() + ".json")


def _fetch(path, params):
    last = None
    for attempt in range(3):
        try:
            response = requests.get(BASE_URL + path, params=params, timeout=30)
        except (requests.ConnectionError, requests.Timeout) as exc:
            last = exc
            time.sleep(1 + attempt)
            continue
        if response.status_code >= 500:
            last = requests.HTTPError(f"server error {response.status_code}")
            time.sleep(1 + attempt)
            continue
        return response
    raise last


def _get(path, params=None):
    mode = os.environ.get("FORECAST_MODE", "live")
    file = _cache_file(path, params)
    if mode == "replay":
        if not file.exists():
            raise KeyError(f"not in cache: {path} {params}")
        saved = json.loads(file.read_text())
    else:
        response = _fetch(path, params)
        try:
            body = response.json()
        except ValueError:
            body = {}
        saved = {"status": response.status_code, "body": body}
        if mode == "record":
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            file.write_text(json.dumps(saved))
    if saved["status"] >= 400:
        detail = saved["body"].get("detail", "") if isinstance(saved["body"], dict) else ""
        raise requests.HTTPError(f"{saved['status']} error: {detail}")
    return saved["body"]


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