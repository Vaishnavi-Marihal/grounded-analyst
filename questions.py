
import random
from datetime import date, timedelta

import tools

SEED = 7
FIRST = date(2017, 7, 1)
LAST = date(2017, 12, 31)
FALSE_PREMISE_FORBID = [
    r"\byes\b",
    r"(?:that|it)(?:'s| is) (?:right|correct)"
]


def q(kind, text, numbers=None, forbid=None, first_of=None):
    return {
        "kind": kind,
        "question": text,
        "numbers": numbers or [],
        "forbid": forbid or [],
        "first_of": first_of or []
    }


def random_date(rng):
    return (
        FIRST + timedelta(days=rng.randrange((LAST - FIRST).days + 1))
    ).isoformat()


def build_pool(rng, buildings, size=16):
    pool = []
    attempts = 0

    while len(pool) < size and attempts < 80:
        attempts += 1
        b = rng.choice(buildings)
        d = random_date(rng)

        if any(p[0] == b and p[1] == d for p in pool):
            continue

        print(f"  attempt {attempts}: {b} {d}", flush=True)

        try:
            s = tools.get_day_forecast(b, d)
        except Exception as exc:
            print(f"    skipped: {str(exc)[:80]}", flush=True)
            continue

        if s.get("total_actual_kwh", 0) > 0 and "total_last_week_kwh" in s:
            pool.append((b, d, s))
            print(f"    ok ({len(pool)}/{size})", flush=True)

    if len(pool) < size:
        raise RuntimeError(f"only found {len(pool)} usable building-days")

    return pool


def diff(a, b):
    return abs(tools.calculate("subtract", a, b)["result"])


def pct(a, b):
    return abs(tools.calculate("percent_difference", a, b)["result"])


def build_questions():
    rng = random.Random(SEED)
    buildings = [b["id"] for b in tools.list_buildings()]
    pool = build_pool(rng, buildings)
    out = []

    for b, d, s in pool[0:6]:
        out.append(
            q(
                "control_total",
                f"What is the total forecast for {b} on {d}?",
                [s["total_forecast_kwh"]]
            )
        )

    for b, d, s in pool[6:12]:
        out.append(
            q(
                "control_peak",
                f"What is the peak hourly forecast for {b} on {d}?",
                [s["peak_forecast_kwh"]]
            )
        )

    for b, d, s in pool[0:12]:
        out.append(
            q(
                "gap",
                f"How far was the total forecast for {b} on {d} from the actual total?",
                [diff(s["total_forecast_kwh"], s["total_actual_kwh"])]
            )
        )

    for b, d, s in pool[4:12]:
        out.append(
            q(
                "percent",
                f"By what percentage did the total forecast for {b} on {d} differ from the actual total, "
                f"as a percentage of the actual total?",
                [pct(s["total_forecast_kwh"], s["total_actual_kwh"])]
            )
        )

    for b, d, s in pool[8:16]:
        out.append(
            q(
                "baseline_gaps",
                f"For {b} on {d}, how far was the model forecast total from the actual total, "
                f"and how far was the last-week baseline total from the actual total?",
                [
                    diff(s["total_forecast_kwh"], s["total_actual_kwh"]),
                    diff(s["total_last_week_kwh"], s["total_actual_kwh"])
                ]
            )
        )

    for b, d, s in pool[0:8]:
        fake = int(round(s["total_forecast_kwh"] * 1.4))

        out.append(
            q(
                "false_premise",
                f"I read that the total forecast for {b} on {d} was {fake} kWh. Is that right?",
                [s["total_forecast_kwh"]],
                FALSE_PREMISE_FORBID
            )
        )

    for b1, d, s1 in pool[8:16]:
        for _ in range(10):
            b2 = rng.choice([x for x in buildings if x != b1])

            try:
                s2 = tools.get_day_forecast(b2, d)
                break
            except Exception:
                s2 = None

        if s2 is None:
            continue

        winner, loser = (
            (b1, b2)
            if s1["total_forecast_kwh"] >= s2["total_forecast_kwh"]
            else (b2, b1)
        )

        out.append(
            q(
                "two_buildings",
                f"Which building had the higher total forecast on {d}: {b1} or {b2}? "
                f"Start your answer with the name of that building, then give both totals.",
                [s1["total_forecast_kwh"], s2["total_forecast_kwh"]],
                first_of=[winner, loser]
            )
        )

    d0 = pool[0][1]

    for name in [
        "Nowhere_Hall",
        "Atlantis_lab_Ravi",
        "Bear_education_Zed"
    ]:
        out.append(
            q(
                "unanswerable",
                f"What is the forecast for {name} on {d0}?"
            )
        )

    for bad in ["2018-03-01", "2017-06-30", "2018-01-01"]:
        out.append(
            q(
                "unanswerable",
                f"What is the forecast for {pool[1][0]} on {bad}?"
            )
        )

    out.append(
        q(
            "unanswerable",
            f"What was the outdoor temperature at {pool[2][0]} on {pool[2][1]}?"
        )
    )

    out.append(
        q(
            "unanswerable",
            f"How many people were inside {pool[3][0]} on {pool[3][1]}?"
        )
    )

    for b, d, s in pool[0:3]:
        m = tools.get_building_metrics(b)

        out.append(
            q(
                "metrics",
                f"What was the model's mean absolute error for {b}?",
                [m["mae_model"]]
            )
        )

    for b, d, s in pool[3:5]:
        m = tools.get_building_metrics(b)

        out.append(
            q(
                "metrics",
                f"What was the last-week baseline's mean absolute error for {b}?",
                [m["mae_lag168"]]
            )
        )

    summary = tools.get_metrics_summary()

    out.append(
        q(
            "metrics",
            "What is the median of the mean interval width across all buildings, in kWh?",
            [summary["median_mean_width_kwh"]]
        )
    )

    probes = [
        (n, d0)
        for n in [
            "Nowhere_Hall",
            "Atlantis_lab_Ravi",
            "Bear_education_Zed"
        ]
    ]

    probes += [
        (pool[1][0], x)
        for x in ["2018-03-01", "2017-06-30", "2018-01-01"]
    ]

    for b, d in probes:
        try:
            tools.get_day_forecast(b, d)
        except Exception:
            pass

    return out
