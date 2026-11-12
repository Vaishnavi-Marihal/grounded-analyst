from checker import collect_numbers, extract_numbers, find_ungrounded

TOOL = [412.6, 380.1, 449.3]


def test_commas_do_not_split_numbers():
    assert extract_numbers("Peak is 1,200 kWh, up 12%") == ["1200", "12"]


def test_rounded_answer_is_grounded():
    assert find_ungrounded("Forecast is 413 kWh, interval 380 to 449.", TOOL) == []


def test_exact_answer_is_grounded():
    assert find_ungrounded("Forecast is 412.6 kWh.", TOOL) == []


def test_invented_percentage_is_flagged():
    assert find_ungrounded("Forecast is 413 kWh, 12% above last week.", TOOL) == ["12"]


def test_comma_number_is_flagged():
    assert find_ungrounded("Peak load is 1,200 kWh.", TOOL) == ["1200"]


def test_close_but_wrong_is_flagged():
    assert find_ungrounded("Forecast is 414 kWh.", TOOL) == ["414"]


def test_dates_are_ignored():
    assert find_ungrounded("Flagged on Dec 25.", TOOL) == []

def test_iso_dates_and_times_are_ignored():
    assert find_ungrounded("On 2017-12-25 the peak was at 10:00.", TOOL) == []

def test_error_text_numbers_are_collected():
    nums = collect_numbers({"error": "422 error: date must be between 2017-07-01 and 2017-12-31"})
    assert nums == [422.0]