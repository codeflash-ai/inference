from datetime import datetime, timedelta

TIMESTAMP_FORMAT = "%Y_%m_%d"


def generate_today_timestamp() -> str:
    return datetime.today().strftime(TIMESTAMP_FORMAT)


def generate_start_timestamp_for_this_week() -> str:
    today = datetime.today()
    return (today - timedelta(days=today.weekday())).strftime(TIMESTAMP_FORMAT)


def generate_start_timestamp_for_this_month() -> str:
    today = datetime.today()
    return f"{today.year}_{today.month:02d}_01"
