"""Shared date parsing — the one public place to turn stored strings into dates.

Previously each module rolled its own (`_as_date`, `_to_date`, `_opt_date`);
centralizing here removes the cross-module private imports the review flagged.
"""
from __future__ import annotations

from datetime import date, datetime

import pandas as pd


def as_date(x) -> date:
    """Parse a date-like value (str/date/datetime) to a plain date."""
    if isinstance(x, datetime):
        return x.date()
    if isinstance(x, date):
        return x
    return datetime.strptime(str(x)[:10], "%Y-%m-%d").date()


def opt_date(x) -> date | None:
    """Parse an optional date; NaN/None/empty -> None."""
    if x is None or (isinstance(x, float) and pd.isna(x)) or x == "":
        return None
    return as_date(x)
