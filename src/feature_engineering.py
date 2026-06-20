"""Feature engineering utilities for illegal-parking hotspot prediction.

The functions in this module convert violation-level Bangalore Traffic Police
records into hotspot-hour level rows suitable for model training or inference.
They are intentionally pandas-only so they can be reused from notebooks,
scripts, batch jobs, and lightweight API services.
"""

from __future__ import annotations

import ast
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_DROP_COLUMNS: tuple[str, ...] = (
    "description",
    "device_id",
    "created_by_id",
    "data_sent_to_scita",
    "data_sent_to_scita_timestamp",
    "modified_datetime",
    "center_code",
    "closed_datetime",
    "action_taken_timestamp",
)

VEHICLE_TYPE_TO_CLASS: dict[str, str] = {
    "MOTOR CYCLE": "Two Wheeler",
    "SCOOTER": "Two Wheeler",
    "BIKE": "Two Wheeler",
    "MOPED": "Two Wheeler",
    "CAR": "Light",
    "VAN": "Light",
    "LGV": "Light",
    "MAXI-CAB": "Light",
    "AUTO": "Light",
    "PASSENGER AUTO": "Light",
    "AUTO RICKSHAW": "Light",
    "JEEP": "Medium",
    "TEMPO": "Medium",
    "GOODS AUTO": "Medium",
    "MINI LORRY": "Medium",
    "TRACTOR": "Medium",
    "BUS": "Heavy",
    "TRUCK": "Heavy",
    "TANKER": "Heavy",
    "LORRY": "Heavy",
    "OTHERS": "Heavy",
    "PRIVATE BUS": "Heavy",
    "BUS (BMTC/KSRTC)": "Heavy",
    "HGV": "Heavy",
    "SCHOOL VEHICLE": "Heavy",
    "TOURIST BUS": "Heavy",
    "LORRY/GOODS VEHICLE": "Heavy",
    "FACTORY BUS": "Heavy",
}

VEHICLE_CLASS_WEIGHTS: dict[str, int] = {
    "Two Wheeler": 1,
    "Light": 2,
    "Medium": 4,
    "Heavy": 6,
}

WEEKDAY_ORDER: tuple[str, ...] = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)


def drop_unused_columns(
    df: pd.DataFrame,
    columns: Iterable[str] = DEFAULT_DROP_COLUMNS,
) -> pd.DataFrame:
    """Return a copy of ``df`` without noisy operational columns."""

    return df.drop(columns=[col for col in columns if col in df.columns]).copy()


def parse_list_column(value: object) -> list:
    """Parse a list-like value stored as a Python list or string literal."""

    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return [value]
        return parsed if isinstance(parsed, list) else [parsed]
    return [value]


def add_temporal_features(
    df: pd.DataFrame,
    datetime_col: str = "created_datetime",
    target_timezone: str | None = "Asia/Kolkata",
) -> pd.DataFrame:
    """Parse timestamps and add calendar, hour, weekday, and slot features."""

    result = df.copy()
    created_at = pd.to_datetime(result[datetime_col], utc=True, errors="coerce")

    if target_timezone:
        created_at = created_at.dt.tz_convert(target_timezone)

    result[datetime_col] = created_at
    result["year"] = created_at.dt.year
    result["month"] = created_at.dt.month
    result["day"] = created_at.dt.day
    result["date"] = created_at.dt.date
    result["hour"] = created_at.dt.hour
    result["minute"] = created_at.dt.minute
    result["weekday"] = created_at.dt.day_name()
    result["weekday_num"] = created_at.dt.weekday
    result["is_weekend"] = result["weekday"].isin(["Saturday", "Sunday"])
    result["time_slot"] = result["hour"].map(get_time_slot)
    result["is_peak_commute_hour"] = result["hour"].map(is_peak_commute_hour)

    return result


def get_time_slot(hour: int | float) -> str:
    """Map an hour of day to a traffic-friendly time slot."""

    if pd.isna(hour):
        return "Unknown"

    hour = int(hour)
    if 6 <= hour < 10:
        return "Morning Peak"
    if 10 <= hour < 16:
        return "Midday"
    if 16 <= hour < 20:
        return "Evening Peak"
    if 20 <= hour < 24:
        return "Night"
    return "Late Night"


def is_peak_commute_hour(hour: int | float) -> bool:
    """Return True for common Bangalore morning/evening commute windows."""

    if pd.isna(hour):
        return False

    hour = int(hour)
    return 8 <= hour <= 10 or 17 <= hour <= 20


def add_vehicle_features(
    df: pd.DataFrame,
    vehicle_type_col: str = "vehicle_type",
    updated_vehicle_type_col: str = "updated_vehicle_type",
) -> pd.DataFrame:
    """Normalize vehicle types into broad classes and numeric weights."""

    result = df.copy()

    if updated_vehicle_type_col in result.columns:
        raw_type = result[updated_vehicle_type_col].fillna(result[vehicle_type_col])
    else:
        raw_type = result[vehicle_type_col]

    normalized_type = raw_type.astype("string").str.strip().str.upper()
    result["vehicle_class"] = normalized_type.map(VEHICLE_TYPE_TO_CLASS).fillna(
        normalized_type
    )
    result["vehicle_weight"] = result["vehicle_class"].map(VEHICLE_CLASS_WEIGHTS)

    return result


def add_offence_features(
    df: pd.DataFrame,
    offence_code_col: str = "offence_code",
    violation_type_col: str = "violation_type",
) -> pd.DataFrame:
    """Parse offence and violation list columns and add compact count features."""

    result = df.copy()

    if offence_code_col in result.columns:
        result[offence_code_col] = result[offence_code_col].apply(parse_list_column)
        result["first_offence"] = result[offence_code_col].apply(
            lambda values: values[0] if values else None
        )
        result["offence_count"] = result[offence_code_col].str.len()

    if violation_type_col in result.columns:
        result[violation_type_col] = result[violation_type_col].apply(parse_list_column)
        result["violation_type_count"] = result[violation_type_col].str.len()

    return result


def preprocess_violations(
    df: pd.DataFrame,
    datetime_col: str = "created_datetime",
    hotspot_col: str = "location",
    target_timezone: str | None = "Asia/Kolkata",
    drop_columns: Iterable[str] = DEFAULT_DROP_COLUMNS,
) -> pd.DataFrame:
    """Clean violation-level data and add reusable row-level features."""

    result = drop_unused_columns(df, drop_columns)
    _require_columns(result, [datetime_col, hotspot_col])

    result = add_temporal_features(result, datetime_col, target_timezone)

    if "vehicle_type" in result.columns:
        result = add_vehicle_features(result)

    result = add_offence_features(result)

    for col in ("latitude", "longitude"):
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")

    for col in ("police_station", "junction_name"):
        if col in result.columns:
            result[col] = result[col].fillna("Unknown")

    result = result.dropna(subset=[datetime_col, hotspot_col]).copy()
    result[hotspot_col] = result[hotspot_col].astype("string").str.strip()
    result = result[result[hotspot_col].notna() & (result[hotspot_col] != "")]

    return result


def build_hotspot_hour_features(
    df: pd.DataFrame,
    hotspot_col: str = "location",
    datetime_col: str = "created_datetime",
    freq: str = "h",
    target_timezone: str | None = "Asia/Kolkata",
    already_preprocessed: bool = False,
) -> pd.DataFrame:
    """Aggregate violation-level records into one row per hotspot-hour window.

    Parameters
    ----------
    df:
        Violation-level records.
    hotspot_col:
        Column identifying a hotspot. Defaults to ``location`` because this is
        the stable road/address-level grouping used in the exploratory notebook.
    datetime_col:
        Timestamp column used to build hourly windows.
    freq:
        Pandas offset alias for the output time window. Keep as ``"h"`` for
        one-hour hotspot windows.
    target_timezone:
        Timezone used for traffic-hour features. ``"Asia/Kolkata"`` aligns the
        feature values with Bangalore local time.
    already_preprocessed:
        Set to True when ``df`` already came from ``preprocess_violations``.
    """

    violations = (
        df.copy()
        if already_preprocessed
        else preprocess_violations(df, datetime_col, hotspot_col, target_timezone)
    )
    _require_columns(violations, [hotspot_col, datetime_col])

    violations["hour_window_start"] = violations[datetime_col].dt.floor(freq)
    violations["hotspot_id"] = violations[hotspot_col]

    group_cols = ["hotspot_id", "hour_window_start"]
    optional_aggs = _available_numeric_aggregations(violations)

    features = (
        violations.groupby(group_cols, observed=True)
        .agg(
            violation_count=(datetime_col, "size"),
            **optional_aggs,
        )
        .reset_index()
    )

    dominant_columns = {
        "police_station": "dominant_police_station",
        "junction_name": "dominant_junction",
        "vehicle_class": "dominant_vehicle_class",
    }
    for source_col, output_col in dominant_columns.items():
        if source_col in violations.columns:
            dominant_values = _dominant_values(
                violations, group_cols, source_col, output_col
            )
            features = features.merge(dominant_values, on=group_cols, how="left")

    features = _add_window_temporal_features(features)
    features = _add_peak_hour_features(features)
    features = _add_hotspot_context(features)

    ordered_cols = [
        "hotspot_id",
        "hour_window_start",
        "violation_count",
        "latitude",
        "longitude",
        "avg_vehicle_weight",
        "dominant_police_station",
        "dominant_junction",
        "has_junction",
        "peak_hour_of_day",
        "is_hotspot_peak_hour",
        "is_peak_commute_hour",
        "time_slot",
        "year",
        "month",
        "day",
        "date",
        "hour",
        "weekday",
        "weekday_num",
        "is_weekend",
        "hotspot_total_violations",
        "hotspot_hour_share",
    ]
    ordered_cols = [col for col in ordered_cols if col in features.columns]
    extra_cols = [col for col in features.columns if col not in ordered_cols]

    return features[ordered_cols + extra_cols].sort_values(
        ["hotspot_id", "hour_window_start"]
    )


def _available_numeric_aggregations(df: pd.DataFrame) -> dict[str, tuple[str, object]]:
    aggregations: dict[str, tuple[str, object]] = {}

    if "latitude" in df.columns:
        aggregations["latitude"] = ("latitude", "mean")
    if "longitude" in df.columns:
        aggregations["longitude"] = ("longitude", "mean")
    if "vehicle_weight" in df.columns:
        aggregations["avg_vehicle_weight"] = ("vehicle_weight", "mean")
        aggregations["max_vehicle_weight"] = ("vehicle_weight", "max")
    if "offence_count" in df.columns:
        aggregations["avg_offence_count"] = ("offence_count", "mean")
    if "violation_type_count" in df.columns:
        aggregations["avg_violation_type_count"] = ("violation_type_count", "mean")

    return aggregations


def _dominant_values(
    df: pd.DataFrame,
    group_cols: list[str],
    source_col: str,
    output_col: str,
) -> pd.DataFrame:
    counts = (
        df[group_cols + [source_col]]
        .dropna(subset=[source_col])
        .groupby(group_cols + [source_col], observed=True)
        .size()
        .rename("_count")
        .reset_index()
    )

    if counts.empty:
        return df[group_cols].drop_duplicates().assign(**{output_col: "Unknown"})

    counts[source_col] = counts[source_col].astype("string")
    ascending = [True] * len(group_cols) + [False, True]
    dominant = (
        counts.sort_values(group_cols + ["_count", source_col], ascending=ascending)
        .drop_duplicates(group_cols)
        .drop(columns="_count")
        .rename(columns={source_col: output_col})
    )

    return dominant


def mode_or_unknown(values: pd.Series) -> object:
    """Return the most common non-null value, using deterministic tie-breaking."""

    counts = values.dropna().astype("string").value_counts()
    if counts.empty:
        return "Unknown"
    return counts.sort_values(ascending=False).index[0]


def _add_window_temporal_features(features: pd.DataFrame) -> pd.DataFrame:
    result = features.copy()
    window = result["hour_window_start"]

    result["year"] = window.dt.year
    result["month"] = window.dt.month
    result["day"] = window.dt.day
    result["date"] = window.dt.date
    result["hour"] = window.dt.hour
    result["weekday"] = window.dt.day_name()
    result["weekday_num"] = window.dt.weekday
    result["is_weekend"] = result["weekday"].isin(["Saturday", "Sunday"])
    result["time_slot"] = result["hour"].map(get_time_slot)
    result["is_peak_commute_hour"] = result["hour"].map(is_peak_commute_hour)

    if "dominant_junction" in result.columns:
        result["has_junction"] = (
            result["dominant_junction"].fillna("Unknown").ne("No Junction")
            & result["dominant_junction"].fillna("Unknown").ne("Unknown")
        ).astype(int)

    return result


def _add_peak_hour_features(features: pd.DataFrame) -> pd.DataFrame:
    result = features.copy()

    hourly_totals = (
        result.groupby(["hotspot_id", "hour"], observed=True)["violation_count"]
        .sum()
        .reset_index()
        .sort_values(["hotspot_id", "violation_count", "hour"], ascending=[True, False, True])
    )
    peak_hours = hourly_totals.drop_duplicates("hotspot_id").rename(
        columns={"hour": "peak_hour_of_day"}
    )[["hotspot_id", "peak_hour_of_day"]]

    result = result.merge(peak_hours, on="hotspot_id", how="left")
    result["is_hotspot_peak_hour"] = result["hour"].eq(result["peak_hour_of_day"])

    return result


def _add_hotspot_context(features: pd.DataFrame) -> pd.DataFrame:
    result = features.copy()

    totals = (
        result.groupby("hotspot_id", observed=True)["violation_count"]
        .sum()
        .rename("hotspot_total_violations")
    )
    result = result.merge(totals, on="hotspot_id", how="left")
    result["hotspot_hour_share"] = np.where(
        result["hotspot_total_violations"] > 0,
        result["violation_count"] / result["hotspot_total_violations"],
        0.0,
    )

    return result


def _require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"Missing required columns: {missing_text}")
