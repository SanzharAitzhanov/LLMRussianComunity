from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from googleapiclient.errors import HttpError
from tqdm import tqdm

from config import ProjectConfig, load_config
from src.utils import (
    append_errors,
    append_jsonl,
    build_youtube_client,
    ensure_project_dirs,
    execute_with_backoff,
    explicit_match_in_metadata,
    is_fatal_quota_error,
    parse_http_error,
    read_csv_if_exists,
    save_table,
    setup_logger,
    utc_now,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search YouTube videos by Russian Community name variants.")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--date-from", default=None)
    parser.add_argument("--date-to", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-results-per-interval", type=int, default=None)
    parser.add_argument("--max-results-per-query", type=int, default=None)
    parser.add_argument("--orders", default=None, help='Comma-separated orders, e.g. "date,relevance".')
    parser.add_argument("--no-saturation", action="store_true")
    parser.add_argument("--max-interval-level", choices=["year", "quarter", "month", "week"], default=None)
    parser.add_argument("--force-intervals", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


@dataclass(frozen=True)
class SearchInterval:
    label: str
    level: str
    published_after: str
    published_before: str


LEVEL_ORDER = {"year": 0, "quarter": 1, "month": 2, "week": 3}
SEARCH_LIST_QUOTA_COST = 100


def parse_youtube_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def format_youtube_time(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def yearly_intervals(date_from: str, date_to: str) -> list[SearchInterval]:
    start = parse_youtube_time(date_from)
    end = parse_youtube_time(date_to)
    intervals: list[SearchInterval] = []
    for year in range(start.year, end.year + 1):
        after = max(start, datetime(year, 1, 1, tzinfo=timezone.utc))
        before = min(end, datetime(year + 1, 1, 1, tzinfo=timezone.utc))
        if after < before:
            intervals.append(SearchInterval(str(year), "year", format_youtube_time(after), format_youtube_time(before)))
    return intervals


def quarter_intervals(parent: SearchInterval) -> list[SearchInterval]:
    start = parse_youtube_time(parent.published_after)
    end = parse_youtube_time(parent.published_before)
    year = start.year
    quarters = [(1, 4, "Q1"), (4, 7, "Q2"), (7, 10, "Q3"), (10, 13, "Q4")]
    children: list[SearchInterval] = []
    for start_month, end_month, suffix in quarters:
        q_start = datetime(year, start_month, 1, tzinfo=timezone.utc)
        q_end = datetime(year + 1, 1, 1, tzinfo=timezone.utc) if end_month == 13 else datetime(year, end_month, 1, tzinfo=timezone.utc)
        after = max(start, q_start)
        before = min(end, q_end)
        if after < before:
            children.append(SearchInterval(f"{year}-{suffix}", "quarter", format_youtube_time(after), format_youtube_time(before)))
    return children


def month_intervals(parent: SearchInterval) -> list[SearchInterval]:
    start = parse_youtube_time(parent.published_after)
    end = parse_youtube_time(parent.published_before)
    current = datetime(start.year, start.month, 1, tzinfo=timezone.utc)
    children: list[SearchInterval] = []
    while current < end:
        next_year = current.year + 1 if current.month == 12 else current.year
        next_month = 1 if current.month == 12 else current.month + 1
        next_start = datetime(next_year, next_month, 1, tzinfo=timezone.utc)
        after = max(start, current)
        before = min(end, next_start)
        if after < before:
            children.append(SearchInterval(f"{current.year}-{current.month:02d}", "month", format_youtube_time(after), format_youtube_time(before)))
        current = next_start
    return children


def week_intervals(parent: SearchInterval) -> list[SearchInterval]:
    start = parse_youtube_time(parent.published_after)
    end = parse_youtube_time(parent.published_before)
    current = start
    children: list[SearchInterval] = []
    week_index = 1
    while current < end:
        next_start = min(current + timedelta(days=7), end)
        children.append(
            SearchInterval(
                f"{start.year}-{start.month:02d}-W{week_index}",
                "week",
                format_youtube_time(current),
                format_youtube_time(next_start),
            )
        )
        current = next_start
        week_index += 1
    return children


def child_intervals(interval: SearchInterval) -> list[SearchInterval]:
    if interval.level == "year":
        return quarter_intervals(interval)
    if interval.level == "quarter":
        return month_intervals(interval)
    if interval.level == "month":
        return week_intervals(interval)
    return []


def can_split(interval: SearchInterval, max_interval_level: str) -> bool:
    return LEVEL_ORDER[interval.level] < LEVEL_ORDER[max_interval_level]


def load_completed_interval_keys(config: ProjectConfig) -> set[tuple[str, str, str, str]]:
    log = read_csv_if_exists(config.processed_dir / "search_interval_log.csv")
    if log.empty:
        return set()
    needed = {"query", "order", "interval_level", "interval_label", "status"}
    if not needed.issubset(log.columns):
        return set()
    done = log[log["status"].isin(["complete", "split_after_probe", "saturated_terminal", "error", "stopped_low_yield"])]
    return {
        (str(row.query), str(row.order), str(row.interval_level), str(row.interval_label))
        for row in done.itertuples(index=False)
    }


def save_interval_status(config: ProjectConfig, status: dict[str, Any]) -> None:
    path = config.processed_dir / "search_interval_log.csv"
    existing = read_csv_if_exists(path)
    new_df = pd.DataFrame([status])
    combined = new_df if existing.empty else pd.concat([existing, new_df], ignore_index=True)
    combined = combined.drop_duplicates(["query", "order", "interval_level", "interval_label"], keep="last")
    save_table(combined, path, config.processed_dir / "search_interval_log.parquet")


def save_iteration_metrics(config: ProjectConfig, metrics: dict[str, Any]) -> None:
    path = config.processed_dir / "search_iteration_metrics.csv"
    existing = read_csv_if_exists(path)
    new_df = pd.DataFrame([metrics])
    combined = new_df if existing.empty else pd.concat([existing, new_df], ignore_index=True)
    save_table(combined, path, config.processed_dir / "search_iteration_metrics.parquet")


def save_query_effectiveness_summary(config: ProjectConfig) -> None:
    metrics = read_csv_if_exists(config.processed_dir / "search_iteration_metrics.csv")
    matches = read_csv_if_exists(config.processed_dir / "video_search_matches.csv")
    if metrics.empty and matches.empty:
        return
    if metrics.empty:
        matches["year"] = pd.to_datetime(matches["published_at"], errors="coerce", utc=True).dt.year
        output = (
            matches.groupby(["query_used", "search_order", "year"], dropna=False)
            .agg(returned_rows=("video_id", "size"), unique_videos=("video_id", "nunique"))
            .reset_index()
            .rename(columns={"query_used": "query", "search_order": "order"})
        )
        output["new_unique_video_count"] = ""
        output["new_relevant_video_count"] = ""
        output["new_video_rate"] = ""
        output["new_unique_rate"] = ""
        output["new_relevant_rate"] = ""
        output["search_calls"] = ""
        output["quota_cost_estimate"] = ""
        output["intervals_observed"] = ""
        output = output[
            [
                "query",
                "order",
                "year",
                "returned_rows",
                "unique_videos",
                "new_unique_video_count",
                "new_relevant_video_count",
                "new_video_rate",
                "new_unique_rate",
                "new_relevant_rate",
                "search_calls",
                "quota_cost_estimate",
                "intervals_observed",
            ]
        ].sort_values(["query", "order", "year"])
        save_table(output, config.processed_dir / "query_effectiveness.csv", config.processed_dir / "query_effectiveness.parquet")
        return
    metrics["year"] = pd.to_datetime(metrics["published_after"], errors="coerce", utc=True).dt.year
    for column in [
        "rows_collected",
        "interval_unique_video_count",
        "new_unique_video_count",
        "new_relevant_video_count",
        "search_calls",
    ]:
        metrics[column] = pd.to_numeric(metrics.get(column, 0), errors="coerce").fillna(0)

    grouped = metrics.groupby(["query", "order", "year"], dropna=False).agg(
        returned_rows=("rows_collected", "sum"),
        interval_unique_video_count=("interval_unique_video_count", "sum"),
        new_unique_video_count=("new_unique_video_count", "sum"),
        new_relevant_video_count=("new_relevant_video_count", "sum"),
        search_calls=("search_calls", "sum"),
        intervals_observed=("interval_label", "count"),
    ).reset_index()
    grouped["new_video_rate"] = grouped.apply(
        lambda row: row["new_unique_video_count"] / row["returned_rows"] if row["returned_rows"] else "",
        axis=1,
    )
    grouped["new_unique_rate"] = grouped.apply(
        lambda row: row["new_unique_video_count"] / row["interval_unique_video_count"]
        if row["interval_unique_video_count"]
        else "",
        axis=1,
    )
    grouped["new_relevant_rate"] = grouped.apply(
        lambda row: row["new_relevant_video_count"] / row["new_unique_video_count"]
        if row["new_unique_video_count"]
        else "",
        axis=1,
    )
    grouped["quota_cost_estimate"] = grouped["search_calls"] * SEARCH_LIST_QUOTA_COST

    if not matches.empty:
        matches["year"] = pd.to_datetime(matches["published_at"], errors="coerce", utc=True).dt.year
        exact_unique = (
            matches.groupby(["query_used", "search_order", "year"], dropna=False)["video_id"]
            .nunique()
            .reset_index(name="unique_videos")
            .rename(columns={"query_used": "query", "search_order": "order"})
        )
        grouped = grouped.merge(exact_unique, on=["query", "order", "year"], how="left")
    else:
        grouped["unique_videos"] = 0
    grouped["unique_videos"] = grouped["unique_videos"].fillna(0).astype("int64")

    output = grouped[
        [
            "query",
            "order",
            "year",
            "returned_rows",
            "unique_videos",
            "new_unique_video_count",
            "new_relevant_video_count",
            "new_video_rate",
            "new_unique_rate",
            "new_relevant_rate",
            "search_calls",
            "quota_cost_estimate",
            "intervals_observed",
        ]
    ].sort_values(["query", "order", "year"])
    save_table(output, config.processed_dir / "query_effectiveness.csv", config.processed_dir / "query_effectiveness.parquet")


def rate_as_float(value: Any) -> float:
    if value == "" or pd.isna(value):
        return 0.0
    return float(value)


def stop_state_key(query: str, order: str, interval_level: str, parent_label: str) -> tuple[str, str, str, str]:
    return query, order, interval_level, parent_label


def should_stop_branch(config: ProjectConfig, rates: list[float]) -> bool:
    if not config.enable_stop_rule or config.stop_rule_window <= 0:
        return False
    if len(rates) < config.stop_rule_window:
        return False
    window = rates[-config.stop_rule_window :]
    return sum(window) / len(window) < config.stop_rule_min_new_video_rate


def build_iteration_metrics(
    *,
    config: ProjectConfig,
    query: str,
    order: str,
    interval: SearchInterval,
    interval_rows: list[dict[str, Any]],
    seen_video_ids: set[str],
    search_calls: int,
) -> dict[str, Any]:
    interval_video_ids = {str(row["video_id"]) for row in interval_rows if row.get("video_id")}
    new_video_ids = interval_video_ids - seen_video_ids
    relevant_new_video_ids = {
        str(row["video_id"])
        for row in interval_rows
        if row.get("video_id") in new_video_ids
        and explicit_match_in_metadata(
            title=str(row.get("title", "")),
            description=str(row.get("description", "")),
            tags="",
            phrases=config.queries,
        )
    }
    interval_unique_count = len(interval_video_ids)
    new_unique_count = len(new_video_ids)
    rows_collected = len(interval_rows)
    return {
        "query": query,
        "order": order,
        "interval_level": interval.level,
        "interval_label": interval.label,
        "published_after": interval.published_after,
        "published_before": interval.published_before,
        "rows_collected": rows_collected,
        "interval_unique_video_count": interval_unique_count,
        "new_unique_video_count": new_unique_count,
        "new_relevant_video_count": len(relevant_new_video_ids),
        "search_calls": search_calls,
        "quota_cost_estimate": search_calls * SEARCH_LIST_QUOTA_COST,
        "new_video_rate": new_unique_count / rows_collected if rows_collected else "",
        "new_unique_rate": new_unique_count / interval_unique_count if interval_unique_count else "",
        "new_relevant_rate": len(relevant_new_video_ids) / new_unique_count if new_unique_count else "",
        "seen_video_count_before": len(seen_video_ids),
        "seen_video_count_after": len(seen_video_ids | interval_video_ids),
        "collected_at": utc_now(),
    }


def rows_from_payload(
    *,
    payload: dict[str, Any],
    query: str,
    order: str,
    interval: SearchInterval,
    page_index: int,
    collected_at: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in payload.get("items", []):
        video_id = (item.get("id") or {}).get("videoId")
        snippet = item.get("snippet") or {}
        if not video_id:
            continue
        rows.append(
            {
                "video_id": video_id,
                "title": snippet.get("title", ""),
                "description": snippet.get("description", ""),
                "channel_id": snippet.get("channelId", ""),
                "channel_title": snippet.get("channelTitle", ""),
                "published_at": snippet.get("publishedAt", ""),
                "live_broadcast_content": snippet.get("liveBroadcastContent", ""),
                "query_used": query,
                "search_order": order,
                "search_interval_level": interval.level,
                "search_interval_label": interval.label,
                "published_after": interval.published_after,
                "published_before": interval.published_before,
                "page_index": page_index,
                "collected_at": collected_at,
            }
        )
    return rows


def search_interval(
    *,
    youtube: Any,
    config: ProjectConfig,
    query: str,
    order: str,
    interval: SearchInterval,
    logger: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    raw_path = config.raw_dir / "search_api_responses.jsonl"
    page_token: str | None = None
    page_index = 0
    total_estimate = 0
    first_page_results = 0
    has_next_page_at_stop = False
    search_calls = 0

    while page_index == 0 or len(rows) < config.max_results_per_interval:
        remaining = max(config.max_results_per_interval - len(rows), 1)
        max_results = min(config.max_results_per_page, remaining)
        collected_at = utc_now()
        try:
            payload = execute_with_backoff(
                lambda: youtube.search().list(
                    part="snippet",
                    q=query,
                    type="video",
                    order=order,
                    publishedAfter=interval.published_after,
                    publishedBefore=interval.published_before,
                    maxResults=max_results,
                    pageToken=page_token,
                    relevanceLanguage="ru",
                ),
                logger=logger,
                max_retries=config.max_retries,
                backoff_base_seconds=config.backoff_base_seconds,
            )
        except HttpError as exc:
            status, reason, message = parse_http_error(exc)
            append_errors(config, [{"stage": "search", "video_id": "", "query": query, "error_reason": reason, "error_message": message, "http_status": status, "collected_at": collected_at}])
            logger.error("search.list failed query=%r order=%s interval=%s: %s", query, order, interval.label, message)
            if is_fatal_quota_error(reason, message):
                status_row = interval_status(
                    query,
                    order,
                    interval,
                    "quota_exhausted",
                    len(rows),
                    first_page_results,
                    total_estimate,
                    has_next_page_at_stop,
                    False,
                    search_calls,
                    reason,
                    message,
                )
                save_interval_status(config, status_row)
                raise SystemExit(
                    "YouTube API quota/rate limit is exhausted. "
                    "Stopping now; rerun the same command after quota resets."
                )
            return rows, interval_status(query, order, interval, "error", len(rows), first_page_results, total_estimate, has_next_page_at_stop, False, search_calls, reason, message)

        append_jsonl(raw_path, [{"endpoint": "search.list", "query_used": query, "order": order, "interval_level": interval.level, "interval_label": interval.label, "page_index": page_index, "page_token": page_token, "publishedAfter": interval.published_after, "publishedBefore": interval.published_before, "collected_at": collected_at, "payload": payload}])
        search_calls += 1
        items = payload.get("items", [])
        page_info = payload.get("pageInfo") or {}
        total_estimate = int(page_info.get("totalResults") or 0)

        if page_index == 0:
            first_page_results = len(items)
            first_page_has_next = bool(payload.get("nextPageToken"))
            if config.enable_saturation and can_split(interval, config.max_interval_level) and first_page_has_next:
                status = interval_status(query, order, interval, "split_after_probe", 0, first_page_results, total_estimate, first_page_has_next, True, search_calls, "", "")
                return [], status

        rows.extend(rows_from_payload(payload=payload, query=query, order=order, interval=interval, page_index=page_index, collected_at=collected_at))
        logger.info("query=%r order=%s interval=%s page=%s returned=%s estimate=%s rows=%s", query, order, interval.label, page_index, len(items), total_estimate, len(rows))

        page_token = payload.get("nextPageToken")
        has_next_page_at_stop = bool(page_token)
        page_index += 1
        if not page_token or not items:
            break
        time.sleep(config.sleep_seconds)

    is_saturated = has_next_page_at_stop and len(rows) >= config.max_results_per_interval
    status_name = "saturated_terminal" if is_saturated and not can_split(interval, config.max_interval_level) else "complete"
    return rows, interval_status(query, order, interval, status_name, len(rows), first_page_results, total_estimate, has_next_page_at_stop, is_saturated, search_calls, "", "")


def interval_status(
    query: str,
    order: str,
    interval: SearchInterval,
    status: str,
    rows_collected: int,
    first_page_results: int,
    total_estimate: int,
    has_next_page_at_stop: bool,
    is_saturated: bool,
    search_calls: int,
    error_reason: str,
    error_message: str,
) -> dict[str, Any]:
    return {
        "query": query,
        "order": order,
        "interval_level": interval.level,
        "interval_label": interval.label,
        "published_after": interval.published_after,
        "published_before": interval.published_before,
        "status": status,
        "rows_collected": rows_collected,
        "first_page_results": first_page_results,
        "total_results_estimate": total_estimate,
        "has_next_page_at_stop": has_next_page_at_stop,
        "is_saturated": is_saturated,
        "search_calls": search_calls,
        "quota_cost_estimate": search_calls * SEARCH_LIST_QUOTA_COST,
        "error_reason": error_reason,
        "error_message": error_message,
        "collected_at": utc_now(),
    }


def search_adaptive_interval(
    *,
    youtube: Any,
    config: ProjectConfig,
    query: str,
    order: str,
    interval: SearchInterval,
    completed_keys: set[tuple[str, str, str, str]],
    seen_video_ids: set[str],
    stop_state: dict[tuple[str, str, str, str], list[float]],
    parent_label_for_stop: str | None,
    force_intervals: bool,
    logger: Any,
) -> list[dict[str, Any]]:
    key = (query, order, interval.level, interval.label)
    if not force_intervals and key in completed_keys:
        logger.info("skip existing query=%r order=%s interval=%s", query, order, interval.label)
        return []

    rows, status = search_interval(youtube=youtube, config=config, query=query, order=order, interval=interval, logger=logger)
    save_interval_status(config, status)
    metrics = build_iteration_metrics(
        config=config,
        query=query,
        order=order,
        interval=interval,
        interval_rows=rows,
        seen_video_ids=seen_video_ids,
        search_calls=int(status.get("search_calls", 0)),
    )
    save_iteration_metrics(config, metrics)
    seen_video_ids.update(str(row["video_id"]) for row in rows if row.get("video_id"))
    if rows:
        save_search_matches(config, rows, logger)
    save_query_effectiveness_summary(config)
    if parent_label_for_stop is not None and status.get("status") not in {"split_after_probe", "error", "quota_exhausted"}:
        branch_key = stop_state_key(query, order, interval.level, parent_label_for_stop)
        stop_state.setdefault(branch_key, []).append(rate_as_float(metrics["new_video_rate"]))
    logger.info(
        "iteration metrics query=%r order=%s interval=%s rows=%s unique=%s new_video_rate=%s new_unique_rate=%s new_relevant_rate=%s",
        query,
        order,
        interval.label,
        len(rows),
        metrics["interval_unique_video_count"],
        metrics["new_video_rate"],
        metrics["new_unique_rate"],
        metrics["new_relevant_rate"],
    )
    if config.enable_saturation and status["is_saturated"] and can_split(interval, config.max_interval_level):
        child_rows: list[dict[str, Any]] = []
        children = child_intervals(interval)
        logger.info("saturated query=%r order=%s interval=%s; split into %s children", query, order, interval.label, len(children))
        for child in children:
            branch_key = stop_state_key(query, order, child.level, interval.label)
            if should_stop_branch(config, stop_state.get(branch_key, [])):
                logger.info(
                    "stop rule query=%r order=%s parent=%s child_level=%s: last rates=%s",
                    query,
                    order,
                    interval.label,
                    child.level,
                    stop_state.get(branch_key, [])[-config.stop_rule_window :],
                )
                save_interval_status(
                    config,
                    interval_status(
                        query,
                        order,
                        child,
                        "stopped_low_yield",
                        0,
                        0,
                        0,
                        False,
                        False,
                        0,
                        "",
                        (
                            f"Stopped because last {config.stop_rule_window} "
                            f"new_video_rate values were below average threshold "
                            f"{config.stop_rule_min_new_video_rate}."
                        ),
                    ),
                )
                continue
            child_rows.extend(
                search_adaptive_interval(
                    youtube=youtube,
                    config=config,
                    query=query,
                    order=order,
                    interval=child,
                    completed_keys=completed_keys,
                    seen_video_ids=seen_video_ids,
                    stop_state=stop_state,
                    parent_label_for_stop=interval.label,
                    force_intervals=force_intervals,
                    logger=logger,
                )
            )
        return rows + child_rows
    return rows


def save_search_matches(config: ProjectConfig, rows: list[dict[str, Any]], logger: Any) -> pd.DataFrame:
    path = config.processed_dir / "video_search_matches.csv"
    existing = read_csv_if_exists(path)
    new_df = pd.DataFrame(rows)
    if existing.empty:
        combined = new_df
    elif new_df.empty:
        combined = existing
    else:
        combined = pd.concat([existing, new_df], ignore_index=True)

    if not combined.empty:
        for column in ["search_order", "search_interval_level", "search_interval_label"]:
            if column not in combined.columns:
                combined[column] = ""
        combined = combined.drop_duplicates(
            ["video_id", "query_used", "search_order", "search_interval_level", "search_interval_label"],
            keep="last",
        )
        combined = combined.sort_values(["video_id", "query_used", "search_order", "search_interval_label"])

    save_table(combined, path, config.processed_dir / "video_search_matches.parquet")
    unique_videos = combined["video_id"].nunique() if not combined.empty else 0
    logger.info("saved video_search_matches rows=%s unique_videos=%s", len(combined), unique_videos)
    return combined


def main() -> None:
    args = parse_args()
    max_results_per_interval = args.max_results_per_interval or args.max_results_per_query
    search_orders = [order.strip() for order in args.orders.split(",") if order.strip()] if args.orders else None
    config = load_config(
        args.env_file,
        date_from=args.date_from,
        date_to=args.date_to,
        output_dir=args.output_dir,
        max_results_per_interval=max_results_per_interval,
        search_orders=search_orders,
        enable_saturation=False if args.no_saturation else None,
        max_interval_level=args.max_interval_level,
    )
    ensure_project_dirs(config)
    logger = setup_logger("search_videos")

    intervals = yearly_intervals(config.date_from, config.effective_date_to)
    logger.info(
        "date_from=%s date_to=%s queries=%s orders=%s saturation=%s max_level=%s",
        config.date_from,
        config.effective_date_to,
        len(config.queries),
        config.search_orders,
        config.enable_saturation,
        config.max_interval_level,
    )
    if args.dry_run:
        print("Dry run: no API calls will be made.")
        print(f"Period: {config.date_from} -> {config.effective_date_to}")
        print(f"Queries: {config.queries}")
        print(f"Orders: {config.search_orders}")
        print(f"Year intervals: {[interval.label for interval in intervals]}")
        print(f"Saturation: {config.enable_saturation}, max level: {config.max_interval_level}")
        print(f"Max results per interval: {config.max_results_per_interval}")
        return

    youtube = build_youtube_client(config.api_key)
    all_rows: list[dict[str, Any]] = []
    before = read_csv_if_exists(config.processed_dir / "video_search_matches.csv")
    before_unique = before["video_id"].nunique() if not before.empty and "video_id" in before else 0
    seen_video_ids = set(before["video_id"].dropna().astype(str)) if not before.empty and "video_id" in before else set()
    completed_keys = load_completed_interval_keys(config)
    stop_state: dict[tuple[str, str, str, str], list[float]] = {}

    search_tasks = [(query, order, interval) for query in config.queries for order in config.search_orders for interval in intervals]
    for query, order, interval in tqdm(search_tasks, desc="search intervals"):
        interval_rows = search_adaptive_interval(
            youtube=youtube,
            config=config,
            query=query,
            order=order,
            interval=interval,
            completed_keys=completed_keys,
            seen_video_ids=seen_video_ids,
            stop_state=stop_state,
            parent_label_for_stop=None,
            force_intervals=args.force_intervals,
            logger=logger,
        )
        all_rows.extend(interval_rows)
        query_unique = len({row["video_id"] for row in interval_rows})
        logger.info(
            "query=%r order=%s interval=%s found_rows=%s unique_video_ids=%s",
            query,
            order,
            interval.label,
            len(interval_rows),
            query_unique,
        )

    combined = save_search_matches(config, all_rows, logger)
    save_query_effectiveness_summary(config)
    after_unique = combined["video_id"].nunique() if not combined.empty else 0
    logger.info("new_unique_video_ids=%s total_unique_video_ids=%s", max(after_unique - before_unique, 0), after_unique)
    print(f"Saved matches: {config.processed_dir / 'video_search_matches.csv'}")
    print(f"Saved interval log: {config.processed_dir / 'search_interval_log.csv'}")
    print(f"Saved iteration metrics: {config.processed_dir / 'search_iteration_metrics.csv'}")
    print(f"Saved query effectiveness: {config.processed_dir / 'query_effectiveness.csv'}")
    print(f"Unique videos: {after_unique} ({max(after_unique - before_unique, 0)} new)")


if __name__ == "__main__":
    main()
