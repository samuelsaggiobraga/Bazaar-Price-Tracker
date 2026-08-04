import requests
import time
import os
import gzip
import pickle
from datetime import datetime, timedelta, timezone
import threading
import asyncio
import aiohttp
from itertools import cycle
from dateutil import parser

# TimescaleDB is the primary store, but importing it eagerly made psycopg2 and
# a running Postgres a hard requirement for the entire codebase -- you could
# not import the trainer, the Flask API or any Testing Files script without
# them, and the published gzip-pickle dataset became unloadable. Degrade to the
# local bazaar_data/ cache instead.
try:
    from Utils.db import get_item_history, get_latest_timestamp, insert_item_history

    DB_AVAILABLE = True
    DB_IMPORT_ERROR = None
except Exception as _db_err:  # psycopg2 missing, no DATABASE_URL, etc.
    DB_AVAILABLE = False
    DB_IMPORT_ERROR = _db_err

    def get_item_history(*a, **kw):
        raise RuntimeError(f"TimescaleDB unavailable: {DB_IMPORT_ERROR}")

    def get_latest_timestamp(*a, **kw):
        raise RuntimeError(f"TimescaleDB unavailable: {DB_IMPORT_ERROR}")

    def insert_item_history(*a, **kw):
        raise RuntimeError(f"TimescaleDB unavailable: {DB_IMPORT_ERROR}")


_SSL_CONTEXT = None


def _ssl_context():
    """Trust store for aiohttp.

    requests bundles certifi, but aiohttp falls back to the OS store -- which on
    a python.org macOS build is empty unless Install Certificates.command was
    run. Every async fetch then dies with CERTIFICATE_VERIFY_FAILED while the
    synchronous helpers keep working, which makes it look like a rate-limit or
    proxy fault rather than a trust-store one.
    """
    global _SSL_CONTEXT
    if _SSL_CONTEXT is None:
        import ssl

        try:
            import certifi

            _SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
        except Exception:
            _SSL_CONTEXT = ssl.create_default_context()
    return _SSL_CONTEXT


def _pickle_cache_path(item_id):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_dir = os.path.join(base_dir, "bazaar_data")
    os.makedirs(json_dir, exist_ok=True)
    return os.path.join(json_dir, f"bazaar_history_{item_id}.pkl.gz")


def load_pickle_cache(item_id):
    """Read the gzip-pickle cache, the format the public dataset ships in."""
    path = _pickle_cache_path(item_id)
    if not os.path.exists(path):
        return None
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def save_pickle_cache(item_id, data):
    with gzip.open(_pickle_cache_path(item_id), "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def _extend_pickle_cache(item_id, cached):
    """Top the cache up to now, mirroring the database path's incremental update.

    Published dataset snapshots go stale, so training on one straight out of the
    box silently evaluates on data that stops months ago.
    """
    latest = None
    for entry in reversed(cached):
        if isinstance(entry, dict) and "timestamp" in entry:
            try:
                latest = parse_timestamp(entry["timestamp"])
                break
            except Exception:
                continue
    if latest is None:
        print("  ⚠ Could not read a timestamp from the cache; skipping update")
        return cached

    now = datetime.now(timezone.utc)
    stale_days = (now - latest).total_seconds() / 86400
    if stale_days < 1:
        print(f"  ✓ Cache is current (newest {latest:%Y-%m-%d %H:%M} UTC)")
        return cached

    print(
        f"  → Cache ends {latest:%Y-%m-%d} ({stale_days:.0f} days stale); "
        f"fetching through {now:%Y-%m-%d}..."
    )
    new_data = fetch_all_data_async(
        item_id, start=latest, end=now, use_binary_search=False
    )
    if not new_data:
        print("  ✓ No new data returned")
        return cached

    # The API is inclusive at the boundary and chunks can overlap, so dedupe on
    # timestamp rather than blindly extending.
    seen = set()
    merged = []
    for row in list(cached) + list(new_data):
        ts = row.get("timestamp") if isinstance(row, dict) else None
        if ts in seen:
            continue
        seen.add(ts)
        merged.append(row)
    merged.sort(key=lambda r: str(r.get("timestamp")))

    added = len(merged) - len(cached)
    save_pickle_cache(item_id, merged)
    print(f"  ✓ Added {added:,} new entries (total {len(merged):,})")
    return merged


def parse_timestamp(ts_str):
    ts_str = str(ts_str)
    fmts = (
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    )
    for fmt in fmts:
        try:
            dt = datetime.strptime(ts_str, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except Exception:
            continue
    try:
        dt = datetime.fromisoformat(ts_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt
    except Exception:
        raise ValueError(f"Unrecognized timestamp format: {ts_str}")


_session = None


def _get_session():
    """Get or create a persistent requests session with connection pooling."""
    global _session
    if _session is None:
        _session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10, pool_maxsize=20, max_retries=3
        )
        _session.mount("https://", adapter)
        _session.mount("http://", adapter)
    return _session


_proxy_pool = []
_proxy_cycle = None
_use_proxies = False


def configure_proxy_pool(proxy_list):

    global _proxy_pool, _proxy_cycle, _use_proxies

    if proxy_list and len(proxy_list) > 0:
        _proxy_pool = proxy_list
        _proxy_cycle = cycle(_proxy_pool)
        _use_proxies = True
        print(f"✓ Configured {len(_proxy_pool)} proxies for IP rotation")
    else:
        _proxy_pool = []
        _proxy_cycle = None
        _use_proxies = False
        print("✓ Disabled proxy usage")


def _get_next_proxy():
    if _use_proxies and _proxy_cycle:
        return next(_proxy_cycle)
    return None


_rate_limit_lock = threading.Lock()
_requests_made = 0
_last_reset_time = time.time()
_max_requests = 30
_window_seconds = 10


def _check_rate_limit():
    global _requests_made, _last_reset_time

    with _rate_limit_lock:
        current_time = time.time()

        if current_time - _last_reset_time >= _window_seconds:
            _requests_made = 0
            _last_reset_time = current_time

        if _requests_made >= _max_requests:
            sleep_time = _window_seconds - (current_time - _last_reset_time)
            if sleep_time > 0:
                print(f"  → Rate limit: waiting {sleep_time:.1f}s...")
                time.sleep(sleep_time)
            _requests_made = 0
            _last_reset_time = time.time()

        _requests_made += 1


async def _async_check_rate_limit(_async_rate_limit_lock):
    global _requests_made, _last_reset_time

    # Use the async lock
    async with _async_rate_limit_lock:
        current_time = time.time()

        if current_time - _last_reset_time >= _window_seconds:
            _requests_made = 0
            _last_reset_time = current_time

        if _requests_made >= _max_requests:
            sleep_time = _window_seconds - (current_time - _last_reset_time)
            if sleep_time > 0:
                print(f"  → Rate limit: waiting {sleep_time:.1f}s...")
                # The crucial fix: This lets the event loop process other tasks while waiting
                await asyncio.sleep(sleep_time)

            _requests_made = 0
            _last_reset_time = time.time()

        _requests_made += 1


def find_oldest_available_data(
    item, fallback_date=datetime(2020, 9, 9, 0, 0, 0, tzinfo=timezone.utc)
):
    """Find the oldest available data for an item by fetching full history.

    Args:
        item: The item ID to check
        fallback_date: Date to use if API call fails (default: Skyblock bazaar launch)

    Returns:
        datetime: The oldest date with available data, clamped to exactly 1 year ago for non-premium users, or fallback_date if not found
    """
    print("  → Finding oldest available data...")
    base_url = (
        "https://sky.coflnet.com/api/bazaar"  # Plug in the API key if you got premium
    )
    url = f"{base_url}/{item}/history"

    try:
        _check_rate_limit()
        resp = _get_session().get(url, timeout=15)
        data = resp.json()

        # Calculate exactly 1 year ago (the paywall limit for non-premium users)
        one_year_ago = datetime.now(timezone.utc) - timedelta(days=365)

        if isinstance(data, list) and len(data) > 0:
            oldest_entry = data[-1]
            if isinstance(oldest_entry, dict) and "timestamp" in oldest_entry:
                ts = oldest_entry["timestamp"]
                if isinstance(ts, int):
                    oldest_date = datetime.fromtimestamp(
                        ts / 1000, tz=timezone.utc
                    )  # ms to s
                else:
                    oldest_date = parser.parse(str(ts))
                    # Ensure timezone-aware
                    if oldest_date.tzinfo is None:
                        oldest_date = oldest_date.replace(tzinfo=timezone.utc)

                # FORCE THE DATE TO BE NO OLDER THAN 365 DAYS AGO
                clamped_date = max(oldest_date, one_year_ago)

                print(
                    f"  ✓ Found data starting from: {oldest_date.strftime('%Y-%m-%d %H:%M:%S')} (Clamped to {clamped_date.strftime('%Y-%m-%d %H:%M:%S')} due to paywall)"
                )
                return clamped_date

        print(
            f"  ⚠ No data found, using 1-year fallback: {one_year_ago.strftime('%Y-%m-%d')}"
        )
        return one_year_ago

    except Exception as e:
        one_year_ago = datetime.now(timezone.utc) - timedelta(days=365)
        print(
            f"  ⚠ Error finding oldest data: {e}, using 1-year fallback: {one_year_ago.strftime('%Y-%m-%d')}"
        )
        return one_year_ago


async def _fetch_chunk_async(
    session,
    item,
    start,
    end,
    proxy=None,
    semaphore=None,
    _async_rate_limit_lock=None,
    max_retries=3,
):
    base_url = "https://sky.coflnet.com/api/bazaar"
    start_str = start.strftime("%Y-%m-%dT%H:%M:%S.000").replace(":", "%3A")
    end_str = end.strftime("%Y-%m-%dT%H:%M:%S.000").replace(":", "%3A")
    url = f"{base_url}/{item}/history?start={start_str}&end={end_str}"

    async with semaphore:
        for attempt in range(max_retries):
            try:
                if not _use_proxies:  # Single ip
                    await _async_check_rate_limit(_async_rate_limit_lock)

                async with session.get(
                    url, proxy=proxy, timeout=aiohttp.ClientTimeout(total=20)
                ) as resp:
                    # Check for bad status codes (e.g., 429 Too Many Requests, 502 Bad Gateway)
                    resp.raise_for_status()

                    data = await resp.json(
                        content_type=None
                    )  # aiohttp NEEDS the content_type=None bypass

                    if isinstance(data, list):
                        return data
                    elif isinstance(data, dict):
                        return [data]
                    return []

            except Exception as e:
                is_last_attempt = attempt == max_retries - 1
                if is_last_attempt:
                    print(
                        f"  ✗ Fatal error fetching {start.strftime('%Y-%m-%d')} after {max_retries} attempts: {e}"
                    )
                    return []
                else:
                    # If using proxies, grab a fresh one from the global cycle for the next attempt
                    if _use_proxies:
                        proxy = _get_next_proxy()


async def _fetch_all_async(item, chunks, proxies=None, max_concurrent=100):
    semaphore = asyncio.Semaphore(max_concurrent)
    _async_rate_limit_lock = asyncio.Lock()

    connector = aiohttp.TCPConnector(
        limit=max_concurrent * 2,
        limit_per_host=max_concurrent,
        ttl_dns_cache=300,
        ssl=_ssl_context(),
    )

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []

        if _use_proxies:
            for idx, (chunk_start, chunk_end) in enumerate(chunks):
                proxy = proxies[idx % len(proxies)]
                tasks.append(
                    _fetch_chunk_async(
                        session=session,
                        item=item,
                        start=chunk_start,
                        end=chunk_end,
                        proxy=proxy,
                        semaphore=semaphore,
                        _async_rate_limit_lock=None,  # No need to use lock since the limit of concurrent request is the lenght of the proxies
                    )
                )
        else:
            for chunk_start, chunk_end in chunks:
                tasks.append(
                    _fetch_chunk_async(
                        session=session,
                        item=item,
                        start=chunk_start,
                        end=chunk_end,
                        proxy=None,
                        semaphore=semaphore,
                        _async_rate_limit_lock=_async_rate_limit_lock,  # 100 concurrent request if no proxies, lock is needed
                    )
                )

        results = []
        completed = 0
        total = len(tasks)

        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.extend(result)
            completed += 1

            if completed % max(1, total // 10) == 0 or completed % 100 == 0:
                print(
                    f"  → Progress: {completed}/{total} chunks ({100 * completed // total}%)"
                )

        return results


def fetch_all_data_async(
    item,
    start=None,
    end=None,
    interval_seconds=82800,
    use_binary_search=True,
):
    if end is None:
        end = datetime.now(timezone.utc)

    if start is None and use_binary_search:
        start = find_oldest_available_data(item)
    elif start is None:
        start = datetime(2020, 9, 9, 0, 0, 0, tzinfo=timezone.utc)

    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    else:
        start = start.astimezone(timezone.utc)

    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    else:
        end = end.astimezone(timezone.utc)

    interval = timedelta(seconds=interval_seconds)

    chunks = []
    current = start
    while current + interval <= end:
        chunks.append((current, current + interval))
        current += interval
    # Catch the remaining tail end of the data
    if current < end:
        chunks.append((current, end))

    print(
        f"  → Fetching {len(chunks)} chunks from {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}..."
    )

    max_concurrent = len(_proxy_pool) if _use_proxies else 100

    if _use_proxies:
        print(
            f"  → FAST MODE: {len(_proxy_pool)} proxies, {max_concurrent} concurrent requests"
        )
        raw_combined = asyncio.run(
            _fetch_all_async(item, chunks, _proxy_pool, max_concurrent=max_concurrent)
        )
        print(f"  ✓ Fetched {len(raw_combined)} total entries")
        return raw_combined
    else:
        print("  → SLOW MODE: without proxies (max 100 concurrent)")
        raw_combined = asyncio.run(
            _fetch_all_async(item, chunks, None, max_concurrent=max_concurrent)
        )
        print(f"  ✓ Fetched {len(raw_combined)} total entries")
        return raw_combined


def load_or_fetch_item_data(
    item_id,
    fetch_if_missing=True,
    update_with_new_data=False,
):
    # No database configured: serve the gzip-pickle cache, which is also the
    # format the public HuggingFace dataset ships in.
    if not DB_AVAILABLE:
        cached = load_pickle_cache(item_id)
        if cached:
            print(f"  ✓ Loaded {len(cached)} entries from local cache (no database)")
            if update_with_new_data:
                cached = _extend_pickle_cache(item_id, cached)
            return cached
        if not fetch_if_missing:
            print("  ✗ No local cache and fetch_if_missing is False")
            return None
        print("  → No local cache, fetching full history from API...")
        all_data = fetch_all_data_async(item_id, use_binary_search=True)
        if all_data:
            save_pickle_cache(item_id, all_data)
            print(f"  ✓ Saved {len(all_data)} entries to local cache")
        return all_data

    # Fetch from database
    print("  → Loading from database...")
    data = get_item_history(item_id, order_by="ASC")

    # If history is missing
    # fetch the api and insert them in database
    if not data:
        if fetch_if_missing:
            print("  → No data found, fetching full history from API...")

            all_data = fetch_all_data_async(item_id, use_binary_search=True)
            insert_item_history(item_id, all_data)

            print(f"  ✓ Saved {len(all_data)} entries")
            return get_item_history(item_id, order_by="ASC")
        else:
            print("  ✗ Data is empty and fetch_if_missing is False ")
            return None

    # If history exists and theres new data available,
    # fetch the api to extend the data and insert them in database
    if update_with_new_data:
        latest_timestamp = get_latest_timestamp(item_id)

        if latest_timestamp:
            print(
                f"  → Fetching new data since {latest_timestamp.strftime('%Y-%m-%d')}..."
            )

            new_data = fetch_all_data_async(
                item_id,
                start=latest_timestamp,
                end=datetime.now(timezone.utc),
                use_binary_search=False,
            )

            if new_data:
                data.extend(new_data)

                insert_item_history(item_id, new_data)

                print(f"  ✓ Added {len(new_data)} new entries (total: {len(data)})")
            else:
                print("  ✓ No new data available")

    print(f"  ✓ Loaded {len(data)} entries")
    return get_item_history(item_id, order_by="ASC")


def fetch_recent_data(item_id, hours=24):
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=hours)

    start_str = start.strftime("%Y-%m-%dT%H:%M:%S.000").replace(":", "%3A")
    end_str = end.strftime("%Y-%m-%dT%H:%M:%S.000").replace(":", "%3A")

    url = f"https://sky.coflnet.com/api/bazaar/{item_id}/history?start={start_str}&end={end_str}"

    try:
        resp = _get_session().get(url, timeout=10)
        data = resp.json()

        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return [data]
        else:
            return []
    except Exception as e:
        print(f"Error fetching recent data: {e}")
        return []
