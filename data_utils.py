import requests
import json
import time
from datetime import datetime, timedelta, timezone
import threading
import gzip
import pickle
import asyncio
import aiohttp
from itertools import cycle
from dateutil import parser
import os


def parse_timestamp(ts_str):
    ts_str = str(ts_str)
    fmts = ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d")
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
            pool_connections=10,
            pool_maxsize=20,
            max_retries=3
        )
        _session.mount('https://', adapter)
        _session.mount('http://', adapter)
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


def find_oldest_available_data(item, fallback_date=datetime(2020, 9, 9, 0, 0, 0, tzinfo=timezone.utc)):
    print(f"  → Finding oldest available data...")
    base_url = "https://sky.coflnet.com/api/bazaar"
    url = f"{base_url}/{item}/history"
    
    try:
        _check_rate_limit()
        resp = _get_session().get(url, timeout=15)
        data = resp.json()
        
        if isinstance(data, list) and len(data) > 0:
            oldest_entry = data[-1]
            if isinstance(oldest_entry, dict) and 'timestamp' in oldest_entry:
                ts = oldest_entry['timestamp']
                if isinstance(ts, int):
                    oldest_date = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
                else:
                    oldest_date = parser.parse(str(ts))
                    if oldest_date.tzinfo is None:
                        oldest_date = oldest_date.replace(tzinfo=timezone.utc)
                    else:
                        oldest_date = oldest_date.astimezone(timezone.utc)
                
                print(f"  ✓ Found data starting from: {oldest_date.strftime('%Y-%m-%d %H:%M:%S')}")
                return oldest_date
        
        print(f"  ⚠ No data found, using fallback: {fallback_date.strftime('%Y-%m-%d')}")
        return fallback_date
        
    except Exception as e:
        print(f"  ⚠ Error finding oldest data: {e}, using fallback: {fallback_date.strftime('%Y-%m-%d')}")
        return fallback_date


def _fetch_chunk(item, start, end):
    _check_rate_limit()
    
    base_url = "https://sky.coflnet.com/api/bazaar"
    start_str = start.strftime("%Y-%m-%dT%H:%M:%S.000").replace(":", "%3A")
    end_str = end.strftime("%Y-%m-%dT%H:%M:%S.000").replace(":", "%3A")
    url = f"{base_url}/{item}/history?start={start_str}&end={end_str}"
    
    try:
        proxy = _get_next_proxy()
        proxies = {'http': proxy, 'https': proxy} if proxy else None
        resp = _get_session().get(url, timeout=15, proxies=proxies)
        data = resp.json()
        
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return [data]
        return []
    except Exception as e:
        print(f"  ✗ Error fetching {start.strftime('%Y-%m-%d')}: {e}")
        return []


async def _fetch_chunk_async(session, item, start, end, proxy=None, semaphore=None):
    base_url = "https://sky.coflnet.com/api/bazaar"
    start_str = start.strftime("%Y-%m-%dT%H:%M:%S.000").replace(":", "%3A")
    end_str = end.strftime("%Y-%m-%dT%H:%M:%S.000").replace(":", "%3A")
    url = f"{base_url}/{item}/history?start={start_str}&end={end_str}"
    
    async with semaphore if semaphore else asyncio.Semaphore(100):
        try:
            async with session.get(url, proxy=proxy, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                data = await resp.json()
                
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict):
                    return [data]
                return []
        except Exception as e:
            print(f"  ✗ Error fetching {start.strftime('%Y-%m-%d')}: {e}")
            return []


async def _fetch_all_async(item, chunks, proxies=None, max_concurrent=100):
    semaphore = asyncio.Semaphore(max_concurrent)
    
    connector = aiohttp.TCPConnector(
        limit=max_concurrent * 2,
        limit_per_host=max_concurrent,
        ttl_dns_cache=300
    )
    
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        
        if proxies and len(proxies) > 0:
            for idx, (chunk_start, chunk_end) in enumerate(chunks):
                proxy = proxies[idx % len(proxies)]
                tasks.append(_fetch_chunk_async(session, item, chunk_start, chunk_end, proxy, semaphore))
        else:
            for chunk_start, chunk_end in chunks:
                tasks.append(_fetch_chunk_async(session, item, chunk_start, chunk_end, None, semaphore))
        
        results = []
        completed = 0
        total = len(tasks)
        
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.extend(result)
            completed += 1
            
            if completed % max(1, total // 10) == 0 or completed % 100 == 0:
                print(f"  → Progress: {completed}/{total} chunks ({100*completed//total}%)")
        
        return results


def fetch_all_data(item, start=None, end=None, interval_seconds=82800, use_binary_search=True, use_fast_mode=False):
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
    
    print(f"  → Fetching {len(chunks)} chunks from {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}...")
    
    if use_fast_mode and _use_proxies:
        print(f"  → Using FAST MODE with {len(_proxy_pool)} proxies for parallel fetching")
        raw_combined = asyncio.run(_fetch_all_async(item, chunks, _proxy_pool, max_concurrent=len(_proxy_pool)))
    elif use_fast_mode:
        print(f"  → Using FAST MODE without proxies (max 100 concurrent)")
        raw_combined = asyncio.run(_fetch_all_async(item, chunks, None, max_concurrent=100))
    else:
        raw_combined = []
        for idx, (chunk_start, chunk_end) in enumerate(chunks, 1):
            if idx % 10 == 0:
                print(f"  → Progress: {idx}/{len(chunks)} chunks")
            
            data = _fetch_chunk(item, chunk_start, chunk_end)
            raw_combined.extend(data)
    
    print(f"  ✓ Fetched {len(raw_combined)} total entries")
    return raw_combined


def fetch_all_data_fast(item, start=None, end=None, interval_seconds=82800, use_binary_search=True, max_concurrent=None):
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
    
    print(f"  → Fetching {len(chunks)} chunks from {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}...")
    
    if max_concurrent is None:
        max_concurrent = len(_proxy_pool) if _use_proxies else 100
    
    if _use_proxies:
        print(f"  → FAST MODE: {len(_proxy_pool)} proxies, {max_concurrent} concurrent requests")
        raw_combined = asyncio.run(_fetch_all_async(item, chunks, _proxy_pool, max_concurrent=max_concurrent))
    else:
        print(f"  → FAST MODE: No proxies, {max_concurrent} concurrent requests")
        print(f"  → TIP: Use configure_proxy_pool() for even faster speeds with IP rotation")
        raw_combined = asyncio.run(_fetch_all_async(item, chunks, None, max_concurrent=max_concurrent))
    
    print(f"  ✓ Fetched {len(raw_combined)} total entries")
    return raw_combined


def load_or_fetch_item_data(item_id, fetch_if_missing=True, update_with_new_data=False, use_compression=True, use_fast_mode=False):
    json_dir = os.path.join(os.path.dirname(__file__), "bazaar_data")
    legacy_dir = os.path.expanduser("~/Json Files")

    os.makedirs(json_dir, exist_ok=True)

    if use_compression:
        primary_filename = os.path.join(json_dir, f"bazaar_history_{item_id}.pkl.gz")
        primary_json_filename = os.path.join(json_dir, f"bazaar_history_combined_{item_id}.json")
        legacy_filename = os.path.join(legacy_dir, f"bazaar_history_{item_id}.pkl.gz")
        legacy_json_filename = os.path.join(legacy_dir, f"bazaar_history_combined_{item_id}.json")

        if os.path.exists(primary_filename):
            filename = primary_filename
        elif os.path.exists(legacy_filename):
            filename = legacy_filename
            print(f"  → Using legacy cache from {legacy_dir} for {item_id}")
        else:
            if os.path.exists(primary_json_filename):
                filename = primary_filename
                print(f"  → Migrating {item_id} to compressed format...")
                try:
                    with open(primary_json_filename, 'r') as f:
                        data = json.load(f)
                    with gzip.open(primary_filename, 'wb') as f:
                        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
                    os.remove(primary_json_filename)
                    print(f"  ✓ Migrated and compressed")
                except Exception as e:
                    print(f"  ✗ Migration failed: {e}")
                    filename = primary_json_filename
            elif os.path.exists(legacy_json_filename):
                filename = legacy_json_filename
                print(f"  → Using legacy JSON cache from {legacy_dir} for {item_id}")
            else:
                filename = primary_filename
    else:
        primary_filename = os.path.join(json_dir, f"bazaar_history_combined_{item_id}.json")
        legacy_filename = os.path.join(legacy_dir, f"bazaar_history_combined_{item_id}.json")
        if os.path.exists(primary_filename):
            filename = primary_filename
        elif os.path.exists(legacy_filename):
            filename = legacy_filename
            print(f"  → Using legacy JSON cache from {legacy_dir} for {item_id}")
        else:
            filename = primary_filename
    
    if not os.path.exists(filename):
        if fetch_if_missing:
            print(f"  → No cache found, fetching full history from API...")
            if use_fast_mode:
                all_data = fetch_all_data_fast(item_id, use_binary_search=True)
            else:
                all_data = fetch_all_data(item_id, use_binary_search=True)
            
            if use_compression:
                with gzip.open(filename, 'wb') as f:
                    pickle.dump(all_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                with open(filename, 'w') as f:
                    json.dump(all_data, f)
            
            print(f"  ✓ Saved {len(all_data)} entries")
            return all_data
        else:
            print(f"  ✗ File {filename} not found")
            return None
    
    print(f"  → Loading from cache...")
    if use_compression and filename.endswith('.pkl.gz'):
        with gzip.open(filename, 'rb') as f:
            data = pickle.load(f)
    else:
        with open(filename, 'r') as f:
            data = json.load(f)
    
    if update_with_new_data and data:
        latest_timestamp = None
        for entry in reversed(data):
            if isinstance(entry, dict) and 'timestamp' in entry:
                try:
                    latest_timestamp = parse_timestamp(entry['timestamp'])
                    break
                except:
                    continue
        
        if latest_timestamp:
            print(f"  → Fetching new data since {latest_timestamp.strftime('%Y-%m-%d')}...")
            if use_fast_mode:
                new_data = fetch_all_data_fast(item_id, start=latest_timestamp, end=datetime.now(timezone.utc), use_binary_search=False)
            else:
                new_data = fetch_all_data(item_id, start=latest_timestamp, end=datetime.now(timezone.utc), use_binary_search=False)
            
            if new_data:
                data.extend(new_data)
                
                if use_compression and filename.endswith('.pkl.gz'):
                    with gzip.open(filename, 'wb') as f:
                        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
                else:
                    with open(filename, 'w') as f:
                        json.dump(data, f)
                
                print(f"  ✓ Added {len(new_data)} new entries (total: {len(data)})")
            else:
                print(f"  ✓ No new data available")
    else:
        print(f"  ✓ Loaded {len(data)} entries")
    
    return data


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

