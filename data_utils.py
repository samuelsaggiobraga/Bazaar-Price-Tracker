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
    """Parse timestamp from various formats."""
    fmts = ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d")
    for fmt in fmts:
        try:
            return datetime.strptime(ts_str, fmt)
        except Exception:
            continue
    try:
        return datetime.fromisoformat(ts_str)
    except Exception:
        raise ValueError(f"Unrecognized timestamp format: {ts_str}")

# Global session with connection pooling
_session = None

def _get_session():
    """Get or create a persistent requests session with connection pooling."""
    global _session
    if _session is None:
        _session = requests.Session()
        # Connection pooling speeds up repeated requests
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=20,
            max_retries=3
        )
        _session.mount('https://', adapter)
        _session.mount('http://', adapter)
    return _session


# Proxy pool management
_proxy_pool = []
_proxy_cycle = None
_use_proxies = False

def configure_proxy_pool(proxy_list):
    """Configure a pool of proxies for IP rotation.
    
    Args:
        proxy_list: List of proxy URLs in format ['http://ip:port', 'http://ip:port', ...]
                   Set to None or empty list to disable proxy usage
    
    Example:
        configure_proxy_pool([
            'http://proxy1.example.com:8080',
            'http://proxy2.example.com:8080',
            'http://user:pass@proxy3.example.com:8080'  # With auth
        ])
    """
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
    """Get next proxy from the rotation pool."""
    if _use_proxies and _proxy_cycle:
        return next(_proxy_cycle)
    return None


# Global rate limiter
_rate_limit_lock = threading.Lock()
_requests_made = 0
_last_reset_time = time.time()
_max_requests = 30
_window_seconds = 10

def _check_rate_limit():
    """Check and enforce API rate limit."""
    global _requests_made, _last_reset_time
    
    with _rate_limit_lock:
        current_time = time.time()
        
        # Reset counter if window has passed
        if current_time - _last_reset_time >= _window_seconds:
            _requests_made = 0
            _last_reset_time = current_time
        
        # Wait if we've hit the limit
        if _requests_made >= _max_requests:
            sleep_time = _window_seconds - (current_time - _last_reset_time)
            if sleep_time > 0:
                print(f"  → Rate limit: waiting {sleep_time:.1f}s...")
                time.sleep(sleep_time)
            _requests_made = 0
            _last_reset_time = time.time()
        
        _requests_made += 1


def find_oldest_available_data(item, fallback_date=datetime(2020, 9, 9, 0, 0, 0, tzinfo=timezone.utc)):
    """Find the oldest available data for an item by fetching full history.
    
    Args:
        item: The item ID to check
        fallback_date: Date to use if API call fails (default: Skyblock bazaar launch)
        
    Returns:
        datetime: The oldest date with available data, or fallback_date if not found
    """
    print(f"  → Finding oldest available data...")
    base_url = "https://sky.coflnet.com/api/bazaar"
    url = f"{base_url}/{item}/history"
    
    try:
        _check_rate_limit()
        resp = _get_session().get(url, timeout=15)
        data = resp.json()
        
        if isinstance(data, list) and len(data) > 0:
            # Get the first (oldest) entry's timestamp
            oldest_entry = data[-1]
            if isinstance(oldest_entry, dict) and 'timestamp' in oldest_entry:
                # Parse timestamp - handle Unix timestamp (int) or ISO string
                ts = oldest_entry['timestamp']
                if isinstance(ts, int):
                    oldest_date = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)  # ms to s
                else:
                    # Try parsing as ISO string
                    oldest_date = parser.parse(str(ts))
                    # Ensure timezone-aware
                    if oldest_date.tzinfo is None:
                        oldest_date = oldest_date.replace(tzinfo=timezone.utc)
                
                print(f"  ✓ Found data starting from: {oldest_date.strftime('%Y-%m-%d %H:%M:%S')}")
                return oldest_date
        
        print(f"  ⚠ No data found, using fallback: {fallback_date.strftime('%Y-%m-%d')}")
        return fallback_date
        
    except Exception as e:
        print(f"  ⚠ Error finding oldest data: {e}, using fallback: {fallback_date.strftime('%Y-%m-%d')}")
        return fallback_date


def _fetch_chunk(item, start, end):
    """Fetch a single time chunk (for parallel fetching)."""
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
    """Async fetch a single time chunk with optional proxy."""
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
    """Fetch all chunks asynchronously with proxy rotation.
    
    Args:
        item: Item ID to fetch
        chunks: List of (start, end) datetime tuples
        proxies: List of proxy URLs to rotate through
        max_concurrent: Maximum concurrent requests
    
    Returns:
        Combined list of all fetched data
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Create connector with high limits for many concurrent connections
    connector = aiohttp.TCPConnector(
        limit=max_concurrent * 2,
        limit_per_host=max_concurrent,
        ttl_dns_cache=300
    )
    
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        
        # Distribute chunks across proxies if available
        if proxies and len(proxies) > 0:
            for idx, (chunk_start, chunk_end) in enumerate(chunks):
                proxy = proxies[idx % len(proxies)]
                tasks.append(_fetch_chunk_async(session, item, chunk_start, chunk_end, proxy, semaphore))
        else:
            for chunk_start, chunk_end in chunks:
                tasks.append(_fetch_chunk_async(session, item, chunk_start, chunk_end, None, semaphore))
        
        # Fetch all chunks concurrently with progress updates
        results = []
        completed = 0
        total = len(tasks)
        
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.extend(result)
            completed += 1
            
            # Progress updates every 10% or at least every 100 requests
            if completed % max(1, total // 10) == 0 or completed % 100 == 0:
                print(f"  → Progress: {completed}/{total} chunks ({100*completed//total}%)")
        
        return results


def fetch_all_data(item, start=None, end=None, interval_seconds=82800, use_binary_search=True, use_fast_mode=False):
    """Fetch bazaar data from API with optimizations.
    
    Args:
        item: The item ID to fetch
        start: Start datetime (if None, uses binary search to find oldest)
        end: End datetime (defaults to now)
        interval_seconds: Interval between API calls
        use_binary_search: Whether to auto-find oldest available data
        use_fast_mode: Use async multi-IP fetching (requires proxy pool configured)
        
    Returns:
        List of data entries from the API
    """
    if end is None:
        end = datetime.now(timezone.utc)
    
    # Use binary search to find oldest data if start not specified
    if start is None and use_binary_search:
        start = find_oldest_available_data(item)
    elif start is None:
        start = datetime(2020, 9, 9, 0, 0, 0, tzinfo=timezone.utc)  # fallback: bazaar launch date
    
    interval = timedelta(seconds=interval_seconds)
    
    # Calculate all time chunks
    chunks = []
    current = start
    while current + interval <= end:
        chunks.append((current, current + interval))
        current += interval
    
    print(f"  → Fetching {len(chunks)} chunks from {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}...")
    
    # Use fast async mode if requested and proxies available
    if use_fast_mode and _use_proxies:
        print(f"  → Using FAST MODE with {len(_proxy_pool)} proxies for parallel fetching")
        raw_combined = asyncio.run(_fetch_all_async(item, chunks, _proxy_pool, max_concurrent=len(_proxy_pool)))
    elif use_fast_mode:
        print(f"  → Using FAST MODE without proxies (max 100 concurrent)")
        raw_combined = asyncio.run(_fetch_all_async(item, chunks, None, max_concurrent=100))
    else:
        # Standard sequential fetching with rate limiting
        raw_combined = []
        for idx, (chunk_start, chunk_end) in enumerate(chunks, 1):
            if idx % 10 == 0:
                print(f"  → Progress: {idx}/{len(chunks)} chunks")
            
            data = _fetch_chunk(item, chunk_start, chunk_end)
            raw_combined.extend(data)
    
    print(f"  ✓ Fetched {len(raw_combined)} total entries")
    return raw_combined


def fetch_all_data_fast(item, start=None, end=None, interval_seconds=82800, use_binary_search=True, max_concurrent=None):
    """Fetch bazaar data using multi-IP async mode for maximum speed.
    
    This is a convenience wrapper that automatically enables fast mode.
    Configure proxies first using configure_proxy_pool() for best performance.
    
    Args:
        item: The item ID to fetch
        start: Start datetime (if None, uses binary search to find oldest)
        end: End datetime (defaults to now)
        interval_seconds: Interval between API calls
        use_binary_search: Whether to auto-find oldest available data
        max_concurrent: Override concurrent request limit (default: proxy count or 100)
        
    Returns:
        List of data entries from the API
        
    Example:
        # With proxies (fastest)
        configure_proxy_pool(['http://proxy1:8080', 'http://proxy2:8080', ...])
        data = fetch_all_data_fast('ENCHANTMENT_ULTIMATE_WISE_5')
        
        # Without proxies (still faster than sequential)
        data = fetch_all_data_fast('ENCHANTMENT_ULTIMATE_WISE_5')
    """
    if end is None:
        end = datetime.now(timezone.utc)
    
    # Use binary search to find oldest data if start not specified
    if start is None and use_binary_search:
        start = find_oldest_available_data(item)
    elif start is None:
        start = datetime(2020, 9, 9, 0, 0, 0)
    
    interval = timedelta(seconds=interval_seconds)
    
    # Calculate all time chunks
    chunks = []
    current = start
    while current + interval <= end:
        chunks.append((current, current + interval))
        current += interval
    
    print(f"  → Fetching {len(chunks)} chunks from {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}...")
    
    # Determine concurrency limit
    if max_concurrent is None:
        max_concurrent = len(_proxy_pool) if _use_proxies else 100
    
    # Use async fetching
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
    """Load item data from file, fetching from API if it doesn't exist.
    
    Args:
        item_id: The item ID to load
        fetch_if_missing: Whether to fetch from API if file doesn't exist
        update_with_new_data: If True and file exists, fetch only new data since last update
        use_compression: Use gzip compression for faster I/O (recommended)
        use_fast_mode: Use async multi-IP fetching (requires proxy pool for best speed)
        
    Returns:
        List of data entries, or None if file doesn't exist and fetch_if_missing is False
    """
    
    json_dir = os.path.join(os.path.dirname(__file__), "bazaar_data")
    
    # Try compressed file first (.pkl.gz is 5-10x smaller and faster)
    if use_compression:
        filename = os.path.join(json_dir, f"bazaar_history_{item_id}.pkl.gz")
        json_filename = os.path.join(json_dir, f"bazaar_history_combined_{item_id}.json")
        
        # Migrate old JSON to compressed format if exists
        if os.path.exists(json_filename) and not os.path.exists(filename):
            print(f"  → Migrating {item_id} to compressed format...")
            try:
                with open(json_filename, 'r') as f:
                    data = json.load(f)
                with gzip.open(filename, 'wb') as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
                # Remove old file to save space
                os.remove(json_filename)
                print(f"  ✓ Migrated and compressed")
            except Exception as e:
                print(f"  ✗ Migration failed: {e}")
                filename = json_filename  # Fallback to JSON
    else:
        filename = os.path.join(json_dir, f"bazaar_history_combined_{item_id}.json")
    
    # Create directory if it doesn't exist
    os.makedirs(json_dir, exist_ok=True)
    
    if not os.path.exists(filename):
        if fetch_if_missing:
            print(f"  → No cache found, fetching full history from API...")
            if use_fast_mode:
                all_data = fetch_all_data_fast(item_id, use_binary_search=True)
            else:
                all_data = fetch_all_data(item_id, use_binary_search=True)
            
            # Save the fetched data
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
    
    # Load existing data
    print(f"  → Loading from cache...")
    if use_compression and filename.endswith('.pkl.gz'):
        with gzip.open(filename, 'rb') as f:
            data = pickle.load(f)
    else:
        with open(filename, 'r') as f:
            data = json.load(f)
    
    # Update with new data if requested
    if update_with_new_data and data:
        # Find the most recent timestamp in existing data
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
            # Fetch data from latest timestamp to now (no binary search needed for updates)
            if use_fast_mode:
                new_data = fetch_all_data_fast(item_id, start=latest_timestamp, end=datetime.now(timezone.utc), use_binary_search=False)
            else:
                new_data = fetch_all_data(item_id, start=latest_timestamp, end=datetime.now(timezone.utc), use_binary_search=False)
            
            if new_data:
                # Append new data
                data.extend(new_data)
                
                # Save updated data
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
    """Fetch only recent data from the API for real-time predictions.
    
    Args:
        item_id: The item ID to fetch
        hours: Number of hours of recent data to fetch (default 24)
        
    Returns:
        List of recent data entries from the API
    """
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

