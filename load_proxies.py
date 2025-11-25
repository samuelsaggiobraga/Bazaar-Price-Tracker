#!/usr/bin/env python3
"""
Load proxies with proper format conversion.
Handles format: IP:PORT:USERNAME:PASSWORD
"""

from data_utils import configure_proxy_pool

def parse_proxy_line(line):
    """
    Parse proxy line and convert to proper format.
    
    Supports formats:
    - IP:PORT:USERNAME:PASSWORD -> http://USERNAME:PASSWORD@IP:PORT
    - IP:PORT -> http://IP:PORT
    - http://... (already formatted)
    """
    line = line.strip()
    
    if not line or line.startswith('#'):
        return None
    
    # Already formatted
    if line.startswith('http://') or line.startswith('https://'):
        return line
    
    # Parse IP:PORT:USERNAME:PASSWORD format
    parts = line.split(':')
    
    if len(parts) == 4:
        # IP:PORT:USERNAME:PASSWORD
        ip, port, username, password = parts
        return f'http://{username}:{password}@{ip}:{port}'
    elif len(parts) == 2:
        # IP:PORT (no auth)
        ip, port = parts
        return f'http://{ip}:{port}'
    else:
        print(f"⚠ Skipping invalid format: {line[:50]}")
        return None


def load_proxies(filename='proxies.txt'):
    """Load and parse proxies from file."""
    proxies = []
    
    with open(filename, 'r') as f:
        for line in f:
            proxy = parse_proxy_line(line)
            if proxy:
                proxies.append(proxy)
    
    return proxies


if __name__ == '__main__':
    print("Loading and configuring proxies...")
    
    proxies = load_proxies('proxies.txt')
    print(f"✓ Loaded {len(proxies)} proxies")
    
    # Show first few as example
    if proxies:
        print("\nFirst 3 proxies (formatted):")
        for i, p in enumerate(proxies[:3], 1):
            # Mask password for display
            display = p.replace(p.split('@')[0].split(':')[-1], '****') if '@' in p else p
            print(f"  {i}. {display}")
    
    configure_proxy_pool(proxies)
    
    print("\n✓ Proxies configured!")
    print("\nNow you can run:")
