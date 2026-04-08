#!/usr/bin/env python3
"""Test RunPod SSH connection — try multiple approaches.

Replace the placeholders below with your own pod ID, proxy suffix, direct
TCP host, port, and SSH key path before running.  These values are issued
per-pod by RunPod and should never be checked into a public repository.
"""

import subprocess

pod_id = "REPLACE_WITH_YOUR_POD_ID"
proxy_user = f"{pod_id}-REPLACE_WITH_PROXY_SUFFIX"
direct_host = "REPLACE_WITH_DIRECT_TCP_HOST"
direct_port = "REPLACE_WITH_DIRECT_TCP_PORT"
ssh_key = "REPLACE_WITH_PATH_TO_SSH_PRIVATE_KEY"

tests = [
    # Proxy with explicit key, simple command
    (f"{proxy_user}@ssh.runpod.io", ["-i", ssh_key], "echo SSH_OK"),
    # Direct TCP with explicit key
    (f"root@{direct_host}", ["-p", direct_port, "-i", ssh_key], "echo SSH_OK"),
]

for target, extra_args, cmd in tests:
    full = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=15"] + extra_args + [target, cmd]
    print(f"Testing: {' '.join(full[:8])}... {cmd}")
    try:
        result = subprocess.run(full, capture_output=True, text=True, timeout=20)
        print(f"  exit={result.returncode} out={result.stdout.strip()!r} err={result.stderr.strip()[:100]!r}")
    except subprocess.TimeoutExpired:
        print("  TIMEOUT after 20s")
    print()
