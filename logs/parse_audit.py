#!/usr/bin/env python3
"""Parse auditd logs for KSD project activity.

Reads ausearch output, filters for ksd_activity/claude_execution/python_execution
keys, and outputs structured JSON records to audit_parsed.jsonl.

Usage:
    sudo python3 parse_audit.py --since recent
    sudo python3 parse_audit.py --since today
    sudo python3 parse_audit.py --since this-week
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = LOG_DIR / "audit_parsed.jsonl"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
)
logger = logging.getLogger(__name__)

KEYS = ["ksd_activity", "claude_execution", "python_execution"]


def run_ausearch(since: str, key: str) -> str:
    """Run ausearch for a specific key and return stdout.

    ausearch --start accepts: 'today', 'recent' (last 10 min),
    'this-week', 'this-month', 'this-year', or 'MM/DD/YYYY HH:MM:SS'.
    """
    try:
        cmd = ["sudo", "ausearch", "-k", key, "--raw"]
        # Only add --start if the value is a recognized ausearch keyword
        # or date format. Skip for values like "1 hour ago" which aren't valid.
        valid_keywords = {"today", "recent", "this-week", "this-month", "this-year", "boot", "checkpoint"}
        if since in valid_keywords:
            cmd.extend(["--start", since])
        elif re.match(r"\d{2}/\d{2}/\d{4}", since):
            # MM/DD/YYYY format, possibly with HH:MM:SS
            cmd.extend(["--start", since])
        # Otherwise, skip --start and search all records

        logger.debug("Running: %s", " ".join(cmd))
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            stderr = result.stderr.strip()
            if "no matches" in stderr.lower():
                logger.info("ausearch: no matches for key=%s", key)
            elif stderr:
                logger.warning("ausearch stderr for key=%s: %s", key, stderr)
            else:
                logger.warning("ausearch exited %d with no stderr for key=%s", result.returncode, key)
        elif result.stdout:
            logger.info("ausearch returned %d bytes for key=%s", len(result.stdout), key)
        return result.stdout
    except subprocess.TimeoutExpired:
        logger.warning("ausearch timed out for key=%s", key)
        return ""
    except FileNotFoundError:
        logger.error("ausearch not found — is auditd installed?")
        return ""
    except Exception as exc:
        logger.warning("ausearch failed for key=%s: %s", key, exc)
        return ""


def parse_ausearch_raw(text: str, key: str) -> list[dict]:
    """Parse ausearch --raw output into structured records.

    Raw format: each line is a separate audit record. Lines sharing the
    same serial number (in msg=audit(EPOCH:SERIAL)) belong to the same
    event. We group by serial, then extract fields from the combined lines.
    """
    # Group lines by audit event serial number
    events: dict[str, list[str]] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        serial_match = re.search(r"msg=audit\(\d+\.\d+:(\d+)\)", line)
        if serial_match:
            serial = serial_match.group(1)
            events.setdefault(serial, []).append(line)

    records = []
    seen_serials: set[str] = set()

    for serial, lines in events.items():
        if serial in seen_serials:
            continue
        seen_serials.add(serial)

        block = "\n".join(lines)

        # Skip CONFIG_CHANGE-only events (rule add/remove, not real activity)
        line_types = {re.match(r"type=(\w+)", l).group(1).upper() for l in lines if re.match(r"type=(\w+)", l)}
        if line_types <= {"CONFIG_CHANGE"}:
            continue

        record = {
            "source": "auditd",
            "audit_key": key,
            "timestamp": None,
            "event_type": None,
            "file_path": None,
            "command": None,
            "pid": None,
            "uid": None,
            "success": None,
        }

        # Extract timestamp from first line
        ts_match = re.search(r"msg=audit\((\d+\.\d+):\d+\)", lines[0])
        if ts_match:
            epoch = float(ts_match.group(1))
            record["timestamp"] = datetime.fromtimestamp(epoch, tz=UTC).isoformat()

        # Extract syscall name from SYSCALL line
        for line in lines:
            if line.startswith("type=SYSCALL"):
                syscall_match = re.search(r" SYSCALL=(\w+)", line)
                if syscall_match:
                    record["event_type"] = syscall_match.group(1).lower()
                else:
                    record["event_type"] = "syscall"

                # Extract comm (command name)
                comm_match = re.search(r' comm="([^"]+)"', line)
                if comm_match:
                    record["command"] = comm_match.group(1)

                # Extract exe (full path)
                exe_match = re.search(r' exe="([^"]+)"', line)
                if exe_match:
                    record["command"] = exe_match.group(1)

                pid_match = re.search(r" pid=(\d+)", line)
                if pid_match:
                    record["pid"] = int(pid_match.group(1))

                uid_match = re.search(r" auid=(\d+)", line)
                if uid_match:
                    record["uid"] = int(uid_match.group(1))

                success_match = re.search(r" success=(\w+)", line)
                if success_match:
                    record["success"] = success_match.group(1) == "yes"
                break

        # Extract file path from PATH lines (get the most specific one)
        for line in lines:
            if line.startswith("type=PATH"):
                name_match = re.search(r' name="([^"]+)"', line)
                nametype_match = re.search(r" nametype=(\w+)", line)
                if name_match:
                    name = name_match.group(1)
                    nametype = nametype_match.group(1) if nametype_match else ""
                    # Prefer NORMAL/CREATE over PARENT
                    if nametype in ("NORMAL", "CREATE") or record["file_path"] is None:
                        record["file_path"] = name

        # Fall back to event_type from first line type
        if not record["event_type"]:
            type_match = re.search(r"type=(\w+)", lines[0])
            if type_match:
                record["event_type"] = type_match.group(1).lower()

        if record["timestamp"]:
            records.append(record)

    return records


def main():
    parser = argparse.ArgumentParser(description="Parse auditd logs for KSD activity")
    parser.add_argument(
        "--since",
        default="today",
        help="Time range for ausearch: today, recent, this-week, "
        "this-month, this-year, or MM/DD/YYYY (default: today)",
    )
    parser.add_argument(
        "--output",
        default=str(OUTPUT_FILE),
        help=f"Output JSONL file (default: {OUTPUT_FILE})",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with open(output_path, "a") as f:
        for key in KEYS:
            logger.info("Querying ausearch for key=%s since=%s", key, args.since)
            text = run_ausearch(args.since, key)
            if not text:
                logger.info("No records for key=%s", key)
                continue

            records = parse_ausearch_raw(text, key)
            for rec in records:
                f.write(json.dumps(rec) + "\n")
                total += 1

    logger.info("Wrote %d records to %s", total, output_path)


if __name__ == "__main__":
    main()
