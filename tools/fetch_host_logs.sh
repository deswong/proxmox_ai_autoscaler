#!/bin/bash
# Helper script to collect Proxmox crash diagnostic logs from target host.
# Usage: ./tools/fetch_host_logs.sh [user@host]

TARGET_HOST="${1:-root@192.168.1.11}"

echo "============================================================"
echo " Fetching Proxmox Crash Diagnostics from ${TARGET_HOST}..."
echo "============================================================"

ssh "${TARGET_HOST}" '
echo "=== 1. HOST MEMORY & SWAP STATUS ==="
free -h
echo ""
echo "=== 2. KERNEL PANICS & ERRORS (PREVIOUS BOOT) ==="
journalctl -k -b -1 -p err..emerg --no-pager | tail -n 60
echo ""
echo "=== 3. OOM KILLER EVENTS (PREVIOUS BOOT) ==="
journalctl -b -1 | grep -i -E "oom|out of memory|killed process" | tail -n 60
echo ""
echo "=== 4. PROXMOX PVE DAEMON LOGS ==="
journalctl -b -1 -u pvedaemon -u pveproxy -u pvestatd -u pve-cluster --no-pager | tail -n 60
echo ""
echo "=== 5. HA & COROSYNC FENCING LOGS ==="
journalctl -b -1 -u corosync -u pve-ha-crm -u pve-ha-lrm --no-pager | grep -i -E "watchdog|fenced|lost contact|quorum" | tail -n 60
echo ""
echo "=== 6. RECENT PROXMOX TASK HISTORY ==="
pvesh get /nodes/$(hostname)/tasks --limit 20
'
