#!/bin/bash
# Tune host swappiness to minimize swap thrashing
# Linux default is 60, which aggressively swaps out page cache.
# Setting to 1 ensures swap is only used as an absolute last resort.
# Setting vfs_cache_pressure to 50 encourages holding onto directory caches.

if [ "$EUID" -ne 0 ]; then
  echo "Please run as root (or with sudo)."
  exit 1
fi

echo "Setting vm.swappiness to 1..."
sysctl vm.swappiness=1

echo "Setting vm.vfs_cache_pressure to 50..."
sysctl vm.vfs_cache_pressure=50

echo "Persisting settings to /etc/sysctl.d/99-proxmox-autoscaler.conf..."
cat << 'SYSCTL_CONF' > /etc/sysctl.d/99-proxmox-autoscaler.conf
vm.swappiness=1
vm.vfs_cache_pressure=50
SYSCTL_CONF

echo "Done. Changes applied immediately and will persist across reboots."
