#!/bin/bash
# Proxmox AI Autoscaler 🚀
# Uninstaller Script
# Usage: curl -sL https://raw.githubusercontent.com/deswong/proxmox_ai_autoscaler/main/uninstall.sh | bash

set -e

echo "========================================="
echo " Proxmox Universal AI Autoscaler Removal "
echo "========================================="

if [ "$EUID" -ne 0 ]; then
  echo "❌ Error: This script must be run as root (or with sudo)."
  exit 1
fi

APP_DIR="/opt/proxmox-ai-autoscaler"
SERVICE_NAME="proxmox-ai-autoscaler"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
LOG_FILE="/var/log/proxmox_ai_autoscaler.log"

# 1. Stop and Disable Systemd Service
echo "🛑 Stopping and disabling $SERVICE_NAME service..."
if systemctl is-active --quiet "$SERVICE_NAME" 2>/dev/null; then
    systemctl stop "$SERVICE_NAME"
fi
if systemctl is-enabled --quiet "$SERVICE_NAME" 2>/dev/null; then
    systemctl disable "$SERVICE_NAME"
fi

# 2. Remove Systemd Service File
if [ -f "$SERVICE_FILE" ]; then
    echo "🗑️ Removing systemd service file..."
    rm -f "$SERVICE_FILE"
    systemctl daemon-reload
fi

# 3. Remove Cron Job
echo "🕒 Removing nightly XGBoost batch training cron job..."
# Backup existing cron, filter out our specific job, and install
if crontab -l 2>/dev/null | grep -q "${APP_DIR}/train_models.py"; then
    crontab -l 2>/dev/null | grep -v "${APP_DIR}/train_models.py" | crontab - || true
    echo "✅ Cron job removed."
else
    echo "✅ No cron job found."
fi

# 4. Remove App Directory
if [ -d "$APP_DIR" ]; then
    echo "🗑️ Removing application directory at $APP_DIR..."
    rm -rf "$APP_DIR"
fi

# 5. Remove Log File
if [ -f "$LOG_FILE" ]; then
    echo "🗑️ Removing log file at $LOG_FILE..."
    rm -f "$LOG_FILE"
fi

echo ""
echo "✅ Uninstallation Complete!"
echo "The Proxmox AI Autoscaler has been completely removed from your system."
