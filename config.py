import os
import json
import logging
from dotenv import load_dotenv

load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("config")

# Proxmox Credentials
PROXMOX_HOST = os.getenv("PROXMOX_HOST", "127.0.0.1")
PROXMOX_USER = os.getenv("PROXMOX_USER", "root@pam")
PROXMOX_TOKEN_ID = os.getenv("PROXMOX_TOKEN_ID", "autoscaler")
PROXMOX_TOKEN_SECRET = os.getenv("PROXMOX_TOKEN_SECRET", "")
NODE_NAME = os.getenv("NODE_NAME", "pve")

# Scaling Settings
MAX_HOST_CPU_ALLOCATION_PERCENT = float(os.getenv("MAX_HOST_CPU_ALLOCATION_PERCENT", 85.0))
MAX_HOST_RAM_ALLOCATION_PERCENT = float(os.getenv("MAX_HOST_RAM_ALLOCATION_PERCENT", 85.0))

# Initial Baselines
try:
    INITIAL_LXC_CONFIGS_STR = os.getenv("INITIAL_LXC_CONFIGS", "{}")
    INITIAL_LXC_CONFIGS = json.loads(INITIAL_LXC_CONFIGS_STR)
except json.JSONDecodeError:
    logger.error("Failed to parse INITIAL_LXC_CONFIGS from .env. Must be valid JSON.")
    INITIAL_LXC_CONFIGS = {}

DATABASE_PATH = os.getenv("DATABASE_PATH", "autoscaler.db")
POLL_INTERVAL_SECONDS = 60
