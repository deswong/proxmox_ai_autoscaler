import sqlite3
import json
import time
import logging
from typing import Dict, List, Tuple
from config import DATABASE_PATH, INITIAL_LXC_CONFIGS

logger = logging.getLogger("storage")

def get_db_connection():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes the SQLite database tables."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Table to store configured baselines for each LXC
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS lxc_baselines (
            lxc_id TEXT PRIMARY KEY,
            min_cpus INTEGER,
            min_ram_mb INTEGER,
            max_cpus INTEGER,
            max_ram_mb INTEGER,
            updated_at REAL
        )
    ''')
    
    conn.commit()
    conn.close()
    
    _seed_initial_baselines()

def _seed_initial_baselines():
    """Seeds the database with baselines from the .env file if they don't already exist."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    current_time = time.time()
    for lxc_id, config in INITIAL_LXC_CONFIGS.items():
        cursor.execute("SELECT lxc_id FROM lxc_baselines WHERE lxc_id = ?", (str(lxc_id),))
        if cursor.fetchone() is None:
            logger.info(f"Seeding DB with initial baseline for LXC {lxc_id}: {config}")
            cursor.execute('''
                INSERT INTO lxc_baselines (lxc_id, min_cpus, min_ram_mb, max_cpus, max_ram_mb, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (str(lxc_id), config.get('min_cpus', 1), config.get('min_ram_mb', 512), 
                  config.get('max_cpus', 4), config.get('max_ram_mb', 4096), current_time))
    
    conn.commit()
    conn.close()

def get_baselines() -> Dict[str, Dict]:
    """Retrieves all LXC baselines from the DB."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM lxc_baselines")
    rows = cursor.fetchall()
    
    baselines = {}
    for row in rows:
        baselines[row['lxc_id']] = {
            'min_cpus': row['min_cpus'],
            'min_ram_mb': row['min_ram_mb'],
            'max_cpus': row['max_cpus'],
            'max_ram_mb': row['max_ram_mb']
        }
    
    conn.close()
    return baselines
