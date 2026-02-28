import logging
import urllib.parse
from proxmoxer import ProxmoxAPI
import urllib3
from config import PROXMOX_HOST, PROXMOX_USER, PROXMOX_TOKEN_ID, PROXMOX_TOKEN_SECRET, NODE_NAME

# Suppress insecure request warnings if Proxmox uses self-signed certs
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger("proxmox_api")

class ProxmoxClient:
    def __init__(self):
        try:
            self.proxmox = ProxmoxAPI(
                PROXMOX_HOST,
                user=PROXMOX_USER,
                token_name=PROXMOX_TOKEN_ID,
                token_value=PROXMOX_TOKEN_SECRET,
                verify_ssl=False
            )
            self.node = self.proxmox.nodes(NODE_NAME)
            logger.info(f"Successfully connected to Proxmox node {NODE_NAME} at {PROXMOX_HOST}")
        except Exception as e:
            logger.error(f"Failed to connect to Proxmox API: {e}")
            self.proxmox = None

    def get_host_usage(self) -> dict:
        """Fetches the current CPU and RAM usage of the host node."""
        if not self.proxmox:
            return {"cpu_percent": 0.0, "ram_percent": 0.0, "total_ram_mb": 0.0}

        try:
            status = self.node.status.get()
            
            # memory
            mem_total = status.get('memory', {}).get('total', 0)
            mem_used = status.get('memory', {}).get('used', 0)
            ram_percent = (mem_used / mem_total * 100) if mem_total > 0 else 0
            
            # cpu
            cpu_percent = status.get('cpu', 0) * 100
            
            return {
                "cpu_percent": float(cpu_percent),
                "ram_percent": float(ram_percent),
                "total_ram_mb": mem_total / (1024 * 1024)
            }
        except Exception as e:
            logger.error(f"Failed to fetch host usage: {e}")
            return {"cpu_percent": 0.0, "ram_percent": 0.0, "total_ram_mb": 0.0}

    def get_lxc_metrics(self, lxc_id: str) -> dict:
        """Fetches the current CPU and RAM metric usage for a specific LXC."""
        if not self.proxmox:
            return None
            
        try:
            status = self.node.lxc(lxc_id).status.current.get()
            
            if status.get("status") != "running":
                return None
            
            # CPU ratio from Proxmox
            cpu = status.get("cpu", 0) * 100
            
            # Memory bytes converted to MB
            mem_bytes = status.get("mem", 0)
            mem_mb = mem_bytes / (1024 * 1024)
            
            # Configured limits
            maxmem_bytes = status.get("maxmem", 0)
            maxmem_mb = maxmem_bytes / (1024 * 1024)
            cpus_allocated = status.get("cpus", 1)
            
            return {
                "cpu_percent": float(cpu),
                "ram_usage_mb": float(mem_mb),
                "allocated_cpus": int(cpus_allocated),
                "allocated_ram_mb": float(maxmem_mb)
            }
        except Exception as e:
            logger.error(f"Failed to fetch metrics for LXC {lxc_id}: {e}")
            return None

    def update_lxc_resources(self, lxc_id: str, cpus: int, ram_mb: int):
        """Updates the CPU cores and RAM allocation of a running LXC."""
        if not self.proxmox:
            return False
            
        try:
            # We hotplug the CPU and Memory via the API.
            # Proxmox config API expects memory in MB
            self.node.lxc(lxc_id).config.put(
                cores=int(cpus),
                memory=int(ram_mb)
            )
            logger.info(f"[LXC {lxc_id}] Successfully hotplugged resources: {cpus} cores, {ram_mb} MB RAM")
            return True
        except Exception as e:
            logger.error(f"Failed to update resources for LXC {lxc_id}: {e}")
            return False

    def get_lxc_rrd_history(self, lxc_id: str, timeframe: str = 'hour') -> list:
        """
        Fetches the RRD historical graph data for an LXC.
        timeframe can be: hour, day, week, month, year.
        Returns a list of dicts: [{'time': UNIX_EPOCH, 'cpu': 0.05, 'mem': bytes, ...}]
        """
        if not self.proxmox:
            return []
            
        try:
            # Proxmox API returns an array of data points for the given timeframe.
            rrd_data = self.node.lxc(lxc_id).rrddata.get(timeframe=timeframe)
            return rrd_data
        except Exception as e:
            logger.error(f"Failed to fetch RRD history for LXC {lxc_id}: {e}")
            return []

    def get_all_lxc_ids(self) -> list:
        """
        Returns a list of all LXC IDs currently existing on the target Proxmox node.
        """
        if not self.proxmox:
            return []
            
        try:
            lxcs = self.node.lxc.get()
            return [str(lxc['vmid']) for lxc in lxcs]
        except Exception as e:
            logger.error(f"Failed to fetch list of LXCs from node {NODE_NAME}: {e}")
            return []
