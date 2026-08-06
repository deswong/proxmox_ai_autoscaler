import logging
import time
from proxmoxer import ProxmoxAPI
import urllib3
from config import (
    PROXMOX_HOST,
    PROXMOX_USER,
    PROXMOX_TOKEN_ID,
    PROXMOX_TOKEN_SECRET,
    NODE_NAME,
)

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
                verify_ssl=False,
                timeout=30,
            )
            self.node = self.proxmox.nodes(NODE_NAME)
            logger.info(
                f"Successfully connected to Proxmox node {NODE_NAME} at {PROXMOX_HOST}"
            )
        except Exception as e:
            logger.error(f"Failed to connect to Proxmox API: {e}")
            self.proxmox = None

    def get_host_usage(self) -> dict:
        """
        Fetches CPU, RAM, swap, load averages, KSM activity, and physical CPU
        count from the host node. All data from a single node.status.get() call.
        """
        _empty = {
            "cpu_percent": 0.0, "ram_percent": 0.0, "swap_percent": 0.0,
            "total_ram_mb": 0.0, "physical_cpus": 1,
            "load_avg_1m": 0.0, "load_avg_5m": 0.0, "ksm_sharing_mb": 0.0,
        }
        if not self.proxmox:
            return _empty

        try:
            status = self.node.status.get()

            # memory
            mem_total = status.get("memory", {}).get("total", 0)
            mem_used = status.get("memory", {}).get("used", 0)
            ram_percent = (mem_used / mem_total * 100) if mem_total > 0 else 0

            # cpu
            cpu_percent = status.get("cpu", 0) * 100

            # swap
            swap_total = status.get("swap", {}).get("total", 0)
            swap_used = status.get("swap", {}).get("used", 0)
            swap_percent = (swap_used / swap_total * 100) if swap_total > 0 else 0

            # load averages (1-min, 5-min) — list of strings
            loadavg = status.get("loadavg", ["0", "0", "0"])
            load_1m = float(loadavg[0]) if len(loadavg) > 0 else 0.0
            load_5m = float(loadavg[1]) if len(loadavg) > 1 else 0.0

            # KSM — bytes of memory currently shared by Kernel Same-Page Merging
            ksm_sharing_mb = float(
                status.get("ksm", {}).get("shared", 0) / (1024 * 1024)
            )

            # Physical CPU count for overcommit ratio computation
            physical_cpus = int(status.get("cpuinfo", {}).get("cpus", 1)) or 1

            return {
                "cpu_percent": float(cpu_percent),
                "ram_percent": float(ram_percent),
                "swap_percent": float(swap_percent),
                "total_ram_mb": float(mem_total / (1024 * 1024)),
                "physical_cpus": physical_cpus,
                "load_avg_1m": load_1m,
                "load_avg_5m": load_5m,
                "ksm_sharing_mb": ksm_sharing_mb,
            }
        except Exception as e:
            logger.error(f"Failed to fetch host usage: {e}")
            return _empty


    def update_lxc_resources(self, lxc_id: str, cpus: int, ram_mb: int, swap_mb: int = 0):
        """Updates the CPU cores and RAM allocation of a running LXC."""
        if not self.proxmox:
            return False

        max_retries = 3

        for attempt in range(max_retries):
            try:
                # Proxmox config API expects memory and swap in MB
                self.node.lxc(lxc_id).config.put(
                    cores=int(cpus), memory=int(ram_mb), swap=int(swap_mb)
                )
                logger.info(
                    f"[LXC {lxc_id}] Successfully hotplugged resources: {cpus} cores, {ram_mb} MB RAM"
                )
                return True
            except Exception as e:
                logger.error(
                    f"Failed to update resources for LXC {lxc_id} (Attempt {attempt + 1}/{max_retries}): {e}"
                )
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)
                else:
                    return False

    def get_lxc_rrd_history(self, lxc_id: str, timeframe: str = "hour") -> list:
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
        Returns a list of all LXC IDs on the node, regardless of running status.
        """
        if not self.proxmox:
            return []

        try:
            lxcs = self.node.lxc.get()
            return [str(lxc["vmid"]) for lxc in lxcs]
        except Exception as e:
            logger.error(f"Failed to fetch LXC IDs from node {NODE_NAME}: {e}")
            return []

    def get_entity_notes(self, entity_id: str, entity_type: str) -> str:
        """
        Retrieves the 'description' field (Notes in UI) for an LXC or VM.
        """
        if not self.proxmox:
            return ""

        try:
            if entity_type == "LXC":
                config = self.node.lxc(entity_id).config.get()
            else:
                config = self.node.qemu(entity_id).config.get()

            return config.get("description", "")
        except Exception as e:
            logger.error(f"Failed to fetch notes for {entity_type} {entity_id}: {e}")
            return ""

    def set_entity_notes(self, entity_id: str, entity_type: str, notes: str) -> bool:
        """
        Updates the 'description' field (Notes in UI) for an LXC or VM.
        """
        if not self.proxmox:
            return False

        try:
            if entity_type == "LXC":
                self.node.lxc(entity_id).config.put(description=notes)
            else:
                self.node.qemu(entity_id).config.put(description=notes)
            return True
        except Exception as e:
            logger.error(f"Failed to set notes for {entity_type} {entity_id}: {e}")
            return False

    def get_all_lxc_metrics(self) -> dict:
        """
        Returns a dictionary mapping LXC IDs to their parsed current telemetry.
        Eliminates the need to sequentially API ping every single LXC for baseline status.
        """
        if not self.proxmox:
            return {}

        try:
            lxcs = self.node.lxc.get()
            metrics_dict = {}
            for lxc in lxcs:
                if lxc.get("status") != "running":
                    continue

                vmid = str(lxc["vmid"])
                metrics_dict[vmid] = {
                    "cpu_percent": float(lxc.get("cpu", 0) * 100),
                    "ram_usage_mb": float(lxc.get("mem", 0) / (1024 * 1024)),
                    "allocated_cpus": int(lxc.get("cpus", 1)),
                    "allocated_ram_mb": float(lxc.get("maxmem", 0) / (1024 * 1024)),
                    "swap_mb": float(lxc.get("swap", 0) / (1024 * 1024)),
                    "allocated_swap_mb": float(lxc.get("maxswap", 0) / (1024 * 1024)),
                    "uptime": int(lxc.get("uptime", 0)),
                    "disk_read_bps": float(lxc.get("diskread", 0)),
                    "disk_write_bps": float(lxc.get("diskwrite", 0)),
                    "net_in_bps": float(lxc.get("netin", 0)),
                    "net_out_bps": float(lxc.get("netout", 0)),
                }
            return metrics_dict
        except Exception as e:
            logger.error(
                f"Failed to fetch bulk telemetry for LXCs from node {NODE_NAME}: {e}"
            )
            return {}

    def update_vm_resources(
        self, vm_id: str, cpus: int, ram_mb: int, sockets: int = None
    ):
        """Updates the CPU cores, RAM allocation, and optionally sockets of a VM."""
        if not self.proxmox:
            return False

        max_retries = 3

        for attempt in range(max_retries):
            try:
                # Requires Hotplug to be enabled in Proxmox VM Hardware configs
                kwargs = {"cores": int(cpus), "memory": int(ram_mb)}
                if sockets is not None:
                    kwargs["sockets"] = int(sockets)
                self.node.qemu(vm_id).config.put(**kwargs)
                log_msg = f"[VM {vm_id}] Successfully updated resources: {cpus} cores, {ram_mb} MB RAM"
                if sockets is not None:
                    log_msg += f", {sockets} sockets"
                logger.info(log_msg)
                return True
            except Exception as e:
                logger.error(
                    f"Failed to update resources for VM {vm_id} (Attempt {attempt + 1}/{max_retries}): {e}"
                )
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)
                else:
                    return False

    def get_vm_config(self, vm_id: str) -> dict:
        """
        Retrieves the exact configuration (CPU cores, RAM, sockets) of a VM, checking both active
        and pending settings to avoid redundant API write loops.
        """
        if not self.proxmox:
            return {}
        try:
            config = self.node.qemu(vm_id).config.get()
            cpus = int(config.get("cores", 1))
            ram_mb = int(config.get("memory", 512))
            sockets = int(config.get("sockets", 1))

            # Query pending configuration changes if available
            try:
                pending_items = self.node.qemu(vm_id).pending.get()
                for item in pending_items:
                    key = item.get("key")
                    if "pending" in item and item["pending"] is not None:
                        if key == "cores":
                            cpus = int(item["pending"])
                        elif key == "memory":
                            ram_mb = int(item["pending"])
                        elif key == "sockets":
                            sockets = int(item["pending"])
            except Exception as pending_err:
                logger.debug(f"[VM {vm_id}] Could not query pending config: {pending_err}")

            return {
                "cpus": cpus,
                "ram_mb": ram_mb,
                "sockets": sockets,
            }
        except Exception as e:
            logger.error(f"Failed to fetch config for VM {vm_id}: {e}")
            return {}

    def get_vm_rrd_history(self, vm_id: str, timeframe: str = "hour") -> list:
        """
        Fetches the RRD historical graph data for a VM.
        """
        if not self.proxmox:
            return []

        try:
            rrd_data = self.node.qemu(vm_id).rrddata.get(timeframe=timeframe)
            return rrd_data
        except Exception as e:
            logger.error(f"Failed to fetch RRD history for VM {vm_id}: {e}")
            return []

    def get_node_rrd_history(self, timeframe: str = "hour") -> list:
        """
        Fetches the RRD historical graph data for the host node itself.
        Returns a list of dicts with keys: time, cpu, memtotal, memused,
        swaptotal, swapused — used by the trainer to correlate host pressure
        with each container's training window.
        """
        if not self.proxmox:
            return []

        try:
            rrd_data = self.node.rrddata.get(timeframe=timeframe)
            return rrd_data
        except Exception as e:
            logger.error(f"Failed to fetch RRD history for host node: {e}")
            return []

    def get_all_vm_ids(self) -> list:
        """
        Returns a list of all VM IDs on the node, regardless of running status.
        """
        if not self.proxmox:
            return []

        try:
            vms = self.node.qemu.get()
            return [str(vm["vmid"]) for vm in vms]
        except Exception as e:
            logger.error(f"Failed to fetch VM IDs from node {NODE_NAME}: {e}")
            return []

    def get_all_committed_resources(self) -> dict:
        """
        Returns the total CPU cores and RAM *configured* across ALL LXCs and VMs
        on the node, regardless of whether they are currently running or stopped.

        This represents the worst-case scenario: if every entity were started
        simultaneously the host would need to service all of these allocations.
        The autoscaler uses this to ensure it never proposes a scale-up that would
        make the host unable to start all tenants at once.

        Returns:
            {
                "committed_cpus":     int    — sum of configured cores (all entities)
                "committed_ram_mb":   float  — sum of configured RAM MB (all entities)
                "total_entity_count": int    — total number of LXCs + VMs (any state)
                "stopped_entity_count": int  — entities not currently running
            }
        """
        _empty = {
            "committed_cpus": 0,
            "committed_ram_mb": 0.0,
            "total_entity_count": 0,
            "stopped_entity_count": 0,
        }
        if not self.proxmox:
            return _empty

        total_cpus = 0
        total_ram_mb = 0.0
        total_count = 0
        stopped_count = 0

        try:
            lxcs = self.node.lxc.get()
            for lxc in lxcs:
                total_cpus += int(lxc.get("cpus", 0))
                total_ram_mb += float(lxc.get("maxmem", 0) / (1024 * 1024))
                total_count += 1
                if lxc.get("status") != "running":
                    stopped_count += 1
        except Exception as e:
            logger.error(f"Failed to fetch committed LXC resources: {e}")

        try:
            vms = self.node.qemu.get()
            for vm in vms:
                total_cpus += int(vm.get("cpus", 0))
                total_ram_mb += float(vm.get("maxmem", 0) / (1024 * 1024))
                total_count += 1
                if vm.get("status") != "running":
                    stopped_count += 1
        except Exception as e:
            logger.error(f"Failed to fetch committed VM resources: {e}")

        return {
            "committed_cpus":       total_cpus,
            "committed_ram_mb":     total_ram_mb,
            "total_entity_count":   total_count,
            "stopped_entity_count": stopped_count,
        }

    def get_all_vm_metrics(self) -> dict:
        """
        Returns a dictionary mapping VM IDs to their parsed current telemetry.
        """
        if not self.proxmox:
            return {}

        try:
            vms = self.node.qemu.get()
            metrics_dict = {}
            for vm in vms:
                if vm.get("status") != "running":
                    continue

                vmid = str(vm["vmid"])
                metrics_dict[vmid] = {
                    "cpu_percent": float(vm.get("cpu", 0) * 100),
                    "ram_usage_mb": float(vm.get("mem", 0) / (1024 * 1024)),
                    "allocated_cpus": int(vm.get("cpus", 1)),
                    "allocated_ram_mb": float(vm.get("maxmem", 0) / (1024 * 1024)),
                    "uptime": int(vm.get("uptime", 0)),
                    "disk_read_bps": float(vm.get("diskread", 0)),
                    "disk_write_bps": float(vm.get("diskwrite", 0)),
                    "net_in_bps": float(vm.get("netin", 0)),
                    "net_out_bps": float(vm.get("netout", 0)),
                }
            return metrics_dict
        except Exception as e:
            logger.error(
                f"Failed to fetch bulk telemetry for VMs from node {NODE_NAME}: {e}"
            )
            return {}
