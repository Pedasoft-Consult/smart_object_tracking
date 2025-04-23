#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Connectivity utilities.
Checks network connection status and switches between online/offline modes.
"""

import socket
import time
import logging
import threading
import http.client
import urllib.parse
import subprocess
import platform
import os

# Set up logger
logger = logging.getLogger("Connectivity")


def check_connectivity(test_urls=None, timeout=5, max_retries=2):
    """
    Check if device has internet connectivity

    Args:
        test_urls: List of URLs to test (default: ['google.com', 'microsoft.com', 'cloudflare.com'])
        timeout: Connection timeout in seconds
        max_retries: Maximum number of retry attempts

    Returns:
        True if connected, False otherwise
    """
    if test_urls is None:
        test_urls = ['google.com', 'microsoft.com', 'cloudflare.com']

    # First, try socket connection to multiple hosts
    for retry in range(max_retries):
        for url in test_urls:
            try:
                # Try to resolve hostname
                socket.gethostbyname(url)

                # Try to establish connection
                conn = http.client.HTTPSConnection(url, timeout=timeout)
                conn.request("HEAD", "/")
                response = conn.getresponse()
                conn.close()

                if 200 <= response.status < 400:
                    logger.debug(f"Successfully connected to {url}")
                    return True

            except Exception as e:
                logger.debug(f"Connection to {url} failed: {e}")
                continue

        # If we reach here, all URLs failed
        if retry < max_retries - 1:
            logger.debug(f"Retrying connectivity check ({retry + 1}/{max_retries})")
            time.sleep(1)

    # If all retries failed, try ping as a fallback
    try:
        return ping_test()
    except Exception as e:
        logger.debug(f"Ping test failed: {e}")
        return False


def ping_test(host="8.8.8.8", count=1, timeout=2):
    """
    Ping a host to check connectivity

    Args:
        host: Host to ping
        count: Number of pings
        timeout: Timeout in seconds

    Returns:
        True if ping successful, False otherwise
    """
    # Choose ping command based on OS
    system = platform.system().lower()

    if system == "windows":
        # Windows
        args = ["ping", "-n", str(count), "-w", str(timeout * 1000), host]
    elif system == "darwin" or system == "linux":
        # MacOS or Linux
        args = ["ping", "-c", str(count), "-W", str(timeout), host]
    else:
        # Unsupported OS
        return False

    try:
        # Run ping command
        result = subprocess.run(
            args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout * 2
        )

        # Check return code (0 = success)
        return result.returncode == 0

    except subprocess.SubprocessError:
        return False


class ConnectivityMonitor:
    """Monitors network connectivity and provides callbacks for status changes"""

    def __init__(self, check_interval=30, online_callback=None, offline_callback=None):
        """
        Initialize connectivity monitor

        Args:
            check_interval: Interval between checks in seconds
            online_callback: Function to call when going online
            offline_callback: Function to call when going offline
        """
        self.check_interval = check_interval
        self.online_callback = online_callback
        self.offline_callback = offline_callback
        self.is_online = False
        self.stop_event = threading.Event()
        self.monitor_thread = None
        self.logger = logging.getLogger("ConnectivityMonitor")

    def start(self):
        """Start monitoring connectivity"""
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.logger.warning("Monitor already running")
            return

        self.stop_event.clear()
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("Connectivity monitor started")

    def stop(self):
        """Stop monitoring connectivity"""
        self.stop_event.set()
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=self.check_interval * 2)
        self.logger.info("Connectivity monitor stopped")

    def _monitor_loop(self):
        """Main monitoring loop"""
        # Initial check
        self.is_online = check_connectivity()
        self.logger.info(f"Initial connectivity status: {'Online' if self.is_online else 'Offline'}")

        while not self.stop_event.is_set():
            try:
                # Wait for interval
                for _ in range(int(self.check_interval * 2)):  # Check more frequently if stopping
                    if self.stop_event.is_set():
                        return
                    time.sleep(0.5)

                # Check connectivity
                current_status = check_connectivity()

                # Detect status change
                if current_status != self.is_online:
                    self.logger.info(f"Connectivity changed: {'Online' if current_status else 'Offline'}")

                    # Update status
                    self.is_online = current_status

                    # Call appropriate callback
                    if current_status and self.online_callback:
                        try:
                            self.online_callback()
                        except Exception as e:
                            self.logger.error(f"Error in online callback: {e}")
                    elif not current_status and self.offline_callback:
                        try:
                            self.offline_callback()
                        except Exception as e:
                            self.logger.error(f"Error in offline callback: {e}")

            except Exception as e:
                self.logger.error(f"Error in connectivity monitoring: {e}")
                time.sleep(1)  # Prevent tight loop on repeated errors


def network_info():
    """
    Get network interface information

    Returns:
        Dictionary with network information
    """
    info = {
        "interfaces": [],
        "hostname": socket.gethostname(),
        "connected": False
    }

    # Check if we can get local IP
    try:
        # Create a temporary socket to get local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        info["local_ip"] = s.getsockname()[0]
        s.close()
    except:
        info["local_ip"] = "127.0.0.1"

    # Get network interfaces
    if platform.system() == "Windows":
        try:
            # Using ipconfig on Windows
            output = subprocess.check_output(["ipconfig", "/all"], text=True)

            # Parse output (simplified)
            current_if = None
            for line in output.splitlines():
                line = line.strip()

                if not line:
                    continue

                if not line.startswith(" "):
                    # New interface section
                    current_if = {"name": line.rstrip(":"), "addresses": []}
                    info["interfaces"].append(current_if)
                elif current_if and "IPv4 Address" in line:
                    # IPv4 address
                    addr = line.split(":")[-1].strip()
                    if addr.endswith("(Preferred)"):
                        addr = addr[:-11].strip()
                    current_if["addresses"].append(addr)
                elif current_if and "Physical Address" in line:
                    # MAC address
                    current_if["mac"] = line.split(":")[-1].strip()

        except Exception as e:
            logger.warning(f"Error getting Windows network info: {e}")

    else:
        try:
            # Using ifconfig on Unix/Linux
            try:
                output = subprocess.check_output(["ifconfig"], text=True)
            except FileNotFoundError:
                # Try using ip on newer systems
                output = subprocess.check_output(["ip", "addr"], text=True)

            # Parse output (simplified)
            current_if = None
            for line in output.splitlines():
                line = line.strip()

                if not line:
                    continue

                if not line.startswith(" ") and not line.startswith("\t"):
                    # New interface section
                    if ":" in line:
                        if_name = line.split(":")[0]
                        current_if = {"name": if_name, "addresses": []}
                        info["interfaces"].append(current_if)
                elif current_if:
                    if "inet " in line:
                        # IPv4 address
                        addr = line.split("inet ")[1].split()[0]
                        current_if["addresses"].append(addr)
                    elif "ether " in line:
                        # MAC address
                        current_if["mac"] = line.split("ether ")[1].split()[0]

        except Exception as e:
            logger.warning(f"Error getting Unix network info: {e}")

    # Check connectivity
    info["connected"] = check_connectivity()

    return info


if __name__ == "__main__":
    """Test connectivity functions"""
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Check connectivity
    print(f"Connected to internet: {check_connectivity()}")

    # Show network info
    print("Network information:")
    info = network_info()
    print(f"Hostname: {info['hostname']}")
    print(f"Local IP: {info.get('local_ip', 'Unknown')}")
    print(f"Internet connected: {info['connected']}")

    print("Network interfaces:")
    for interface in info["interfaces"]:
        print(f"  - {interface['name']}")
        if 'mac' in interface:
            print(f"    MAC: {interface['mac']}")
        if interface['addresses']:
            print(f"    Addresses: {', '.join(interface['addresses'])}")