#!/usr/bin/env python3
"""
System Resource Monitor for Edge Devices

Monitors and publishes system metrics critical for edge deployment:
- CPU usage (per core and total)
- Memory usage
- Temperature (if available)
- Disk usage
- Network I/O

This helps identify bottlenecks and tune edge deployments.
"""

import os
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32, Float32MultiArray

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class ResourceMonitor(Node):
    """
    System resource monitor for edge deployments.

    Publishes:
    - /edge/cpu_total (Float32): Total CPU percentage
    - /edge/cpu_per_core (Float32MultiArray): Per-core CPU
    - /edge/memory_percent (Float32): Memory usage
    - /edge/temperature (Float32): CPU temperature (if available)
    - /edge/diagnostics (String): Human-readable summary
    """

    def __init__(self):
        super().__init__('resource_monitor')

        if not PSUTIL_AVAILABLE:
            self.get_logger().error('psutil not available - install with: pip install psutil')
            return

        # Parameters
        self.declare_parameter('monitor_rate', 1.0)  # Hz
        self.declare_parameter('warn_cpu_threshold', 80.0)
        self.declare_parameter('warn_memory_threshold', 80.0)
        self.declare_parameter('warn_temp_threshold', 70.0)

        rate = self.get_parameter('monitor_rate').value
        self.cpu_warn = self.get_parameter('warn_cpu_threshold').value
        self.mem_warn = self.get_parameter('warn_memory_threshold').value
        self.temp_warn = self.get_parameter('warn_temp_threshold').value

        # Publishers
        self.cpu_total_pub = self.create_publisher(Float32, 'edge/cpu_total', 10)
        self.cpu_cores_pub = self.create_publisher(Float32MultiArray, 'edge/cpu_per_core', 10)
        self.memory_pub = self.create_publisher(Float32, 'edge/memory_percent', 10)
        self.temp_pub = self.create_publisher(Float32, 'edge/temperature', 10)
        self.diag_pub = self.create_publisher(String, 'edge/diagnostics', 10)

        # Preallocated messages
        self._float_msg = Float32()
        self._array_msg = Float32MultiArray()
        self._string_msg = String()

        # Timer
        self.timer = self.create_timer(1.0 / rate, self.monitor_callback)

        # Initialize psutil CPU measurement
        psutil.cpu_percent(percpu=True)

        self.get_logger().info(f'Resource monitor started at {rate} Hz')
        self.get_logger().info(f'Thresholds - CPU: {self.cpu_warn}%, MEM: {self.mem_warn}%, TEMP: {self.temp_warn}C')

    def monitor_callback(self):
        """Gather and publish system metrics."""
        warnings = []

        # CPU Usage
        cpu_total = psutil.cpu_percent(interval=None)
        cpu_per_core = psutil.cpu_percent(interval=None, percpu=True)

        self._float_msg.data = cpu_total
        self.cpu_total_pub.publish(self._float_msg)

        self._array_msg.data = cpu_per_core
        self.cpu_cores_pub.publish(self._array_msg)

        if cpu_total > self.cpu_warn:
            warnings.append(f'HIGH_CPU:{cpu_total:.0f}%')

        # Memory Usage
        mem = psutil.virtual_memory()
        self._float_msg.data = mem.percent
        self.memory_pub.publish(self._float_msg)

        if mem.percent > self.mem_warn:
            warnings.append(f'HIGH_MEM:{mem.percent:.0f}%')

        # Temperature (platform-dependent)
        temp = self._get_temperature()
        if temp is not None:
            self._float_msg.data = temp
            self.temp_pub.publish(self._float_msg)

            if temp > self.temp_warn:
                warnings.append(f'HIGH_TEMP:{temp:.0f}C')

        # Build diagnostics message
        diag_parts = [
            f'CPU:{cpu_total:.0f}%',
            f'MEM:{mem.percent:.0f}%({mem.used // (1024*1024)}MB)',
        ]

        if temp is not None:
            diag_parts.append(f'TEMP:{temp:.0f}C')

        if warnings:
            diag_parts.append(f'WARN:[{",".join(warnings)}]')

        self._string_msg.data = '|'.join(diag_parts)
        self.diag_pub.publish(self._string_msg)

        # Log warnings
        for warn in warnings:
            self.get_logger().warn(warn)

    def _get_temperature(self) -> Optional[float]:
        """
        Get CPU temperature (platform-specific).

        Works on:
        - Linux with thermal sensors
        - Raspberry Pi
        - NVIDIA Jetson
        """
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                # Try common sensor names
                for name in ['coretemp', 'cpu_thermal', 'cpu-thermal', 'k10temp', 'acpitz']:
                    if name in temps:
                        return temps[name][0].current
                # Fall back to first available
                first_sensor = list(temps.values())[0]
                if first_sensor:
                    return first_sensor[0].current
        except (AttributeError, KeyError, IndexError):
            pass

        # Try Raspberry Pi thermal zone
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                return float(f.read().strip()) / 1000.0
        except (FileNotFoundError, ValueError, PermissionError):
            pass

        # Try NVIDIA Jetson
        try:
            with open('/sys/devices/virtual/thermal/thermal_zone0/temp', 'r') as f:
                return float(f.read().strip()) / 1000.0
        except (FileNotFoundError, ValueError, PermissionError):
            pass

        return None


def main(args=None):
    rclpy.init(args=args)

    if not PSUTIL_AVAILABLE:
        print('Error: psutil required. Install with: pip install psutil')
        return

    node = ResourceMonitor()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
