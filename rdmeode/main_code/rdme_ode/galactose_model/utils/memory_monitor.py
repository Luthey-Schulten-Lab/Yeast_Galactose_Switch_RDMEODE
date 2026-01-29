"""Memory monitoring utilities"""

import os
import psutil


def print_memory_usage():
    """Print current memory usage of the process"""
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"Memory usage: {memory_mb:.2f} MB")
    return memory_mb
