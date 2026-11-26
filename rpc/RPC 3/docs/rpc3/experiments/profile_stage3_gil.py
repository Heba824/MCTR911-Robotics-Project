"""Measure FLC wall versus thread CPU latency with a competing Python thread."""

import json
from pathlib import Path
import sys
from threading import Event, Thread

import numpy as np

from panda_kinematics.control_timing import record_timing, timing_start
from panda_kinematics.fuzzy import JointFLC


def measure(interval):
    fuzzy = JointFLC([1]*7, [1]*7, [1]*7)
    error = np.linspace(-1, 1, 7)
    fuzzy(error, -error)
    stop = Event()

    def compete():
        while not stop.is_set():
            for _ in range(200):
                pass

    previous = sys.getswitchinterval()
    sys.setswitchinterval(interval)
    worker = Thread(target=compete)
    worker.start()
    profiles = []
    try:
        for _ in range(300):
            profile = {}
            start = timing_start()
            fuzzy(error, -error, profile=profile)
            record_timing(profile, 'total', start)
            profiles.append(profile)
    finally:
        stop.set()
        worker.join()
        sys.setswitchinterval(previous)
    wall = [p['total_ms'] for p in profiles]
    return {
        'gil_switch_interval_ms': interval*1000,
        'mean_ms': float(np.mean(wall)),
        'p95_ms': float(np.percentile(wall, 95)),
        'p99_ms': float(np.percentile(wall, 99)),
        'worst': max(profiles, key=lambda p: p['total_ms']),
    }


if __name__ == '__main__':
    report = json.dumps([measure(interval) for interval in (.005, .001)], indent=2)
    if len(sys.argv) == 2:
        Path(sys.argv[1]).write_text(report + '\n')
    print(report)
