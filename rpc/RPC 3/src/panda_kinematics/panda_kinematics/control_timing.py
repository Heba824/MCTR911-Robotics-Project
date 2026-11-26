"""Timing helpers shared by the Stage 3 control law and fuzzy inference."""

from time import perf_counter_ns, thread_time_ns


def timing_start():
    """Read wall and current-thread CPU clocks."""
    return perf_counter_ns(), thread_time_ns()


def record_timing(profile, name, start):
    end = timing_start()
    profile[f'{name}_ms'] = (end[0] - start[0])/1e6
    profile[f'{name}_cpu_ms'] = (end[1] - start[1])/1e6
    return end
