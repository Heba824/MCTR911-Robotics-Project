"""ROS configuration, tracking state, and telemetry for the Stage 3 node."""

import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
from threading import Lock, get_native_id
import time

import numpy as np
from rcl_interfaces.msg import ParameterDescriptor

from .cartesian_flc import CARTESIAN_FIELDS, CartesianController, CartesianVelocityFLC
from .control_timing import record_timing, timing_start
from .dynamics import PandaDynamics
from .fuzzy import JointFLC
from .joint_control import (
    JointController, LIMITS, POSITION_SCALE, TORQUE_SCALE, VELOCITY_SCALE,
)
from .panda_model import ARM_NAMES, DEFAULT_SCENE, PandaModel
from .trajectory_node import Q_MAX, Q_MIN


MAX_FEEDBACK_AGE = 0.05
BRIDGE_TIMEOUT = 0.1
PAUSED_STARTUP_RETRY_SECONDS = 2.0
MAX_TRACKING_ERROR = 0.5
HOME_TOLERANCE = 0.01
OBSERVATION_SECONDS = 1.0
RECORDED_HOLD_SECONDS = 0.5
PROFILE_PHASES = (
    'snapshot_lock_wait', 'snapshot_lock_work', 'trajectory_reference',
    'dynamics', 'reference_and_model_numpy', 'correction', 'torque_limits_numpy',
    'torque_compute', 'command_validation', 'publish_lock_wait', 'publish_lock_work',
    'publication', 'cartesian_mapping', 'fuzzy_memberships', 'fuzzy_rules', 'fuzzy_aggregation', 'fuzzy_centroid',
)
PROFILE_FIELDS = tuple(f'{name}_{unit}' for name in PROFILE_PHASES for unit in ('ms', 'cpu_ms')) + (
    'correction_voluntary_switches', 'correction_involuntary_switches',
    'correction_minor_faults', 'correction_major_faults',
)


def configured_controller(node):
    """Load per-run configuration outside the timed control loop."""
    def setting(name, default):
        return node.declare_parameter(
            name, default, ParameterDescriptor(read_only=True),
        ).value

    mode = setting('control_mode', 'joint_flc')
    scene = setting('scene', str(DEFAULT_SCENE))
    output = setting('results_csv', '')
    dynamics = PandaDynamics(PandaModel.from_scene(scene))
    if mode == 'cartesian_flc':
        fuzzy = CartesianVelocityFLC(
            setting('cartesian_position_scale', [0.035]*3),
            setting('cartesian_velocity_scale', [0.24]*3),
            setting('cartesian_correction_scale', [0.14]*3),
        )
        return CartesianController(dynamics, fuzzy), output
    fuzzy = JointFLC(
        setting('position_scale', POSITION_SCALE),
        setting('velocity_scale', VELOCITY_SCALE),
        setting('torque_scale', [float(v) for v in TORQUE_SCALE]),
    )
    return JointController(dynamics, mode, fuzzy), output


@dataclass(frozen=True)
class Feedback:
    q: np.ndarray
    dq: np.ndarray
    effort: np.ndarray
    stamp_ns: int
    received_at: float
    source_age: float


class TrackingRun:
    """Generate commands only while fresh feedback and timing are valid."""

    def __init__(self, controller, trajectory, period, output=''):
        self.controller = controller
        self.times, self.q, self.dq, self.ddq = trajectory
        self.period = period
        self.output = output
        self._lock = Lock()
        self.feedback = None
        self.last_receipt = {}
        self.receipt_count = 0
        self.feedback_first_receipt = None
        self.rejected_feedback = 0
        self._last_stamp = 0
        self.last_diagnostics = {}
        self.last_callback_at = None
        self.last_scheduler_deadline = None
        self.skipped_scheduler_ticks = 0
        self.publish_deadline_misses = 0
        self.last_publish_at = None
        self.discarded_commands = 0
        self.status = None
        self.status_at = None
        self.fault = None
        self.fault_diagnostics = None
        self.started_at = None
        self.startup_retry_since = None
        self.startup_deferrals = 0
        self.callback_attempts = 0
        self.first_attempts = []
        self.last_slot = None
        self.endpoint_sent = False
        self.rows = []
        self.warmup = []
        self.shutdown_ros_ns = None
        self.deadline_overruns = 0
        self.compute_overruns = 0
        self.missed_slots = 0
        self.command_count = 0
        self.terminal_hold_commands = 0
        self.saturation_count = 0
        self.watchdog_events = 0

    def _latch_fault(self, reason):
        if self.fault is None:
            self.fault = reason
            self.fault_diagnostics = {
                'timing': self.last_diagnostics.copy(),
                'last_joint_state': self.last_receipt.copy(),
            }

    def latch_fault(self, reason):
        with self._lock:
            self._latch_fault(reason)

    def on_status(self, msg, now=None):
        now = time.monotonic() if now is None else now
        with self._lock:
            if msg.data == 'timed_out' and self.status != 'timed_out':
                self.watchdog_events += 1
                self._latch_fault('Bridge watchdog reported a latched timeout')
            self.status, self.status_at = msg.data, now

    def on_feedback(self, msg, received_at, ros_now_ns):
        stamp = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
        age = (ros_now_ns - stamp) / 1e9
        error = None
        try:
            names = tuple(msg.name)
            if any(names.count(name) != 1 for name in ARM_NAMES):
                raise ValueError('Missing or duplicate arm joint names')
            indices = [names.index(name) for name in ARM_NAMES]
            q, dq, effort = (np.asarray(field, dtype=float)[indices]
                             for field in (msg.position, msg.velocity, msg.effort))
            if stamp <= 0 or age < -0.01 or age > MAX_FEEDBACK_AGE:
                raise ValueError('Joint-state source stamp outside clock/age limits')
            if (not np.all(np.isfinite([q, dq, effort]))
                    or np.any(q < Q_MIN) or np.any(q > Q_MAX)):
                raise ValueError('Nonfinite or out-of-bounds joint state')
        except (IndexError, ValueError) as exc:
            error = str(exc)
        with self._lock:
            previous_receipt = self.last_receipt.get('receipt_monotonic_s')
            previous_stamp = self._last_stamp
            if previous_stamp and stamp <= previous_stamp:
                error = error or 'Duplicate or out-of-order joint-state stamp'
            self.receipt_count += 1
            if self.feedback_first_receipt is None:
                self.feedback_first_receipt = received_at
            self.last_receipt = {
                'receipt_monotonic_s': received_at, 'receipt_ros_ns': ros_now_ns,
                'source_stamp_ns': stamp, 'source_age_ms': age*1000,
                'receipt_gap_ms': None if previous_receipt is None
                else (received_at - previous_receipt)*1000,
                'source_gap_ms': None if not previous_stamp else (stamp - previous_stamp)/1e6,
                'error': error,
            }
            if error:
                self.rejected_feedback += 1
                if stamp > previous_stamp:
                    self.feedback = None
                return
            self._last_stamp = stamp
            self.feedback = Feedback(q, dq, effort, stamp, received_at, max(0.0, age))

    @staticmethod
    def _age(feedback, now):
        return None if feedback is None else feedback.source_age + now - feedback.received_at

    def _fresh(self, now):
        age = self._age(self.feedback, now)
        return (age is not None and age <= MAX_FEEDBACK_AGE
                and self.status_at is not None and now - self.status_at <= MAX_FEEDBACK_AGE)

    def _ready(self, now):
        return (not self.fault and self.status == 'waiting' and self._fresh(now)
                and np.max(abs(self.feedback.q - self.q[0])) <= HOME_TOLERANCE
                and np.max(abs(self.feedback.dq)) <= 0.01)

    def ready(self, now):
        with self._lock:
            return self._ready(now)

    def diagnostic_message(self):
        with self._lock:
            return f'timing={self.last_diagnostics!r}, last_joint_state={self.last_receipt!r}'

    def _defer_paused_start(self, now):
        # A paused bridge has received no torque from this controller. Retry from
        # home only after new feedback and status satisfy the normal gates.
        if (self.started_at is not None or self.status != 'waiting'
                or now - self.startup_retry_since >= PAUSED_STARTUP_RETRY_SECONDS):
            return False
        self.startup_deferrals += 1
        self.last_slot = None
        return True

    def step(self, now, publish, *, deadline=None):
        self.callback_attempts += 1
        try:
            return self._step(now, publish, deadline)
        finally:
            if self.callback_attempts <= 8:
                self.first_attempts.append(self.last_diagnostics.copy())

    def _step(self, now, publish, deadline):
        interval = None if self.last_callback_at is None else (now - self.last_callback_at)*1000
        self.last_callback_at = now
        if deadline is not None:
            if self.last_scheduler_deadline is not None:
                elapsed_ticks = round((deadline - self.last_scheduler_deadline)/self.period)
                self.skipped_scheduler_ticks += max(0, elapsed_ticks - 1)
            self.last_scheduler_deadline = deadline
        profile = {}
        start = timing_start()
        with self._lock:
            start = record_timing(profile, 'snapshot_lock_wait', start)
            feedback = self.feedback
            receipt_count = self.receipt_count
            status = self.status
            age_before = self._age(feedback, now)
            self.last_diagnostics = {
                'callback_monotonic_s': now, 'callback_interval_ms': interval,
                'attempt': self.callback_attempts, 'worker_native_id': get_native_id(),
                'scheduled_deadline_s': deadline,
                'dispatch_lateness_ms': None if deadline is None else (now - deadline)*1000,
                'used_feedback_age_before_ms': None if age_before is None else age_before*1000,
                'used_feedback_stamp_ns': None if feedback is None else feedback.stamp_ns,
                'used_feedback_receipt_monotonic_s': None if feedback is None else feedback.received_at,
                'bridge_status_age_ms': None if self.status_at is None else (now - self.status_at)*1000,
                'missed_reference_slots': self.missed_slots, 'bridge_status': status,
            }
            if self.fault:
                raise RuntimeError(self.fault)
            if self.started_at is None:
                if self.startup_retry_since is None:
                    self.startup_retry_since = now
                elif now - self.startup_retry_since >= PAUSED_STARTUP_RETRY_SECONDS:
                    raise RuntimeError('Paused startup retry window expired without a command')
            if not self._fresh(now) or status not in ('waiting', 'active'):
                if self._defer_paused_start(now):
                    return 'startup_deferred'
                raise RuntimeError('Bridge feedback stale, invalid, or unavailable')
            if self.started_at is None and not self._ready(now):
                raise RuntimeError('First torque command requires paused bridge at home')
        start = record_timing(profile, 'snapshot_lock_work', start)
        elapsed = 0.0 if self.started_at is None else now - self.started_at
        if elapsed > 0.08 and status != 'active':
            raise RuntimeError('Bridge did not acknowledge the first torque commands')

        slot = math.floor(elapsed / self.period + 1e-8)
        if self.last_slot is not None:
            self.missed_slots += max(0, slot - self.last_slot - 1)
        self.last_slot = slot
        end = self.times[-1]
        final = len(self.times) - 1
        index = min(slot, final)
        phase = 'tracking'
        if index == final:
            if self.endpoint_sent:
                phase = 'observation' if elapsed < end + OBSERVATION_SECONDS else 'terminal_hold'

        q_des, dq_des, ddq_des = self.q[index], self.dq[index], self.ddq[index]
        error = q_des - feedback.q
        if np.max(abs(error)) > MAX_TRACKING_ERROR:
            raise RuntimeError('Tracking error exceeded 0.5 rad')
        start = record_timing(profile, 'trajectory_reference', start)
        torque = self.controller.compute(feedback.q, feedback.dq, q_des, dq_des, ddq_des,
                                         terminal_hold=phase == 'terminal_hold', profile=profile)
        start = record_timing(profile, 'torque_compute', start)
        compute_ms = profile['torque_compute_ms']
        if (np.shape(torque.command) != (7,) or not np.all(np.isfinite(torque.command))
                or np.any(abs(torque.command) > LIMITS)):
            raise ValueError('Controller produced an invalid torque command')
        start = record_timing(profile, 'command_validation', start)
        with self._lock:
            start = record_timing(profile, 'publish_lock_wait', start)
            checked_at = time.monotonic()
            used_age = self._age(feedback, checked_at)
            latest_age = self._age(self.feedback, checked_at)
            self.last_diagnostics.update(
                used_feedback_age_after_ms=used_age*1000,
                latest_feedback_age_after_ms=None if latest_age is None else latest_age*1000,
                feedback_callbacks_during_compute=self.receipt_count - receipt_count,
                latest_feedback_stamp_ns=None if self.feedback is None else self.feedback.stamp_ns,
                compute_ms=compute_ms, callback_to_check_ms=(checked_at - now)*1000,
                missed_reference_slots=self.missed_slots, bridge_status=self.status,
                bridge_status_age_ms=(checked_at - self.status_at)*1000,
                command_gap_ms=None if self.last_publish_at is None
                else (checked_at - self.last_publish_at)*1000,
                profile=profile,
            )
            self.compute_overruns += int(compute_ms > self.period * 1000)
            if self.fault or self.status not in ('waiting', 'active'):
                raise RuntimeError(self.fault or 'Bridge left the permitted command states')
            if not self._fresh(checked_at):
                if self._defer_paused_start(checked_at):
                    self.discarded_commands += 1
                    self.deadline_overruns += int(checked_at - now > self.period)
                    return 'startup_deferred'
                raise RuntimeError('Latest bridge feedback or status stale after torque computation')
            if self.last_publish_at is not None and checked_at - self.last_publish_at >= BRIDGE_TIMEOUT:
                raise RuntimeError('Command gap reached the bridge watchdog interval')
            if used_age > MAX_FEEDBACK_AGE:
                # The new sample is fresh, but this torque was calculated from the old one.
                self.discarded_commands += 1
                if self.started_at is None:
                    if self._defer_paused_start(checked_at):
                        self.deadline_overruns += int(checked_at - now > self.period)
                        return 'startup_deferred'
                    raise RuntimeError('First torque sample expired before publication')
                self.deadline_overruns += int(checked_at - now > self.period)
                return 'discarded'
            start = record_timing(profile, 'publish_lock_work', start)
            publish_ros_ns = publish(torque.command)
            record_timing(profile, 'publication', start)
            self.last_publish_at = time.monotonic()
            if self.started_at is None:
                self.started_at = self.last_publish_at
        self.endpoint_sent |= index == final
        saturated = int(np.count_nonzero(torque.command != torque.requested))
        self.saturation_count += saturated
        self.command_count += 1
        self.terminal_hold_commands += int(phase == 'terminal_hold')
        callback_ms = (self.last_publish_at - now) * 1000
        self.deadline_overruns += int(callback_ms > self.period * 1000)
        if deadline is not None:
            self.publish_deadline_misses += int(self.last_publish_at > deadline + self.period)
        if self.output and (elapsed <= end + OBSERVATION_SECONDS + RECORDED_HOLD_SECONDS
                            or phase == 'tracking'):
            self.rows.append((elapsed, self.times[index], phase, feedback.stamp_ns, self.status,
                              compute_ms, callback_ms, self.missed_slots, self.deadline_overruns,
                              saturated, *feedback.q, *feedback.dq, *q_des, *dq_des, *error,
                              *torque.requested, *torque.command, *feedback.effort,
                              now, interval, feedback.received_at, feedback.source_age*1000,
                              age_before*1000, used_age*1000,
                              self.last_diagnostics['latest_feedback_age_after_ms'],
                              self.last_diagnostics['feedback_callbacks_during_compute'],
                              self.command_count, publish_ros_ns, self.last_publish_at,
                              self.last_diagnostics['dispatch_lateness_ms'],
                              *(profile.get(field) for field in (*PROFILE_FIELDS, *CARTESIAN_FIELDS))))
        return 'published'


def save_tracking(run):
    """Write buffered tracking and diagnostics after stopping command timers."""
    if not run.output:
        return
    path = Path(run.output).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ('q', 'dq', 'q_des', 'dq_des', 'error', 'tau_requested', 'tau_command',
              'tau_feedback_before_command')
    header = ['elapsed_s', 'reference_s', 'phase', 'feedback_stamp_ns', 'bridge_status',
              'compute_ms', 'callback_to_publish_ms', 'missed_slots', 'deadline_overruns',
              'saturated_joints']
    header.extend(f'{field}_{i}' for field in fields for i in range(1, 8))
    header.extend(('callback_monotonic_s', 'callback_interval_ms', 'feedback_receipt_monotonic_s',
                   'feedback_source_age_at_receipt_ms', 'used_feedback_age_before_ms',
                   'used_feedback_age_after_ms', 'latest_feedback_age_after_ms',
                   'feedback_callbacks_during_compute'))
    header.extend(('command_sequence', 'publish_ros_ns', 'publish_monotonic_s',
                   'dispatch_lateness_ms'))
    header.extend((*PROFILE_FIELDS, *CARTESIAN_FIELDS))
    with path.open('w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(run.rows)
    tracked = [row for row in run.rows if row[2] == 'tracking']
    summary = {
        'mode': run.controller.mode, 'commands': run.command_count, 'saved_rows': len(run.rows),
        'tracking_endpoint_sent': run.endpoint_sent, 'terminal_hold_commands': run.terminal_hold_commands, 'missed_trajectory_slots': run.missed_slots,
        'callback_deadline_overruns': run.deadline_overruns,
        'compute_deadline_overruns': run.compute_overruns,
        'skipped_scheduler_ticks': run.skipped_scheduler_ticks,
        'publish_deadline_misses': run.publish_deadline_misses,
        'saturated_joint_samples': run.saturation_count, 'bridge_status': run.status,
        'watchdog_events_observed': run.watchdog_events, 'fault': run.fault,
        'feedback_callbacks': run.receipt_count, 'rejected_feedback': run.rejected_feedback,
        'discarded_commands': run.discarded_commands,
        'last_joint_state_receipt': run.last_receipt, 'last_callback_timing': run.last_diagnostics,
        'shutdown_ros_ns': run.shutdown_ros_ns,
        'first_fault_diagnostics': run.fault_diagnostics, 'warmup': run.warmup,
        'first_control_attempts': run.first_attempts, 'startup_deferrals': run.startup_deferrals,
    }
    if run.receipt_count > 1:
        duration = run.last_receipt['receipt_monotonic_s'] - run.feedback_first_receipt
        if duration > 0:
            summary['feedback_receipt_hz'] = (run.receipt_count - 1) / duration
    if run.rows:
        for name, column in (('compute_ms', 5), ('callback_to_publish_ms', 6)):
            latencies = np.asarray([row[column] for row in run.rows])
            summary[name] = dict(zip(
                ('mean', 'p95', 'p99', 'maximum'),
                (float(latencies.mean()), *np.percentile(latencies, [95, 99]), float(latencies.max())),
            ))
    if len(run.rows) > 1:
        # Publication times derived from the monotonic callback start plus its duration.
        publishes = np.asarray([row[0] + row[6]/1000 for row in run.rows])
        summary['recorded_mean_publish_hz'] = float((len(publishes) - 1)
                                                    / (publishes[-1] - publishes[0]))
        summary['recorded_max_publish_gap_ms'] = float(np.diff(publishes).max()*1000)
    if tracked:
        errors = np.asarray([row[38:45] for row in tracked])
        summary['tracking_rms_rad_by_joint'] = np.sqrt(np.mean(errors**2, axis=0)).tolist()
        summary['tracking_peak_rad_by_joint'] = np.max(abs(errors), axis=0).tolist()
        if tracked[-1][1] == run.times[-1]:
            summary['tracking_final_rad_by_joint'] = abs(errors[-1]).tolist()
    path.with_suffix('.json').write_text(json.dumps(summary, indent=2) + '\n')
