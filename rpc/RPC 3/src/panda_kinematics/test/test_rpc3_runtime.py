"""Controller timing and safety checks with named synthetic ROS feedback."""

import csv
import json
from threading import Event, Thread
from types import SimpleNamespace

import numpy as np
import pytest
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from panda_kinematics import rpc3_runtime
from panda_kinematics.joint_control import TorqueCommand
from panda_kinematics.panda_model import ARM_NAMES
from panda_kinematics.rpc3_runtime import TrackingRun, save_tracking
from panda_kinematics.trajectory_node import Q_END, Q_START, plan_trajectory


class Controller:
    mode = 'model_only'

    def __init__(self):
        self.holds = []
        self.command = np.zeros(7)

    def compute(self, q, dq, qd, dqd, ddqd, *, terminal_hold, profile=None):
        self.holds.append(terminal_hold)
        return TorqueCommand(self.command, self.command, self.command, self.command)


@pytest.fixture
def run(tmp_path, monkeypatch):
    run = TrackingRun(Controller(), plan_trajectory(Q_START, Q_END, 4, .01), .01,
                      str(tmp_path / 'tracking.csv'))
    run.test_now = 0.0
    monkeypatch.setattr(rpc3_runtime, 'time', SimpleNamespace(
        monotonic=lambda: run.test_now, perf_counter_ns=lambda: 0,
    ))
    return run


def feedback(run, now, status='waiting', q=Q_START, effort=None, source_delay=0.):
    run.test_now = now
    stamp = round((now - source_delay)*1e9)
    msg = JointState(name=list(reversed(ARM_NAMES)),
                     position=q[::-1].tolist(), velocity=[0.0]*7,
                     effort=[0.0]*7 if effort is None else list(effort))
    msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(stamp, 10**9)
    run.on_feedback(msg, now, round(now*1e9))
    run.on_status(String(data=status), now)


def test_named_feedback_and_stale_stop(run):
    commands = []
    assert not run.ready(1.0)
    feedback(run, 1.0, q=Q_END)
    assert not run.ready(1.0)
    feedback(run, 1.01)
    np.testing.assert_array_equal(run.feedback.q, Q_START)
    assert run.ready(1.01)
    run.step(1.01, commands.append)
    with pytest.raises(RuntimeError, match='stale'):
        run.step(1.07, commands.append)
    assert len(commands) == 1


def test_rejects_nonfinite_feedback_and_out_of_bounds_torque(run):
    commands = []
    feedback(run, 1.0, effort=[float('nan')]*7)
    assert not run.ready(1.0)
    feedback(run, 1.01)
    run.controller.command = np.full(7, 88.)
    with pytest.raises(ValueError, match='invalid torque'):
        run.step(1.01, commands.append)
    assert not commands


def test_elapsed_reference_endpoint_hold_deadline_and_csv(run, monkeypatch):
    clock = SimpleNamespace(monotonic=lambda: 0., perf_counter_ns=lambda: 0)
    monkeypatch.setattr(rpc3_runtime, 'time', clock)
    for now, status in [(1.0, 'waiting'), (4.99, 'active'), (5.02, 'active'),
                        (5.30, 'active'), (6.01, 'active')]:
        clock.monotonic = lambda: now + (0.012 if run.last_publish_at is not None else 0)
        feedback(run, now, status)
        if run.last_publish_at is not None:
            # Fast-forward the reference while modeling intervening valid publications.
            run.last_publish_at = now - .01
        run.step(now, lambda _: None)
    assert [row[2] for row in run.rows] == [
        'tracking', 'tracking', 'tracking', 'observation', 'terminal_hold',
    ]
    assert [row[1] for row in run.rows] == [0, 3.99, 4, 4, 4]
    assert run.controller.holds == [False, False, False, False, True]
    assert run.missed_slots == 497
    assert run.deadline_overruns == 4
    save_tracking(run)
    with open(run.output, newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert float(rows[2]['torque_compute_cpu_ms']) >= 0
    assert float(rows[2]['publication_ms']) >= 0
    assert float(rows[2]['trajectory_reference_ms']) >= 0
    assert float(rows[2]['error_4']) == pytest.approx(Q_END[3] - Q_START[3])
    with open(run.output.replace('.csv', '.json')) as stream:
        summary = json.load(stream)
    np.testing.assert_allclose(summary['tracking_final_rad_by_joint'], abs(Q_END - Q_START))


def test_bridge_watchdog_is_latched(run):
    feedback(run, 1.)
    run.step(1., lambda _: None)
    for now in (1.1, 1.11):
        run.on_status(String(data='timed_out'), now)
    assert run.watchdog_events == 1
    with pytest.raises(RuntimeError, match='watchdog'):
        run.step(1.11, lambda _: pytest.fail('must not publish after watchdog'))


def test_delayed_source_stamp_uses_ros_age_at_receipt(run):
    feedback(run, 1.0, source_delay=.04)
    assert run.ready(1.0)
    assert not run.ready(1.011)
    feedback(run, 1.08, source_delay=.06)
    assert not run.ready(1.08)
    assert run.last_receipt['source_age_ms'] == pytest.approx(60)
    assert run.rejected_feedback == 1


def test_feedback_can_refresh_during_compute_and_stale_result_is_discarded(run):
    feedback(run, 1.0)
    commands = []
    assert run.step(1.0, commands.append) == 'published'
    feedback(run, 1.04, status='active', source_delay=.04)
    compute = run.controller.compute
    entered, refreshed = Event(), Event()

    def receive_concurrently():
        if entered.wait(2):
            feedback(run, 1.06, status='active')
            refreshed.set()

    def slow_compute(*args, **kwargs):
        entered.set()
        assert refreshed.wait(2), 'Feedback must be accepted while compute is in progress'
        return compute(*args, **kwargs)

    worker = Thread(target=receive_concurrently)
    worker.start()
    run.controller.compute = slow_compute
    try:
        assert run.step(1.04, commands.append) == 'discarded'
    finally:
        worker.join(3)
    assert len(commands) == 1
    assert run.discarded_commands == 1
    assert run.last_diagnostics['used_feedback_age_after_ms'] == pytest.approx(60)
    assert run.last_diagnostics['latest_feedback_age_after_ms'] == pytest.approx(0)
    assert run.last_diagnostics['feedback_callbacks_during_compute'] == 1

    run.controller.compute = compute
    feedback(run, 1.07, status='active')
    assert run.step(1.07, commands.append) == 'published'
    assert len(commands) == 2


def test_stale_without_replacement_rejects_active_command(run):
    feedback(run, 1.)
    run.step(1., lambda _: None)
    feedback(run, 1.05, status='active', source_delay=.04)
    compute = run.controller.compute

    def slow_compute(*args, **kwargs):
        run.test_now = 1.07
        return compute(*args, **kwargs)

    run.controller.compute = slow_compute
    with pytest.raises(RuntimeError, match='stale after torque computation'):
        run.step(1.05, lambda _: pytest.fail('must not publish stale torque'))


def test_no_recovery_after_watchdog_interval_or_latch(run):
    feedback(run, 1.)
    run.step(1., lambda _: None)
    feedback(run, 1.101, status='active')
    with pytest.raises(RuntimeError, match='watchdog interval'):
        run.step(1.101, lambda _: pytest.fail('must not publish after command gap'))
    run.on_status(String(data='timed_out'), 1.102)
    feedback(run, 1.103, status='waiting')
    with pytest.raises(RuntimeError, match='latched timeout'):
        run.step(1.103, lambda _: pytest.fail('requires relaunch'))


def test_latch_arriving_during_compute_blocks_publish(run):
    feedback(run, 1.)
    compute = run.controller.compute

    def receive_fault(*args, **kwargs):
        run.on_status(String(data='timed_out'), 1.01)
        return compute(*args, **kwargs)

    run.controller.compute = receive_fault
    with pytest.raises(RuntimeError, match='latched timeout'):
        run.step(1., lambda _: pytest.fail('must not publish after observed latch'))


def test_jazzy_executor_services_feedback_group_during_control_work(monkeypatch):
    import rclpy
    from rclpy.executors import MultiThreadedExecutor

    from panda_kinematics import controller_node

    controller = Controller()
    monkeypatch.setattr(controller_node, 'configured_controller', lambda _: (controller, ''))
    create_publisher = controller_node.ControllerNode.create_publisher

    def publisher_after_warmup(node, *args, **kwargs):
        if args[1] == 'joint_commands':
            assert controller.holds == [False]*6 + [True]
            assert len(node.run.warmup) == 7
            assert node.run.started_at is None and node.run.command_count == 0
        return create_publisher(node, *args, **kwargs)

    monkeypatch.setattr(controller_node.ControllerNode, 'create_publisher', publisher_after_warmup)
    entered, serviced, completed = Event(), Event(), Event()
    rclpy.init(args=[])
    executor = MultiThreadedExecutor(num_threads=2)
    node = None
    try:
        node = controller_node.ControllerNode()
        node.start_timer.cancel()

        def control_work():
            entered.set()
            if serviced.wait(2):
                completed.set()

        control = node.create_guard_condition(control_work, callback_group=node.control_group)
        receive = node.create_guard_condition(serviced.set, callback_group=node.feedback_group)
        executor.add_node(node)
        control.trigger()
        executor.spin_once(timeout_sec=1)
        assert entered.wait(1)
        receive.trigger()
        executor.spin_once(timeout_sec=1)
        assert completed.wait(1)
    finally:
        executor.shutdown(wait_for_threads=True)
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


@pytest.mark.parametrize('stall_ms', [65, 120])
def test_first_control_stall_defers_without_publishing(run, stall_ms):
    feedback(run, 1., source_delay=.009)
    compute = run.controller.compute
    commands = []

    def stall(*args, **kwargs):
        run.test_now = 1 + stall_ms/1000
        return compute(*args, **kwargs)

    run.controller.compute = stall
    assert run.step(1., commands.append) == 'startup_deferred'
    assert not commands
    assert run.started_at is None
    assert run.first_attempts[0]['latest_feedback_age_after_ms'] == pytest.approx(stall_ms + 9)
    assert run.first_attempts[0]['command_gap_ms'] is None
    run.controller.compute = compute
    feedback(run, 1.15)
    assert run.step(1.15, commands.append) == 'published'
    assert len(commands) == 1
    assert run.rows[0][0:2] == (0., 0.)


def test_paused_startup_deferral_is_bounded_and_requires_home(run):
    feedback(run, 1., source_delay=.04)
    assert run.step(1.02, lambda _: pytest.fail('expired startup sample')) == 'startup_deferred'
    feedback(run, 1.2, q=Q_END)
    with pytest.raises(RuntimeError, match='home'):
        run.step(1.2, lambda _: pytest.fail('start must be at home'))

    feedback(run, 3.1)
    with pytest.raises(RuntimeError, match='retry window expired'):
        run.step(3.1, lambda _: pytest.fail('expired startup window'))


def test_monotonic_scheduler_skips_overrun_instead_of_bursting(monkeypatch):
    from panda_kinematics import controller_node

    clock = SimpleNamespace(now=0.)
    monkeypatch.setattr(controller_node, 'time', SimpleNamespace(monotonic=lambda: clock.now))
    deadlines = []

    def wait(delay):
        clock.now += delay
        return len(deadlines) >= 3

    def tick(deadline):
        deadlines.append(deadline)
        if len(deadlines) == 1:
            clock.now += .056

    controller_node.steady_control_loop(SimpleNamespace(wait=wait), tick, .01)
    assert deadlines == pytest.approx([.01, .07, .08])
