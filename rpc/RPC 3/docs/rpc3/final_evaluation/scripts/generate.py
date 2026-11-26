"""Rebuild evaluation tables and PNGs from bundled telemetry and the verified FK."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from panda_kinematics.cartesian_flc import (
    JOINT_ACCEL, JOINT_SPEED, MAX_DAMPING, SINGULAR_THRESHOLD, TASK_SPEED, CartesianVelocityFLC,
)
from panda_kinematics.joint_control import LIMITS
from panda_kinematics.trajectory_node import forward_kinematics

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'evidence'
SOURCES = json.loads((DATA / 'sources.json').read_text())
MODES = ('model_only', 'joint_flc', 'cartesian_flc')
LABELS = ('Model only', 'Joint FLC', 'Cartesian FLC')
COLORS = ('#66758b', '#126b8f', '#bb4f18')
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})


def read_json(relative):
    return json.loads((DATA / relative).read_text())


def read_rows(relative):
    with (DATA / relative).open(newline='') as stream:
        return list(csv.DictReader(stream))


def col(rows, key):
    return np.array([float(row[key]) for row in rows])


def mat(rows, key, suffix=range(1, 8)):
    return np.column_stack([col(rows, f'{key}_{i}') for i in suffix])


def table(name, rows):
    with (ROOT / 'tables' / (name + '.csv')).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    def cell(value):
        if value is None or value == '':
            return '—'
        if isinstance(value, float):
            return f'{value:.6g}'
        return str(value).replace('|', r'\|')
    with (ROOT / 'tables' / (name + '.md')).open('w') as stream:
        stream.write(f'## {name.replace("_", " ").title()}\n\n')
        stream.write('| ' + ' | '.join(rows[0]) + ' |\n')
        stream.write('| ' + ' | '.join('---' for _ in rows[0]) + ' |\n')
        for row in rows:
            stream.write('| ' + ' | '.join(cell(row[k]) for k in rows[0]) + ' |\n')


def rms(v):
    return float(np.sqrt(np.mean(v*v)))


def tracking(relative, condition, mode, trial, summary, acceptance=None):
    assert summary['mode'] == mode, relative
    rows = read_rows(relative)
    track = [r for r in rows if r['phase'] == 'tracking']
    reference_key = 'sim_time' if condition == 'test pipe' else 'reference_s'
    assert len(track) and abs(float(track[-1][reference_key]) - 4) < 1e-8, relative
    q, qd = (mat(track, key) for key in ('q', 'q_des'))
    xyz, xyzd = (np.array([forward_kinematics(v)[:3, 3] for v in poses]) for poses in (q, qd))
    qe, xe = qd-q, (xyzd-xyz)*1000
    np.testing.assert_allclose(qe, mat(track, 'error'), atol=1e-12)
    if 'hand_error_x' in track[0]:
        np.testing.assert_allclose(xe/1000, mat(track, 'hand_error', 'xyz'), atol=1e-12)
    norm = np.linalg.norm(xe, axis=1)
    if condition == 'test pipe':
        send = col(rows, 'sim_time')*1000 + col(rows, 'send_offset_from_tick_ms')
    else:
        send = col(rows, 'publish_monotonic_s')*1000
    gap = np.diff(send)
    latency = col(rows, 'compute_ms')
    torque = mat(rows, 'tau_command')
    assert np.all(abs(torque) <= LIMITS+1e-12)
    source = SOURCES[relative]
    identity = {'condition': condition, 'mode': mode, 'trial': trial}
    track_stats = {
        **identity, 'samples': len(track), 'joint_rmse_rad': rms(qe),
        'joint_peak_rad': float(abs(qe).max()), 'joint_final_max_rad': float(abs(qe[-1]).max()),
        'joint_final_rms_rad': rms(qe[-1]),
        'xyz_rmse_mm': rms(norm), 'xyz_peak_mm': float(norm.max()), 'xyz_final_mm': float(norm[-1]),
        'peak_command_Nm': float(abs(torque).max()), 'source_csv': source,
    }
    times = {
        **identity, 'timed_rows': len(rows), 'compute_mean_ms': float(latency.mean()),
        'compute_p95_ms': float(np.percentile(latency, 95)),
        'compute_p99_ms': float(np.percentile(latency, 99)), 'compute_max_ms': float(latency.max()),
        'command_gap_mean_ms': float(gap.mean()), 'command_gap_max_ms': float(gap.max()),
        'bridge_receipt_gap_max_ms': None if acceptance is None else acceptance.get('max_valid_receipt_gap_ms'),
        'feedback_receive_hz': summary.get('feedback_receipt_hz'), 'source_csv': source,
    }

    def count(field):
        values = [int(float(row[field])) for row in rows if row.get(field)]
        return sum(values) if values else None

    safety = {
        **identity, 'total_commands': summary.get('commands', summary.get('cycles')),
        'missed_reference_slots': summary.get('missed_trajectory_slots'),
        'skipped_scheduler_ticks': summary.get('skipped_scheduler_ticks'),
        'publish_deadline_misses': summary.get('publish_deadline_misses',
                                               summary.get('send_deadline_misses')),
        'callback_overruns': summary.get('callback_deadline_overruns'),
        'compute_overruns': summary.get('compute_deadline_overruns', summary.get('compute_over_10ms')),
        'torque_saturated_joint_samples': summary.get('saturated_joint_samples'),
        'task_speed_limited_ticks': count('cartesian_speed_limited'),
        'joint_speed_limited_components': count('cartesian_joint_speed_limited'),
        'accel_limited_components': count('cartesian_accel_limited'),
        'controller_watchdogs': summary.get('watchdog_events_observed', summary.get('watchdog_events')),
        'bridge_watchdogs_before_shutdown': None if acceptance is None else
        acceptance.get('bridge_watchdogs_before_shutdown'),
        'bridge_watchdogs_total': None if acceptance is None else acceptance.get('bridge_watchdogs_total'),
        'bridge_stale_torques': None if acceptance is None else acceptance.get('bridge_rejected_stale'),
        'rejected_feedback_samples': summary.get('rejected_feedback'),
        'discarded_commands': summary.get('discarded_commands'), 'fault': summary.get('fault'),
        'source_csv': source,
    }
    t = col(track, reference_key if condition == 'test pipe' else 'elapsed_s')
    return dict(rows=rows, track=track, t=t, q=q, qd=qd, qe=qe, xyz=xyz, xyzd=xyzd,
                xe=xe, norm=norm, latency=latency, gap=gap, stats=track_stats, timing=times,
                safety=safety, per_joint={**identity, **{f'joint_{i+1}_rmse_rad': rms(qe[:, i])
                                                        for i in range(7)}, 'source_csv': source})


def figure(title, ylabel='', xlabel='Simulation time (s)', n=1):
    fig, axes = plt.subplots(n, 1, figsize=(8.5, 3.1 if n == 1 else 2*n+1), sharex=n>1,
                             squeeze=False)
    fig.suptitle(title, weight='bold')
    axs = axes[:, 0]
    axs[-1].set_xlabel(xlabel)
    if ylabel:
        axs[0].set_ylabel(ylabel)
    return fig, axs


def save(fig, name):
    for ax in fig.axes:
        if ax.axison:
            ax.grid(alpha=.21)
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=8, loc='best')
    fig.tight_layout()
    fig.savefig(ROOT / 'figures' / (name+'.png'), dpi=240, facecolor='white')
    plt.close(fig)


def joints(data, kind, title, filename):
    fig, axes = plt.subplots(4, 2, sharex=True, figsize=(10, 8))
    fig.suptitle(title, weight='bold')
    axes.flat[7].axis('off')
    for i, ax in enumerate(list(axes.flat)[:7]):
        for d, label, color in data:
            if kind == 'position':
                if d is data[0][0]:
                    ax.plot(d['t'], d['qd'][:, i], '--', color='#252836', label='Desired')
                y = d['q'][:, i]
            elif kind == 'error':
                y = d['qe'][:, i]
            else:
                y = mat(d['track'], 'tau_command')[:, i]
                if d is data[0][0]:
                    ax.axhline(LIMITS[i], color='k', ls=':', label='Actuator bounds')
                    ax.axhline(-LIMITS[i], color='k', ls=':')
            ax.plot(d['t'], y, color=color, label=label)
        ax.set_title(f'Joint {i+1}')
        ax.set_ylabel('Torque (N·m)' if kind == 'torque' else 'Position (rad)' if kind == 'position'
                      else 'Error (rad)')
        if i >= 5:
            ax.set_xlabel('Simulation time (s)')
    save(fig, filename)


def bars(values, labels, title, ylabel, name):
    fig, (ax,) = figure(title, ylabel, '')
    rectangles = ax.bar(labels, values, color=COLORS[:len(values)])
    ax.bar_label(rectangles, fmt='%.3g', padding=4)
    ax.margins(y=.16)
    save(fig, name)


def plots(pipe, gui, singular):
    m, j, c = pipe
    pair = [(m, LABELS[0], COLORS[0]), (j, LABELS[1], COLORS[1])]
    joints(pair, 'position', '01 · Joint position tracking · headless test pipe', '01_joint_positions')
    joints(pair, 'error', '02 · Joint tracking errors · headless test pipe', '02_joint_errors')
    for no, kind in [('03', 'RMS'), ('04', 'peak absolute')]:
        fig, (ax,) = figure(f'{no} · Per-joint {kind} error · headless test pipe', 'Error (rad)', 'Joint')
        for i, (d, label, color) in enumerate(pair):
            val = np.sqrt(np.mean(d['qe']**2, axis=0)) if no == '03' else abs(d['qe']).max(axis=0)
            ax.bar(np.arange(1, 8)+(.34*i-.17), val, width=.34, label=label, color=color)
        save(fig, f'{no}_joint_{"rms" if no == "03" else "maximum"}')
    joints(pair, 'torque', '05 · Arm torque commands and actuator bounds · headless test pipe', '05_joint_torques')
    for no, which in [('06', 'position'), ('07', 'error')]:
        fig, axes = figure(f'{no} · Cartesian FLC XYZ {which} · headless test pipe', n=3)
        for k, ax in enumerate(axes):
            if which == 'position':
                ax.plot(c['t'], c['xyzd'][:, k], 'k--', label='Desired')
                ax.plot(c['t'], c['xyz'][:, k], color=COLORS[2], label='Actual')
            else:
                ax.plot(c['t'], c['xe'][:, k], color=COLORS[2])
            ax.set_ylabel(f'{"XYZ"[k]} ({"m" if which == "position" else "mm"})')
        save(fig, f'{no}_cartesian_{which}')
    fig, (ax,) = figure('08 · Cartesian FLC XYZ error norm · headless test pipe', 'Error norm (mm)')
    ax.plot(c['t'], c['norm'], color=COLORS[2])
    save(fig, '08_xyz_error_norm')
    joints([(c, LABELS[2], COLORS[2])], 'position', '09 · Joint motion in Cartesian FLC · headless test pipe',
           '09_cartesian_joint_motion')
    flc = CartesianVelocityFLC([.035]*3, [.24]*3, [.14]*3)
    logged_error = mat(c['track'], 'cartesian_error', 'xyz')
    np.testing.assert_allclose(logged_error, c['xe']/1000, atol=1e-12)
    correction = np.array([flc(e, v) for e, v in zip(
        logged_error, mat(c['track'], 'cartesian_velocity_error', 'xyz'))])
    command = mat(c['track'], 'cartesian_command', 'xyz')
    fig, axes = figure('10 · Recomputed fuzzy correction and logged XYZ command · test pipe', n=3)
    for i, ax in enumerate(axes):
        ax.plot(c['t'], correction[:, i], label='Fuzzy correction (recomputed)')
        ax.plot(c['t'], command[:, i], label='Velocity command (recorded)')
        ax.set_ylabel(f'{"XYZ"[i]} (m/s)')
    save(fig, '10_fuzzy_velocity')
    for no, key, text in [('11', 'jacobian_sigma_3', 'Minimum Jacobian singular value'),
                          ('12', 'jacobian_damping', 'Adaptive damping coefficient')]:
        fig, (ax,) = figure(f'{no} · {text} · Cartesian FLC test pipe', 'm/rad')
        ax.plot(c['t'], col(c['track'], key), color=COLORS[2], label=text)
        if no == '11':
            ax.axhline(SINGULAR_THRESHOLD, color='k', ls=':', label='Damping onset')
        else:
            ax.set_ylim(-.005, MAX_DAMPING*1.05)
        save(fig, f'{no}_{"sigma_min" if no == "11" else "damping"}')
    fig, axes = figure('13 · Cartesian FLC command limits · headless test pipe', n=3)
    series = [(np.linalg.norm(command, axis=1), TASK_SPEED, 'Task speed (m/s)'),
              (abs(mat(c['track'], 'cartesian_joint_speed')).max(axis=1), JOINT_SPEED, 'Joint speed (rad/s)'),
              ((abs(mat(c['track'], 'tau_command'))/LIMITS).max(axis=1), 1., 'Torque / joint limit')]
    for ax, (y, limit, label) in zip(axes, series):
        ax.plot(c['t'], y, color=COLORS[2], label='Largest command')
        ax.axhline(limit, ls=':', color='k', label='Limit')
        ax.set_ylabel(label)
    counts = c['safety']
    axes[2].set_title(f'Activations · task / joint speed / accel. ({JOINT_ACCEL:g} rad/s²) / torque = ' +
                      ' / '.join(str(counts[k]) for k in ('task_speed_limited_ticks',
                        'joint_speed_limited_components', 'accel_limited_components',
                        'torque_saturated_joint_samples')), fontsize=9)
    save(fig, '13_limit_behavior')
    bars([d['stats']['xyz_rmse_mm'] for d in pipe], LABELS, '14 · XYZ RMSE · headless test pipe',
         'XYZ norm RMSE (mm)', '14_xyz_rmse')
    bars([d['stats']['joint_rmse_rad'] for d in pipe], LABELS, '15 · Joint RMSE · headless test pipe',
         'Joint RMSE (rad)', '15_joint_rmse')
    fig, (ax,) = figure('16 · Torque computation latency · headless test pipe', 'Wall time (ms)', '')
    for i, d in enumerate(pipe):
        values = [d['timing'][f'compute_{k}_ms'] for k in ('mean', 'p95', 'p99', 'max')]
        ax.bar(np.arange(4)+.23*(i-1), values, width=.22, color=COLORS[i], label=LABELS[i])
    ax.set_xticks(range(4), ('Mean', 'p95', 'p99', 'Max'))
    ax.axhline(10, ls=':', color='k', label='10 ms budget')
    save(fig, '16_latency')
    fig, axes = figure('17 · Test-pipe send gaps and deadline misses', xlabel='Send index', n=2)
    for d, label, color in zip(pipe, LABELS, COLORS):
        axes[0].plot(np.arange(len(d['gap'])), d['gap'], color=color, alpha=.75, label=label)
        offset = col(d['rows'], 'send_offset_from_tick_ms')
        axes[1].step(np.arange(len(offset)), np.cumsum(offset > 10), color=color, label=label)
    axes[0].axhline(10, ls=':', color='k', label='Nominal period')
    axes[0].set_ylabel('Send gap (ms)')
    axes[1].set_ylabel('Cumulative deadline misses')
    if all(np.all(col(d['rows'], 'send_offset_from_tick_ms') <= 10) for d in pipe):
        axes[1].set_ylim(-.05, 1.05)
        axes[1].set_yticks([0, 1])
    save(fig, '17_command_timing')
    for no, suffix, ylabel in [('18', 'joint_speed_norm_rad_s', 'Joint-speed norm (rad/s)'),
                                ('19', 'task_residual_m_s', 'Task residual (m/s)')]:
        fig, (ax,) = figure(f'{no} · Near-singular pose · offline request sweep', ylabel,
                            'Weak-direction request (m/s)')
        for tag, name in [('undamped', 'Undamped'), ('damped', 'Adaptively damped')]:
            ax.plot(col(singular, 'desired_speed_m_s'), col(singular, f'{tag}_{suffix}'),
                    'o-', label=name)
        if no == '18':
            ax.set_yscale('log')
        save(fig, f'{no}_singular_{"speed" if no == "18" else "residual"}')
    fig, (ax,) = figure('20 · Adaptive damping law · offline parameter evaluation',
                        'Damping λ (m/rad)', 'Minimum singular value (m/rad)')
    sigma = np.linspace(0, .13, 131)
    values = MAX_DAMPING*np.clip(1-sigma/SINGULAR_THRESHOLD, 0, 1)**2
    ax.plot(sigma, values, label='Implemented damping law')
    pose = read_json('offline/evaluation.json')['singularity']
    ax.scatter(pose['sigma'][-1], pose['damping'], color=COLORS[2], zorder=3, label='Test pose')
    table('damping_law_offline', [dict(sigma_min_m_per_rad=float(s), damping_m_per_rad=float(v))
                                 for s, v in zip(sigma, values)])
    save(fig, '20_damping_law')
    fig, axes = figure('21 · Cartesian FLC timing and terminal hold · graphical ROS run',
                       xlabel='Elapsed time (s)', n=3)
    gt = col(gui['rows'], 'elapsed_s')
    error = mat(gui['rows'], 'error')
    axes[0].plot(gt, np.max(abs(error), axis=1), color=COLORS[2], label='Maximum joint error')
    axes[1].plot(gt[1:], gui['gap'], label='Publication gap')
    axes[1].plot(gt, gui['latency'], label='Torque computation', alpha=.75)
    axes[1].axhline(10, color='k', ls=':', label='Nominal 10 ms')
    axes[2].step(gt, col(gui['rows'], 'missed_slots'), where='post', label='Recorded missed reference slots')
    for ax, text in zip(axes, ('Joint error (rad)', 'Time (ms)', 'Cumulative slots')):
        ax.set_ylabel(text)
        ax.axvline(gui['t'][-1], color='#66758b', ls='--', label='Tracking endpoint')
    save(fig, '21_graphical_timing_hold')


def main():
    pipe = [tracking(f'pipe/{mode}.csv', 'test pipe', mode, 1, summary)
            for mode, summary in zip(MODES, read_json('pipe/summary.json'))]
    live, trials = [], []
    for group in ('stage3', 'stage4_cartesian', 'stage4_regression'):
        for result in read_json(f'live/{group}/summary.json'):
            n, mode = result['repeat'], result['mode']
            key = f'live/{group}/{n:02d}_{mode}'
            summary = read_json(key+'/tracking.json')
            trace = read_rows(key+'/bridge_timing.csv')
            seq = {int(r['sequence']) for r in trace if r['event'] == 'valid_command'}
            completed = result['passed'] and result['endpoint_received']
            assert completed and not summary['fault'] and result['commands'] == summary['commands']
            assert summary['watchdog_events_observed'] == 0
            assert result['bridge_watchdogs_before_shutdown'] == 0
            assert len(seq - {0}) == result['valid_receipts']
            live.append(tracking(key+'/tracking.csv', f'live ROS {group}', mode, n, summary, result))
            trials.append({'dataset': group, 'mode': mode, 'trial': n, 'pass': completed,
                           'commands': result['commands'], 'valid_receipts': result['valid_receipts'],
                           'endpoint_received': result['endpoint_received'], 'fault': summary['fault'],
                           'source_summary': SOURCES[f'live/{group}/summary.json']})
    gui_summary = read_json('gui/gui_cartesian_flc.json')
    gui = tracking('gui/gui_cartesian_flc.csv', 'graphical ROS', 'cartesian_flc', 1, gui_summary)
    last = gui['rows'][-1]
    assert last['phase'] == 'terminal_hold' and float(last['reference_s']) == 4
    for name, key in [('tracking', 'stats'), ('timing', 'timing'), ('safety', 'safety'),
                      ('per_joint_rmse', 'per_joint')]:
        table(name, [d[key] for d in (*pipe, *live, gui)])
    table('live_trials', trials)
    groups = sorted({(r['dataset'], r['mode']) for r in trials})
    table('trial_counts', [{'dataset': group, 'mode': mode,
                            'passed': sum(r['pass'] for r in trials if (r['dataset'], r['mode']) == (group, mode)),
                            'failed': sum(not r['pass'] for r in trials if (r['dataset'], r['mode']) == (group, mode))}
                           for group, mode in groups])
    table('gui_terminal_sample', [{
        'elapsed_s': float(last['elapsed_s']), 'reference_s': float(last['reference_s']),
        'phase': last['phase'], 'joint_max_rad': float(abs(mat([last], 'error')).max()),
        'joint_rms_rad': rms(mat([last], 'error')),
        'source_csv': SOURCES['gui/gui_cartesian_flc.csv'],
    }])
    dynamics = read_json('offline/dynamics_saved.json')
    checked = read_json('dynamics_recheck.json')
    assert all(r['passed'] for r in checked['results'].values())
    table('dynamics', [{'term': name, 'unit': 'kg·m²' if name == 'mass' else 'N·m',
                        'max_abs': result['maximum'], 'mean_abs': result['mean'],
                        'tolerance': result['threshold'], 'pass': result['passed'],
                        'source_json': SOURCES['offline/dynamics_saved.json']}
                       for name, result in dynamics['results'].items()])
    singular = read_rows('offline/singularity.csv')
    table('singularity', singular)
    plots(pipe, gui, singular)


if __name__ == '__main__':
    main()
