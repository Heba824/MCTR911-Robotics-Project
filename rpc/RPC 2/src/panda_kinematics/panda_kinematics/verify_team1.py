#!/usr/bin/env python3
"""Independent Team 1 joint-limit and endpoint checks (does not run ROS)."""
from pathlib import Path
import ast
import numpy as np
from verify_rpc2 import load_functions

HERE = Path(__file__).resolve().parent
SRC = HERE / 'trajectory_node.py'


def read_config():
    """Read only the Q_START / Q_END assignment statements from source."""
    tree = ast.parse(SRC.read_text(), filename=str(SRC))
    selected = []
    for statement in tree.body:
        if isinstance(statement, ast.Assign) and len(statement.targets)==1:
            if isinstance(statement.targets[0], ast.Name) and statement.targets[0].id in ('Q_START', 'Q_END'):
                selected.append(statement)
    if len(selected) != 2:
        raise AssertionError('Expected exactly one Q_START and Q_END assignment')
    vars_ = {'np': np}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(SRC), 'exec'), vars_)
    return vars_['Q_START'], vars_['Q_END']


def main():
    f = load_functions()
    q0, q1 = read_config()
    fk, plan = f['forward_kinematics'], f['plan_trajectory']
    qmin, qmax = f['Q_MIN'], f['Q_MAX']
    t, q, dq, ddq = plan(q0, q1, 4.0, 0.05)
    assert q.shape == dq.shape == ddq.shape == (81, 7)
    assert np.allclose(q[0], q0) and np.allclose(q[-1], q1)
    assert np.all(q >= qmin - 1e-10) and np.all(q <= qmax + 1e-10)
    assert np.allclose(dq[[0,-1]],0) and np.allclose(ddq[[0,-1]],0)
    assert np.all(q1 != q0), 'All 7 joints must move for the team demonstration'
    # Monotonicity is the key no-overshoot property for single-segment quintic.
    lo = np.minimum(q0,q1); hi = np.maximum(q0,q1)
    assert np.all(q>=lo-1e-10) and np.all(q<=hi+1e-10)
    # Stronger proof than samples: for u in [0,1], ds/du=30u²(1-u)²>=0.
    p0, p1 = fk(q0)[:3,3], fk(q1)[:3,3]
    assert p1[0]>p0[0] and p1[2]>p0[2], 'Forward reach and lift not achieved'
    assert abs(p1[1])<0.025, 'End pose not near centerline'
    print('PASS: Team 1 start/end inside all seven joint limits')
    print('PASS: all seven joints move moderately; quintic never overshoots endpoint ranges')
    print('PASS: 81 samples, zero endpoint velocities and accelerations')
    print('Start q (rad):', np.round(q0,5))
    print('End   q (rad):', np.round(q1,5))
    print('Start hand XYZ (m):', np.round(p0,4))
    print('End hand XYZ   (m):', np.round(p1,4))
    print('Hand displacement (m):',np.round(p1-p0,4))
    print('Maximum |dq| (rad/s):', round(float(np.max(np.abs(dq))),4))
    print('Maximum |ddq| (rad/s²):', round(float(np.max(np.abs(ddq))),4))
    margins = np.minimum(q-qmin, qmax-q).min(axis=0)
    print('Minimum margin per joint (rad):', np.round(margins,4))
    print('Mathematical team test passed; check actual MuJoCo dynamics/collisions separately.')

if __name__ == '__main__':
    main()
