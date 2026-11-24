from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

APP = Path(__file__).resolve().parents[1] / "application"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from surgical_handover_collision_preflight import (  # noqa: E402
    build_quintic_samples,
    compute_attachment_relative_pose,
    load_plan,
    rotation_matrix_to_quaternion_wxyz,
)


class SurgicalHandoverCollisionPreflightTests(unittest.TestCase):
    def make_plan(self, instrument="scalpel"):
        q0 = [0.0] * 6
        waypoints = []
        names = (
            "approach_clearance",
            "pregrasp",
            "pickup",
            "lift",
            "transfer_clearance",
            "handover_approach",
            "handover",
            "retreat",
            "return_home",
        )
        for i, name in enumerate(names, start=1):
            q = [0.01 * i] * 6
            waypoints.append({
                "name": name,
                "joint_positions_rad": q,
                "duration_from_previous_s": 0.02,
                "ik_success": True,
            })
        return {
            "instrument": instrument,
            "initial_joint_positions_rad": q0,
            "waypoints": waypoints,
        }

    def test_quintic_samples_include_start_and_final_endpoint(self):
        plan = self.make_plan()
        plan["_initial_q"] = np.zeros(6)
        samples, attach_time, release_time = build_quintic_samples(plan, 0.005)
        self.assertAlmostEqual(samples[0].time_s, 0.0)
        self.assertAlmostEqual(samples[-1].time_s, 0.18)
        self.assertTrue(np.allclose(samples[-1].q, np.full(6, 0.09)))
        self.assertAlmostEqual(attach_time, 0.06)
        self.assertAlmostEqual(release_time, 0.14)
        self.assertEqual(samples[-1].phase, "return_home")

    def test_rotation_identity_quaternion(self):
        q = rotation_matrix_to_quaternion_wxyz(np.eye(3))
        self.assertTrue(np.allclose(q, [1.0, 0.0, 0.0, 0.0]))

    def test_rotation_quarter_turn_about_z(self):
        angle = math.pi / 2.0
        r = np.array([
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ])
        q = rotation_matrix_to_quaternion_wxyz(r)
        self.assertTrue(np.allclose(q, [math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)], atol=1e-8))

    def test_attachment_relative_pose_identity(self):
        relative_p, relative_r = compute_attachment_relative_pose(
            np.array([1.0, 2.0, 3.0]), np.eye(3),
            np.array([1.1, 2.2, 3.3]), np.eye(3),
        )
        self.assertTrue(np.allclose(relative_p, [0.1, 0.2, 0.3]))
        self.assertTrue(np.allclose(relative_r, np.eye(3)))

    def test_load_plan_rejects_wrong_instrument(self):
        import json
        import tempfile
        plan = self.make_plan("forceps")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plan.json"
            path.write_text(json.dumps(plan), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_plan(path, "scalpel")

    def test_load_plan_accepts_expected_waypoint_sequence(self):
        import json
        import tempfile
        plan = self.make_plan("forceps")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plan.json"
            path.write_text(json.dumps(plan), encoding="utf-8")
            loaded = load_plan(path, "forceps")
            self.assertEqual(loaded["instrument"], "forceps")
            self.assertEqual(loaded["_initial_q"].shape, (6,))


if __name__ == "__main__":
    unittest.main(verbosity=2)
