from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from check_task_space_collision import (
    make_sample_times,
    run_collision_preflight,
    select_collision_geometries,
)


class FakeModel:
    def __init__(self):
        self.nbody = 5
        self.ngeom = 5
        self.nq = 6
        self.nv = 6
        self.njnt = 6
        self.body_parentid = np.array([0, 0, 1, 0, 0], dtype=int)
        self.geom_bodyid = np.array([2, 1, 3, 4, 0], dtype=int)
        self.geom_group = np.array([3, 2, 0, 0, 0], dtype=int)
        self.geom_contype = np.array([1, 0, 0, 0, 0], dtype=int)
        self.geom_conaffinity = np.array([1, 0, 0, 0, 0], dtype=int)
        self.jnt_type = np.zeros(6, dtype=int)
        self.jnt_qposadr = np.arange(6, dtype=int)
        self.jnt_limited = np.ones(6, dtype=bool)
        self.jnt_range = np.tile(np.array([-2.0, 2.0]), (6, 1))
        self.key_qpos = np.zeros((1, 6), dtype=float)


class FakeMuJoCo:
    __version__ = "3.2.0-fake"
    mjtObj = SimpleNamespace(
        mjOBJ_BODY="body", mjOBJ_GEOM="geom", mjOBJ_JOINT="joint", mjOBJ_KEY="key"
    )
    mjtJoint = SimpleNamespace(mjJNT_HINGE=0)

    body_names = {
        0: "world",
        1: "base",
        2: "shoulder_link",
        3: "robot_station",
        4: "tool_trolley",
    }
    geom_names = {
        0: "upperarm_collision",
        1: "base_visual",
        2: "robot_top",
        3: "trolley_top",
        4: "room_floor",
    }
    joint_names = {
        "shoulder_pan_joint": 0,
        "shoulder_lift_joint": 1,
        "elbow_joint": 2,
        "wrist_1_joint": 3,
        "wrist_2_joint": 4,
        "wrist_3_joint": 5,
    }
    distances: dict[tuple[int, int], float] = {}

    @classmethod
    def mj_name2id(cls, model, object_type, name):
        if object_type == "body":
            return {v: k for k, v in cls.body_names.items()}.get(name, -1)
        if object_type == "geom":
            return {v: k for k, v in cls.geom_names.items()}.get(name, -1)
        if object_type == "joint":
            return cls.joint_names.get(name, -1)
        if object_type == "key":
            return 0 if name == "home" else -1
        return -1

    @classmethod
    def mj_id2name(cls, model, object_type, object_id):
        if object_type == "body":
            return cls.body_names.get(object_id)
        if object_type == "geom":
            return cls.geom_names.get(object_id)
        return None

    @staticmethod
    def mj_forward(model, data):
        return None

    @classmethod
    def mj_geomDistance(cls, model, data, geom1, geom2, distmax, fromto):
        return cls.distances.get((geom1, geom2), distmax)


class FakeData:
    def __init__(self):
        self.qpos = np.zeros(6, dtype=float)
        self.qvel = np.zeros(6, dtype=float)


class FakeTrajectory:
    start_time_s = 0.0
    end_time_s = 1.0
    source_path = Path("fake_reference.csv")

    def __init__(self, final_q=None, start_q=None):
        self.start_q = np.zeros(6) if start_q is None else np.asarray(start_q, dtype=float)
        self.final_q = np.zeros(6) if final_q is None else np.asarray(final_q, dtype=float)

    def evaluate(self, time_s):
        fraction = min(max(float(time_s), 0.0), 1.0)
        q = self.start_q + (self.final_q - self.start_q) * fraction
        return SimpleNamespace(position_rad=q, phase="move" if time_s < 1.0 else "hold")


class TestGeometrySelection(unittest.TestCase):
    def setUp(self):
        self.model = FakeModel()

    def test_selects_collision_proxies_and_excludes_visuals_and_mount(self):
        selection = select_collision_geometries(self.model, FakeMuJoCo)
        self.assertEqual(selection.robot_geom_ids, (0,))
        self.assertEqual(selection.obstacle_geom_ids, (3, 4))
        self.assertNotIn("base_visual", selection.robot_geom_names)
        self.assertNotIn("robot_top", selection.obstacle_geom_names)

    def test_disabled_environment_contact_masks_do_not_remove_obstacles(self):
        selection = select_collision_geometries(self.model, FakeMuJoCo)
        self.assertEqual(selection.disabled_obstacle_geom_names, ("trolley_top", "room_floor"))
        self.assertEqual(len(selection.obstacle_geom_ids), 2)

    def test_missing_support_body_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "support body"):
            select_collision_geometries(
                self.model, FakeMuJoCo, excluded_body_names=("missing_support",)
            )


class TestCollisionPreflight(unittest.TestCase):
    def setUp(self):
        FakeMuJoCo.distances = {}
        self.model = FakeModel()
        self.data = FakeData()

    def _run(self, trajectory=None, **kwargs):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        summary = run_collision_preflight(
            self.model,
            self.data,
            FakeMuJoCo,
            trajectory or FakeTrajectory(),
            output_dir=temporary.name,
            sample_period_s=0.5,
            **kwargs,
        )
        return summary, Path(temporary.name)

    def test_exact_final_time_is_included_in_sample_grid(self):
        times = make_sample_times(0.0, 1.0, 0.3)
        np.testing.assert_allclose(times, [0.0, 0.3, 0.6, 0.9, 1.0], atol=1e-12)

    def test_clear_path_passes_with_capped_clearance_lower_bound(self):
        summary, output = self._run()
        self.assertEqual(summary["status"], "PASS")
        self.assertTrue(summary["safe_to_execute"])
        self.assertTrue(summary["minimum_distance_is_capped_lower_bound"])
        self.assertEqual(summary["minimum_clearance_lower_bound_m"], 0.05)
        self.assertTrue((output / "task_space_collision_preflight.csv").is_file())
        saved = json.loads((output / "task_space_collision_preflight.json").read_text())
        self.assertEqual(saved["status"], "PASS")

    def test_negative_signed_distance_fails_as_collision(self):
        FakeMuJoCo.distances[(0, 3)] = -0.004
        summary, _ = self._run()
        self.assertEqual(summary["status"], "FAIL_COLLISION")
        self.assertFalse(summary["safe_to_execute"])
        self.assertGreater(summary["collision_event_count"], 0)

    def test_clearance_below_minimum_fails(self):
        FakeMuJoCo.distances[(0, 3)] = 0.01
        summary, _ = self._run()
        self.assertEqual(summary["status"], "FAIL_CLEARANCE")
        self.assertAlmostEqual(summary["minimum_distance_m"], 0.01)

    def test_clearance_between_minimum_and_warning_requires_review(self):
        FakeMuJoCo.distances[(0, 3)] = 0.03
        summary, _ = self._run()
        self.assertEqual(summary["status"], "WARNING_REVIEW_REQUIRED")
        self.assertFalse(summary["safe_to_execute"])
        self.assertEqual(summary["minimum_distance_pair"]["environment_geom"], "trolley_top")

    def test_trajectory_start_must_match_home_keyframe(self):
        bad_start = np.array([0.1, 0, 0, 0, 0, 0], dtype=float)
        with self.assertRaisesRegex(ValueError, "does not match MuJoCo HOME"):
            self._run(FakeTrajectory(final_q=bad_start, start_q=bad_start))

    def test_joint_limit_violation_fails_preflight(self):
        out_of_range = np.array([2.2, 0, 0, 0, 0, 0], dtype=float)
        summary, _ = self._run(FakeTrajectory(final_q=out_of_range))
        self.assertEqual(summary["status"], "FAIL_JOINT_LIMIT")
        self.assertGreater(summary["joint_limit_violation_count"], 0)

    def test_warning_threshold_cannot_exceed_distance_cap(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "cannot exceed distance_cap_m"):
                run_collision_preflight(
                    self.model,
                    self.data,
                    FakeMuJoCo,
                    FakeTrajectory(),
                    output_dir=temporary,
                    distance_cap_m=0.04,
                    warning_clearance_m=0.05,
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
