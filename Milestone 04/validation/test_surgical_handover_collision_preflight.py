from __future__ import annotations

import csv
import math
import tempfile
from unittest.mock import patch
import sys
import unittest
from pathlib import Path

import numpy as np

APP = Path(__file__).resolve().parents[1] / "application"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from surgical_handover_collision_preflight import (  # noqa: E402
    DEFAULT_SCENE,
    JOINT_NAMES,
    _enable_native_ccd,
    _record_pair,
    _select_geometries,
    _update_instrument_mocap,
    pickup_support_pairs,
    run_preflight,
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
        plan = self.make_plan("forceps")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plan.json"
            path.write_text(json.dumps(plan), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_plan(path, "scalpel")

    def test_load_plan_accepts_expected_waypoint_sequence(self):
        import json
        plan = self.make_plan("forceps")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plan.json"
            path.write_text(json.dumps(plan), encoding="utf-8")
            loaded = load_plan(path, "forceps")
            self.assertEqual(loaded["instrument"], "forceps")
            self.assertEqual(loaded["_initial_q"].shape, (6,))

    def test_saved_pickup_overlap_is_reported_and_corrected_scene_is_clear(self):
        import mujoco

        results = APP.parent / "Results" / "Surgical_Handover"
        plan = load_plan(
            results / "Planning/surgical_handover_plan_scalpel_20261124_032955_257327.json",
            "scalpel",
        )
        csv_path = results / "Execution/surgical_handover_execution_scalpel_20261124_113401_483815.csv"
        with csv_path.open() as stream:
            row = list(csv.DictReader(stream))[-1]
        self.assertEqual(row["phase"], "pickup_settle")
        q = [float(row[f"q_actual_{name}_rad"]) for name in JOINT_NAMES]

        model = mujoco.MjModel.from_xml_path(str(DEFAULT_SCENE))
        _enable_native_ccd(mujoco, model)
        data = mujoco.MjData(model)
        selection = _select_geometries(mujoco, model, "scalpel")
        mocap = selection["instrument_mocap_id"]
        data.qpos[:] = q
        for z, expected_mm in ((0.101, -3.0), (0.105, 1.0)):
            with self.subTest(z=z):
                data.mocap_pos[mocap, 2] = z
                mujoco.mj_forward(model, data)
                site = data.site("attachment_site")
                body = data.body("scalpel")
                relative = compute_attachment_relative_pose(
                    site.xpos, site.xmat.reshape(3, 3), body.xpos, body.xmat.reshape(3, 3),
                )
                _update_instrument_mocap(mujoco, model, data, selection, *relative)
                minimum, a, b, _, count, _ = _record_pair(
                    mujoco, model, data, pickup_support_pairs(selection),
                    0.05, 0.05, "pickup_support", 0.0, "lift", allow_touch=True,
                )
                self.assertAlmostEqual(minimum * 1000, expected_mm, places=6)
                self.assertEqual((a, b), ("scalpel_grip", "tray_scalpel_pad"))
                self.assertEqual(count, int(expected_mm < 0))

        # Restore the historical XML placement in a private scene and run the
        # same preflight entry point that previously missed the negative distance.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for entry in DEFAULT_SCENE.parent.iterdir():
                if entry.name != DEFAULT_SCENE.name:
                    (root / entry.name).symlink_to(entry)
            historical = root / DEFAULT_SCENE.name
            historical.write_text(DEFAULT_SCENE.read_text().replace(
                'name="scalpel" pos="0.75 -0.13 0.105"',
                'name="scalpel" pos="0.75 -0.13 0.101"',
            ).replace(
                'name="forceps" pos="0.75 -0.29 0.104"',
                'name="forceps" pos="0.75 -0.29 0.101"',
            ))
            samples, attach, release = build_quintic_samples(plan, 0.005)
            endpoints = [samples[0], next(s for s in samples if s.time_s == attach),
                         next(s for s in samples if s.phase == "lift")]
            with patch("surgical_handover_collision_preflight.build_quintic_samples",
                       return_value=(endpoints, attach, release)):
                result = run_preflight(
                    plan, historical, root, sample_period_s=0.005,
                    distance_cap_m=0.05, minimum_clearance_m=0.02, warning_clearance_m=0.05,
                )
            self.assertEqual(result["status"], "FAIL_COLLISION")
            event = next(event for event in result["collision_events"] if event["phase"] == "pickup")
            self.assertEqual(event["category"], "pickup_support")
            self.assertEqual((event["geom_a"], event["geom_b"]), ("scalpel_grip", "tray_scalpel_pad"))
            self.assertAlmostEqual(event["signed_distance_m"], -0.003, places=9)

    def test_robot_environment_and_self_pairs_still_report_overlap(self):
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(DEFAULT_SCENE))
        _enable_native_ccd(mujoco, model)
        selection = _select_geometries(mujoco, model, "scalpel")
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        first, second = selection["self_pairs"][-1]
        pad = model.geom("tray_scalpel_pad").id
        self.assertIn(pad, selection["all_environment_geoms"])
        for category, other in (("robot_environment", pad), ("robot_self", second)):
            with self.subTest(category=category):
                mujoco.mj_forward(model, data)
                data.geom_xpos[other] = data.geom_xpos[first] + np.array([0.01, 0.01, 0.01])
                minimum, _, _, _, count, events = _record_pair(
                    mujoco, model, data, [(first, other)], 0.05, 0.05, category, 0.0, "lift",
                )
                self.assertLess(minimum, 0.0)
                self.assertEqual(count, 1)
                self.assertEqual(events[0]["category"], category)

    def test_support_allows_exact_touch_and_rejects_negative_distance(self):
        # Use exact values to pin the classification boundary independently of CCD rounding.
        for distance in (0.0, -1e-12):
            with patch("surgical_handover_collision_preflight._distance", return_value=distance), \
                 patch("surgical_handover_collision_preflight._geom_name", side_effect=("grip", "pad")):
                result = _record_pair(None, None, None, [(0, 1)], 0.05, 0.05,
                                      "pickup_support", 0.0, "lift", allow_touch=True)
            self.assertEqual(result[4], int(distance < 0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
