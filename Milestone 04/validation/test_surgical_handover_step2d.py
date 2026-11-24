#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

APP_DIR = Path(__file__).resolve().parents[1] / "application"
import sys
sys.path.insert(0, str(APP_DIR))

from surgical_handover_planner import (  # noqa: E402
    build_surgical_handover_plan,
    validate_transform,
)
from full_pose_ik import HOME  # noqa: E402
from ur5e_dh_fk import forward_kinematics  # noqa: E402
from surgical_handover_collision_preflight import (  # noqa: E402
    REQUIRED_WAYPOINTS,
    load_plan,
    pickup_support_contact_allowed,
    _select_geometries,
)


class SurgicalHandoverStep2DTests(unittest.TestCase):
    def test_waypoint_sequence_contains_clearance_route(self):
        self.assertEqual(REQUIRED_WAYPOINTS[0], "approach_clearance")
        self.assertEqual(REQUIRED_WAYPOINTS[-1], "return_home")
        self.assertIn("handover", REQUIRED_WAYPOINTS)

    def test_pickup_support_contact_only_allowed_during_pickup_and_lift(self):
        self.assertTrue(pickup_support_contact_allowed("pickup", True, False))
        self.assertTrue(pickup_support_contact_allowed("lift", True, False))
        self.assertFalse(pickup_support_contact_allowed("transfer_clearance", True, False))
        self.assertFalse(pickup_support_contact_allowed("pickup", False, False))
        self.assertFalse(pickup_support_contact_allowed("lift", True, True))

    def test_pickup_support_envelope_includes_only_tray_and_trolley_top(self):
        try:
            import mujoco
        except ImportError:
            self.skipTest("MuJoCo is not installed in this test environment")

        project_root = Path(__file__).resolve().parents[2]
        scene = project_root / "Milestone 01" / "Robot Model" / "scene_surgical_handover.xml"
        if not scene.is_file():
            self.skipTest(f"Surgical handover scene is not present: {scene}")

        model = mujoco.MjModel.from_xml_path(str(scene))
        selected = _select_geometries(mujoco, model, "scalpel")
        names = set(selected["pickup_support_geom_names"])

        self.assertEqual(
            names,
            {"tray_base", "tray_scalpel_pad", "trolley_top"},
        )
        # Do not exempt the trolley rim, shelf, legs, or other scene geometry.
        self.assertNotIn("trolley_top_rim", names)
        self.assertNotIn("trolley_shelf", names)

    def test_load_plan_rejects_old_waypoint_sequence(self):
        payload = {
            "instrument": "scalpel",
            "initial_joint_positions_rad": [0.0] * 6,
            "waypoints": [
                {"name": name, "joint_positions_rad": [0.0] * 6,
                 "duration_from_previous_s": 1.0, "ik_success": True}
                for name in REQUIRED_WAYPOINTS[1:]
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plan.json"
            path.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                load_plan(path, "scalpel")

    def test_load_plan_accepts_new_waypoint_sequence(self):
        payload = {
            "instrument": "forceps",
            "initial_joint_positions_rad": [0.0] * 6,
            "waypoints": [
                {"name": name, "joint_positions_rad": [0.0] * 6,
                 "duration_from_previous_s": 1.0, "ik_success": True}
                for name in REQUIRED_WAYPOINTS
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plan.json"
            path.write_text(json.dumps(payload))
            loaded = load_plan(path, "forceps")
            self.assertEqual(tuple(x["name"] for x in loaded["waypoints"]), REQUIRED_WAYPOINTS)

    def test_relative_transform_compensates_tool_target(self):
        # Instrument body is 35 mm below the pickup-site/attachment target.
        tool_pick = np.eye(4)
        instrument_initial = np.eye(4)
        instrument_initial[2, 3] = -0.035
        tool_from_instrument = np.linalg.inv(tool_pick) @ instrument_initial
        desired_instrument = np.eye(4)
        desired_instrument[:3, 3] = [0.3, 0.43, 0.38]
        desired_tool = desired_instrument @ np.linalg.inv(tool_from_instrument)
        reconstructed_instrument = desired_tool @ tool_from_instrument
        self.assertTrue(np.allclose(reconstructed_instrument, desired_instrument, atol=1e-12))
        self.assertAlmostEqual(desired_tool[2, 3], 0.415, places=12)

    def test_planner_generates_compensated_nine_waypoint_plan_for_both_instruments(self):
        attachment = forward_kinematics(HOME)
        handover = np.eye(4)
        handover[:3, 3] = [0.3, 0.43, 0.38]
        for instrument, pickup_xyz in (
            ("scalpel", [0.75, -0.13, 0.136]),
            ("forceps", [0.75, -0.29, 0.136]),
        ):
            pickup = np.eye(4)
            pickup[:3, 3] = pickup_xyz
            instrument_body = np.eye(4)
            instrument_body[:3, 3] = [pickup_xyz[0], pickup_xyz[1], 0.101]
            plan = build_surgical_handover_plan(
                instrument=instrument,
                initial_joint_positions_rad=HOME,
                pickup_pose_world=pickup,
                handover_pose_world=handover,
                attachment_pose_world=attachment,
                instrument_body_pose_world=instrument_body,
            )
            self.assertEqual(len(plan.waypoints), 9)
            self.assertEqual(plan.waypoints[0].name, "approach_clearance")
            self.assertEqual(plan.waypoints[-1].name, "return_home")
            self.assertAlmostEqual(
                np.asarray(plan.desired_tool_handover_pose_world)[2, 3], 0.415, places=6
            )

    def test_transform_validation_rejects_non_rigid_rotation(self):
        bad = np.eye(4)
        bad[0, 0] = 2.0
        with self.assertRaises(ValueError):
            validate_transform(bad, "bad")


if __name__ == "__main__":
    unittest.main(verbosity=2)
