#!/usr/bin/env python3
"""Offline unit tests for surgical-handover execution gates and trajectory assembly."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

APP_DIR = Path(__file__).resolve().parents[1] / "application"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from surgical_handover_execution_core import (  # noqa: E402
    ExecutionValidationError,
    build_execution_segments,
    instrument_body_pose_from_pickup_site,
    load_execution_plan,
    transform_error,
    validate_pickup_geometry,
    validate_preflight_for_execution,
)

WAYPOINTS = (
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


def make_plan() -> dict:
    identity = np.eye(4).tolist()
    q0 = np.zeros(6)
    waypoints = []
    for index, name in enumerate(WAYPOINTS, start=1):
        q = np.full(6, index * 0.01)
        waypoints.append(
            {
                "name": name,
                "position_world_m": [0.1, 0.2, 0.3],
                "joint_positions_rad": q.tolist(),
                "duration_from_previous_s": 1.0,
                "ik_success": True,
                "ik_iterations": 1,
                "ik_position_error_m": 0.0,
                "ik_orientation_error_rad": 0.0,
                "ik_message": "Converged",
            }
        )
    return {
        "instrument": "scalpel",
        "world_from_dh": identity,
        "initial_joint_positions_rad": q0.tolist(),
        "tool_from_instrument": identity,
        "desired_instrument_handover_pose_world": identity,
        "desired_tool_handover_pose_world": identity,
        "instrument_body_initial_pose_world": identity,
        "planning_only": True,
        "motion_executed": False,
        "collision_check_performed": False,
        "actuator_lower_limits_rad": [-6.0] * 6,
        "actuator_upper_limits_rad": [6.0] * 6,
        "scene_xml": "/tmp/scene.xml",
        "waypoints": waypoints,
    }


def make_preflight(plan_path: Path, scene_path: Path, *, status: str = "PASS") -> dict:
    warning = status == "WARNING_REVIEW_REQUIRED"
    return {
        "status": status,
        "preflight_passed": status == "PASS",
        "instrument": "scalpel",
        "plan_json": str(plan_path.resolve()),
        "scene_xml": str(scene_path.resolve()),
        "scene_xml_sha256": hashlib.sha256(scene_path.read_bytes()).hexdigest(),
        "mujoco_version": "3.2.7",
        "collision_distance_backend": {
            "name": "MuJoCo native CCD",
            "native_ccd_enabled": True,
            "enable_flag": "mjtEnableBit.mjENBL_NATIVECCD",
        },
        "model_dimensions": {"nq": 6, "nv": 6, "nu": 6},
        "trajectory": {
            "sample_count": 100,
            "sample_period_s": 0.005,
            "duration_s": 9.0,
            "pickup_attachment_time_s": 3.0,
            "handover_release_time_s": 7.0,
            "segments": list(WAYPOINTS),
            "interpolation": "project QuinticJointTrajectory point-to-point segments",
        },
        "clearance_thresholds": {
            "distance_cap_m": 0.05,
            "minimum_clearance_m": 0.02,
            "warning_clearance_m": 0.05,
        },
        "minimum_clearances": {
            "robot_environment": {"minimum_distance_m": 0.021 if warning else 0.05},
            "robot_self": {"minimum_distance_m": 0.044 if warning else 0.05},
            "carried_instrument_environment": {"minimum_distance_m": 0.0205 if warning else 0.05},
        },
        "collision_event_count": 0,
        "joint_limit_violation_count": 0,
        "closest_pairs_below_warning": (
            [
                {
                    "category": "robot_environment",
                    "geom_a": "geom_33",
                    "geom_b": "handover_glove_palm",
                    "minimum_distance_m": 0.021,
                    "time_s": 8.0,
                    "phase": "handover",
                }
            ] if warning else []
        ),
        "pickup_support_contact": {
            "support_geometries": ["tray_base", "tray_scalpel_pad", "trolley_top"],
            "allowed_only_during_pickup_and_lift": True,
            "penetration_allowed": False,
            "signed_support_distances_checked_for_penetration": True,
            "cleared": True,
            "clearance_violation": False,
        },
    }


class SurgicalHandoverExecutionCoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.plan_path = self.root / "plan.json"
        self.scene_path = self.root / "scene.xml"
        self.scene_path.write_text("<mujoco/>", encoding="utf-8")
        payload = make_plan()
        payload["scene_xml"] = str(self.scene_path.resolve())
        self.plan_path.write_text(json.dumps(payload), encoding="utf-8")
        self.plan = load_execution_plan(self.plan_path, "scalpel")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_load_plan_accepts_expected_nine_waypoint_sequence(self) -> None:
        self.assertEqual(tuple(item["name"] for item in self.plan["waypoints"]), WAYPOINTS)
        self.assertEqual(self.plan["instrument"], "scalpel")

    def test_load_plan_rejects_wrong_instrument(self) -> None:
        with self.assertRaises(ExecutionValidationError):
            load_execution_plan(self.plan_path, "forceps")

    def test_load_plan_rejects_motion_executed_metadata(self) -> None:
        payload = make_plan()
        payload["motion_executed"] = True
        self.plan_path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(ExecutionValidationError):
            load_execution_plan(self.plan_path, "scalpel")

    def test_segments_reconstruct_all_waypoints_contiguously(self) -> None:
        segments = build_execution_segments(self.plan)
        self.assertEqual(tuple(segment.name for segment in segments), WAYPOINTS)
        self.assertEqual(segments[0].start_position_rad.tolist(), [0.0] * 6)
        for first, second in zip(segments, segments[1:]):
            np.testing.assert_allclose(first.goal_position_rad, second.start_position_rad)
        self.assertAlmostEqual(sum(segment.duration_s for segment in segments), 9.0)

    def test_pass_preflight_is_accepted(self) -> None:
        preflight = make_preflight(self.plan_path, self.scene_path)
        result = validate_preflight_for_execution(
            preflight,
            plan=self.plan,
            plan_path=self.plan_path,
            scene_path=self.scene_path,
        )
        self.assertEqual(result["status"], "PASS")
        self.assertFalse(result["warning_review_approved"])

    def test_warning_preflight_requires_explicit_approval_for_execution(self) -> None:
        preflight = make_preflight(
            self.plan_path, self.scene_path, status="WARNING_REVIEW_REQUIRED"
        )
        result = validate_preflight_for_execution(
            preflight,
            plan=self.plan,
            plan_path=self.plan_path,
            scene_path=self.scene_path,
            approve_warnings=False,
        )
        self.assertEqual(result["status"], "WARNING_REVIEW_REQUIRED")
        self.assertFalse(result["warning_review_approved"])
        approved = validate_preflight_for_execution(
            preflight,
            plan=self.plan,
            plan_path=self.plan_path,
            scene_path=self.scene_path,
            approve_warnings=True,
        )
        self.assertTrue(approved["warning_review_approved"])

    def test_preflight_rejects_legacy_report_and_changed_scene(self) -> None:
        report = make_preflight(self.plan_path, self.scene_path)
        del report["pickup_support_contact"]["signed_support_distances_checked_for_penetration"]
        with self.assertRaisesRegex(ExecutionValidationError, "signed pickup-support"):
            validate_preflight_for_execution(
                report, plan=self.plan, plan_path=self.plan_path, scene_path=self.scene_path,
            )
        report = make_preflight(self.plan_path, self.scene_path)
        self.scene_path.write_text("<mujoco model='changed'/>", encoding="utf-8")
        with self.assertRaisesRegex(ExecutionValidationError, "scene XML digest"):
            validate_preflight_for_execution(
                report, plan=self.plan, plan_path=self.plan_path, scene_path=self.scene_path,
            )

    def test_preflight_rejects_collision(self) -> None:
        preflight = make_preflight(self.plan_path, self.scene_path)
        preflight["collision_event_count"] = 1
        with self.assertRaises(ExecutionValidationError):
            validate_preflight_for_execution(
                preflight, plan=self.plan, plan_path=self.plan_path, scene_path=self.scene_path
            )

    def test_preflight_rejects_clearance_below_hard_minimum_even_when_approved(self) -> None:
        preflight = make_preflight(
            self.plan_path, self.scene_path, status="WARNING_REVIEW_REQUIRED"
        )
        preflight["minimum_clearances"]["robot_environment"]["minimum_distance_m"] = 0.019
        with self.assertRaises(ExecutionValidationError):
            validate_preflight_for_execution(
                preflight,
                plan=self.plan,
                plan_path=self.plan_path,
                scene_path=self.scene_path,
                approve_warnings=True,
            )

    def test_preflight_rejects_wrong_plan_path(self) -> None:
        preflight = make_preflight(self.root / "other.json", self.scene_path)
        with self.assertRaises(ExecutionValidationError):
            validate_preflight_for_execution(
                preflight, plan=self.plan, plan_path=self.plan_path, scene_path=self.scene_path
            )

    def test_preflight_rejects_support_clearance_violation(self) -> None:
        preflight = make_preflight(self.plan_path, self.scene_path)
        preflight["pickup_support_contact"]["clearance_violation"] = True
        with self.assertRaises(ExecutionValidationError):
            validate_preflight_for_execution(
                preflight, plan=self.plan, plan_path=self.plan_path, scene_path=self.scene_path
            )


    def test_pickup_site_orientation_offset_is_valid_grasp_geometry(self) -> None:
        # The pickup site and attachment site occupy the same position, but
        # the planned robot tool orientation is intentionally about 180 deg
        # different. This offset is captured as T_tool_instrument at grasp.
        pickup_site = np.eye(4)
        pickup_site[:3, 3] = [0.75, -0.13, 0.136]
        planned_tool = pickup_site.copy()
        planned_tool[:3, :3] = np.diag([1.0, -1.0, -1.0])
        metrics = validate_pickup_geometry(
            pickup_site,
            pickup_site,
            planned_tool,
            site_position_tolerance_m=0.02,
            site_orientation_tolerance_rad=0.10,
            tool_site_position_tolerance_m=0.02,
        )
        self.assertAlmostEqual(metrics["tool_to_pickup_site_position_error_m"], 0.0)
        self.assertGreater(
            metrics["planned_tool_to_pickup_site_orientation_offset_rad"],
            np.deg2rad(179.0),
        )

    def test_pickup_geometry_rejects_instrument_site_drift(self) -> None:
        actual = np.eye(4)
        expected = np.eye(4)
        expected[0, 3] = 0.03
        with self.assertRaises(ExecutionValidationError):
            validate_pickup_geometry(
                actual,
                expected,
                expected,
                site_position_tolerance_m=0.02,
                site_orientation_tolerance_rad=0.10,
                tool_site_position_tolerance_m=0.02,
            )

    def test_pickup_geometry_checks_actual_tool_pose_at_grasp(self) -> None:
        pickup_site = np.eye(4)
        planned_tool = pickup_site.copy()
        planned_tool[:3, :3] = np.diag([1.0, -1.0, -1.0])
        actual_tool = planned_tool.copy()
        actual_tool[0, 3] += 0.03
        with self.assertRaises(ExecutionValidationError):
            validate_pickup_geometry(
                pickup_site,
                pickup_site,
                planned_tool,
                site_position_tolerance_m=0.02,
                site_orientation_tolerance_rad=0.10,
                tool_site_position_tolerance_m=0.02,
                actual_tool_pose=actual_tool,
                tool_position_tolerance_m=0.02,
                tool_orientation_tolerance_rad=0.10,
            )

    def test_transform_error_reports_translation_and_rotation(self) -> None:
        actual = np.eye(4)
        expected = np.eye(4)
        expected[0, 3] = 0.01
        position_error, orientation_error = transform_error(actual, expected)
        self.assertAlmostEqual(position_error, 0.01)
        self.assertAlmostEqual(orientation_error, 0.0)

    def test_instrument_body_pose_is_recovered_from_pickup_site(self) -> None:
        world_from_body = np.eye(4)
        world_from_body[:3, 3] = [0.3, 0.4, 0.5]
        body_from_site = np.eye(4)
        body_from_site[:3, 3] = [0.0, 0.0, 0.035]
        world_from_site = world_from_body @ body_from_site
        recovered = instrument_body_pose_from_pickup_site(world_from_site, body_from_site)
        np.testing.assert_allclose(recovered, world_from_body, atol=1e-12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
