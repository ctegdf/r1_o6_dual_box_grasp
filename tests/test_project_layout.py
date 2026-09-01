from __future__ import annotations

import ast
import hashlib
import json
import re
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from scripts.prepare_curobo_configs import (
    CONFIG_NAMES,
    CONFIG_TOKEN,
    materialize_configs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_TEXT_SUFFIXES = {".json", ".md", ".py", ".sh", ".toml", ".yaml", ".yml"}
IGNORED_SCAN_PARTS = {
    ".git",
    ".omx",
    ".runtime",
    "__pycache__",
    "artifacts",
    "assets",
    "recordings",
    "runs",
}


class ProjectLayoutTests(unittest.TestCase):
    def test_scene_object_library_contains_only_supported_boxes(self):
        source = (PROJECT_ROOT / "delivery_objects_cfg.py").read_text(
            encoding="utf-8",
        )
        tree = ast.parse(source)
        keys: set[str] | None = None
        for node in tree.body:
            if isinstance(node, ast.AnnAssign):
                target = node.target
                if isinstance(target, ast.Name) and target.id == "DELIVERY_OBJECTS":
                    self.assertIsInstance(node.value, ast.Dict)
                    keys = {
                        key.value
                        for key in node.value.keys
                        if isinstance(key, ast.Constant)
                    }
                    break
        self.assertEqual(keys, {"courier_box_m", "foam_box"})

    def test_required_robot_sources_are_present(self):
        required = (
            "R1-o6.urdf",
            "meshes/pelvis_link.STL",
            "meshes/o6_left/hand_base_link.STL",
            "meshes/o6_right/hand_base_link.STL",
            "scripts/convert_r1_o6_urdf.py",
        )
        for relative in required:
            self.assertTrue((PROJECT_ROOT / relative).is_file(), relative)

    def test_every_urdf_mesh_reference_exists(self):
        root = ET.parse(PROJECT_ROOT / "R1-o6.urdf").getroot()
        references = {
            element.attrib["filename"]
            for element in root.iter("mesh")
            if "filename" in element.attrib
        }
        self.assertTrue(references)
        for reference in references:
            self.assertTrue((PROJECT_ROOT / reference).is_file(), reference)

    def test_no_source_file_exceeds_github_single_file_limit(self):
        limit_bytes = 100 * 1024 * 1024
        oversized = [
            path.relative_to(PROJECT_ROOT)
            for path in PROJECT_ROOT.rglob("*")
            if path.is_file()
            and ".runtime" not in path.parts
            and path.stat().st_size >= limit_bytes
        ]
        self.assertEqual(oversized, [])

    def test_config_templates_use_portable_project_root(self):
        for name in CONFIG_NAMES:
            text = (PROJECT_ROOT / "configs" / name).read_text(encoding="utf-8")
            self.assertIn(CONFIG_TOKEN, text)

    def test_published_text_does_not_contain_local_environment_identifiers(self):
        local_path = re.compile(r"/(?:home|Users)/|[A-Za-z]:\\Users\\")
        private_ip = re.compile(
            r"10\.\d{1,3}\.\d{1,3}\.\d{1,3}"
            r"|192\.168\.\d{1,3}\.\d{1,3}"
            r"|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}"
        )
        email = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
        violations = []
        for path in PROJECT_ROOT.rglob("*"):
            if (
                not path.is_file()
                or any(part in IGNORED_SCAN_PARTS for part in path.parts)
                or (path.suffix not in PUBLIC_TEXT_SUFFIXES and path.name != "LICENSE")
            ):
                continue
            text = path.read_text(encoding="utf-8")
            if local_path.search(text) or private_ip.search(text) or email.search(text):
                violations.append(path.relative_to(PROJECT_ROOT))
        self.assertEqual(violations, [])

    def test_server_defaults_to_loopback(self):
        source = (PROJECT_ROOT / "pose_grasp_server.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        host_defaults = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            if not (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "--host"
            ):
                continue
            host_defaults.extend(
                keyword.value.value
                for keyword in node.keywords
                if keyword.arg == "default"
                and isinstance(keyword.value, ast.Constant)
            )
        self.assertEqual(host_defaults, ["127.0.0.1"])

    def test_config_materialization_uses_absolute_project_paths(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir) / "configs"
            written = materialize_configs(PROJECT_ROOT, output_dir)
            self.assertEqual({path.name for path in written}, set(CONFIG_NAMES))
            for path in written:
                text = path.read_text(encoding="utf-8")
                self.assertNotIn(CONFIG_TOKEN, text)
                self.assertIn(str(PROJECT_ROOT / "R1-o6.urdf"), text)
                self.assertIn(f'asset_root_path: "{PROJECT_ROOT}"', text)

    def test_validated_server_baselines_match_project_files(self):
        baseline_path = PROJECT_ROOT / "configs" / "validated_runs.json"
        data = json.loads(baseline_path.read_text(encoding="utf-8"))
        profiles = data["profiles"]
        self.assertEqual(set(profiles), {"courier_box_m", "foam_box"})

        digest = hashlib.sha256(
            (PROJECT_ROOT / "auto_clamp.py").read_bytes()
        ).hexdigest()
        self.assertEqual(digest, data["provenance"]["auto_clamp_sha256"])
        for name, expected_digest in data["provenance"][
            "curobo_template_sha256"
        ].items():
            digest = hashlib.sha256(
                (PROJECT_ROOT / "configs" / name).read_bytes()
            ).hexdigest()
            self.assertEqual(digest, expected_digest, name)

        scene_tree = ast.parse(
            (PROJECT_ROOT / "r1_o6_scene_cfg.py").read_text(encoding="utf-8")
        )
        init_x = None
        for node in scene_tree.body:
            if (
                isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name)
                    and target.id == "VALIDATED_BOX_INIT_X"
                    for target in node.targets
                )
            ):
                init_x = ast.literal_eval(node.value)
                break
        self.assertEqual(
            init_x,
            {
                key: profile["object_initial_world_x_m"]
                for key, profile in profiles.items()
            },
        )

        operations = (PROJECT_ROOT / "docs" / "OPERATIONS.md").read_text(
            encoding="utf-8"
        )
        for key, profile in profiles.items():
            start = operations.index(f"--object {key}")
            end = operations.index("```", start)
            command = operations[start:end]
            self.assertIn(
                f'--robot-root-x-offset-m {profile["robot_root_x_offset_m"]}',
                command,
            )
            self.assertIn(
                f'--clamp-x-offset-m {profile["clamp_x_offset_m"]}',
                command,
            )
            self.assertIn(
                f'--clamp-force-target-n {profile["force_target_n"]}',
                command,
            )


if __name__ == "__main__":
    unittest.main()
