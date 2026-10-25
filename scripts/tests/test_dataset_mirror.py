"""The resident-container mirror preserves feature and training receipt semantics."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shlex
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "dataset_mirror", Path(__file__).resolve().parents[1] / "byconstruction/dataset_mirror.py"
)
mirror = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mirror)


class RetainedPipe(io.BytesIO):
    def close(self):
        self.flushed_bytes = self.getvalue()
        super().close()


class MirrorKinds(unittest.TestCase):
    def test_mounted_transport_and_corruption_refusal(self):
        for corrupt in (False, True):
            with self.subTest(corrupt=corrupt), tempfile.TemporaryDirectory() as directory:
                parent = Path(directory)
                root, destination = parent / "source", parent / "mirror"
                root.mkdir()
                for i in range(3):
                    (root / f"artifact-{i}.txt").write_text(f"retained {i}")
                argv = ["dataset_mirror.py", "--root", str(root), "--nfs",
                        "--destination", str(destination)]
                original_copy = mirror.shutil.copyfile

                def copy(source, target):
                    original_copy(source, target)
                    if corrupt:
                        Path(target).write_bytes(b"corrupt")

                with patch.object(sys, "argv", argv), \
                        patch.object(mirror.subprocess, "run", side_effect=AssertionError("remote command")), \
                        patch.object(mirror.subprocess, "Popen", side_effect=AssertionError("remote process")), \
                        patch.object(mirror.subprocess, "check_output", side_effect=AssertionError("remote query")), \
                        patch.object(mirror.shutil, "copyfile", copy), contextlib.redirect_stdout(io.StringIO()):
                    if corrupt:
                        with self.assertRaises(AssertionError):
                            mirror.main()
                        self.assertFalse((root / "MIRROR_VERIFIED.json").exists())
                        continue
                    mirror.main()
                receipt = json.loads((root / "MIRROR_VERIFIED.json").read_text())
                self.assertEqual(receipt["transport"], "mounted-filesystem")
                self.assertFalse(receipt["remote_processes_started"])
                self.assertNotIn("checksum_container_image_id", receipt)
                self.assertEqual(len(receipt["random_readbacks"]), 3)
                for line in (destination / "SHA256SUMS").read_text().splitlines():
                    digest, name = line.split("  ", 1)
                    self.assertEqual(mirror.sha(destination / name), digest)
                    self.assertEqual((root / name).read_bytes(), (destination / name).read_bytes())

    def check_kind(self, kind):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payloads = {f"artifact-{i}.txt": f"retained artifact {i}\n".encode() for i in range(3)}
            for name, data in payloads.items():
                (root / name).write_bytes(data)
            image = "sha256:" + "a" * 64
            commands, transfers = [], []

            def output(command, **kwargs):
                remote = shlex.split(command[2])
                if remote[:3] == ["docker", "image", "inspect"]:
                    return image + "\n"
                self.assertIn("--entrypoint=/usr/bin/sha256sum", remote)
                commands.append(remote)
                name = remote[-1]
                return hashlib.sha256(payloads[name]).hexdigest() + "  " + name + "\n"

            def run(command, **kwargs):
                remote = shlex.split(command[2])
                if remote[:2] == ["docker", "run"]:
                    commands.append(remote)

            def popen(command, **kwargs):
                remote = shlex.split(command[2])
                commands.append(remote)
                child = unittest.mock.MagicMock()
                child.stdin = RetainedPipe()
                child.__enter__.return_value = child
                child.wait.return_value = 0
                transfers.append(child.stdin)
                return child

            argv = ["dataset_mirror.py", "--root", str(root), "--host", "fixture",
                    "--destination", "/array/artifacts", "--capacity-path", "/array",
                    "--artifact-kind", kind, "--image", "resident-tool:fixture"]
            with patch.object(sys, "argv", argv), patch.object(mirror.subprocess, "run", run), \
                    patch.object(mirror.subprocess, "check_output", output), \
                    patch.object(mirror.subprocess, "Popen", popen), contextlib.redirect_stdout(io.StringIO()):
                mirror.main()
            receipt = json.loads((root / "MIRROR_VERIFIED.json").read_text())
            self.assertEqual(receipt["schema"], f"dataset-{kind}-mirror-v1")
            self.assertEqual(receipt["checksum_container_image_id"], image)
            self.assertEqual(receipt["artifact_files_verified_before_receipt"], 3)
            self.assertEqual(len(receipt["random_readbacks"]), 3)
            if kind == "features":
                self.assertIs(receipt["response_tables_opened"], False)
                self.assertNotIn("label_values_interpreted_by_mirror", receipt)
            else:
                self.assertIs(receipt["label_values_interpreted_by_mirror"], False)
                self.assertNotIn("response_tables_opened", receipt)
                self.assertNotIn("readme_example_exposure_receipt", receipt)
            self.assertEqual(len(transfers), 2)
            with tarfile.open(fileobj=io.BytesIO(transfers[0].flushed_bytes)) as archive:
                self.assertEqual(set(archive.getnames()), {*payloads, "SHA256SUMS"})
                for name, data in payloads.items():
                    self.assertEqual(archive.extractfile(name).read(), data)
            with tarfile.open(fileobj=io.BytesIO(transfers[1].flushed_bytes)) as archive:
                self.assertEqual(set(archive.getnames()), {"MIRROR_VERIFIED.json", "SHA256SUMS"})
                self.assertEqual(json.load(archive.extractfile("MIRROR_VERIFIED.json")), receipt)
            self.assertTrue(commands)
            for command in commands:
                for option in ("--pull=never", "--network=none", "--read-only", "--cpuset-cpus=0-23",
                               "--cpu-shares=256", "--memory=1g"):
                    self.assertIn(option, command)
                self.assertIn(image, command)
                self.assertNotIn("rsync", command)

    def test_features(self):
        self.check_kind("features")

    def test_training(self):
        self.check_kind("training")


if __name__ == "__main__":
    unittest.main()
