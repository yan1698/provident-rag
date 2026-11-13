"""Exercise secret checks against isolated Git indexes, with synthetic credentials."""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from check_repository import check
from prepare_public_release import prepare


class RepositoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.git("init", "-q")
        self.secret = "sk-" + "A1b2C3d4" * 5

    def git(self, *args):
        return subprocess.check_output(["git", "-C", str(self.root), *args], stderr=subprocess.DEVNULL)

    def write(self, name, text):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def test_safe_sources_and_template(self):
        self.write("app.py", "print('safe')\n")
        self.write(".env.example", "DASHSCOPE_API_KEY=\nJWT_SECRET=dev-only-change-me\n")
        count, issues = check(self.root)
        self.assertEqual(count, 2)
        self.assertEqual(issues, [])

    def test_ignored_local_secrets_are_not_upload_candidates(self):
        self.write(".gitignore", ".env\ndocs/\n")
        self.write(".env", "DASHSCOPE_API_KEY=" + self.secret)
        self.write("docs/trace.json", self.secret)
        self.assertEqual(check(self.root)[1], [])

    def test_forced_tracked_private_file_is_blocked(self):
        self.write(".gitignore", ".env\n")
        self.write(".env", "DASHSCOPE_API_KEY=" + self.secret)
        self.git("add", "-f", ".env")
        for staged in (False, True):
            issues = check(self.root, staged)[1]
            self.assertTrue(any("local/private" in issue for issue in issues))
            self.assertNotIn(self.secret, "\n".join(issues))

    def test_provider_key_in_source_is_blocked_without_printing_it(self):
        self.write("app.py", "token = '" + self.secret + "'")
        issues = check(self.root)[1]
        self.assertTrue(any("provider API key" in issue for issue in issues))
        self.assertNotIn(self.secret, "\n".join(issues))

    def test_staged_scan_reads_index_not_working_tree(self):
        self.write("app.py", self.secret)
        self.git("add", "app.py")
        self.write("app.py", "safe")
        self.assertEqual(check(self.root)[1], [])
        self.assertTrue(check(self.root, staged=True)[1])

    def test_unstaged_secret_is_still_reported(self):
        self.write("app.py", "safe")
        self.git("add", "app.py")
        self.write("app.py", self.secret)
        self.assertEqual(check(self.root, staged=True)[1], [])
        self.assertTrue(check(self.root)[1])

    def test_local_secret_without_provider_prefix_is_blocked(self):
        value = "synthetic" + "LocalCredential987654"
        self.write(".gitignore", ".env\n")
        self.write(".env", "JWT_SECRET=" + value)
        self.write("config.json", value)
        issues = check(self.root)[1]
        self.assertTrue(any("local .env credential JWT_SECRET" in issue for issue in issues))
        self.assertNotIn(value, "\n".join(issues))

    def test_artifacts_are_blocked(self):
        for name in (
            "backup.dump",
            "benchmarks/reports/result.json",
            "data/runtime/state.json",
            "tmp-ui-check/run.json",
        ):
            with self.subTest(name=name):
                self.write(name, "output")
                self.assertTrue(any(name in issue for issue in check(self.root)[1]))

    def test_large_file_is_blocked(self):
        self.write("result.txt", "x" * (5 * 1024 * 1024 + 1))
        self.assertTrue(any("5 MiB" in issue for issue in check(self.root)[1]))

    def test_template_credentials_cannot_be_filled(self):
        self.write(".env.example", "DASHSCOPE_API_KEY=example-value")
        self.assertTrue(any("must be empty" in issue for issue in check(self.root)[1]))

    def test_release_copies_sources_without_history_or_local_state(self):
        self.write(".gitignore", ".env\ntmp/\npublic-release/\n")
        self.write(".env", "DASHSCOPE_API_KEY=" + self.secret)
        self.write("README.md", "A public project")
        self.write("LICENSE", "MIT")
        self.write(".env.example", "DASHSCOPE_API_KEY=\n")
        self.write("backend/app/main.py", "print('public')")
        self.write("tmp/response.json", self.secret)
        self.write(".cursor/session.json", "private editor state")
        self.git("add", "backend/app/main.py")
        self.write("backend/app/main.py", "print('latest uncommitted source')")
        destination = self.root / "public-release/project"
        manifest = prepare(self.root, destination)
        self.assertEqual((destination / "backend/app/main.py").read_text(), "print('latest uncommitted source')")
        self.assertFalse((destination / ".git").exists())
        self.assertFalse((destination / ".env").exists())
        self.assertFalse((destination / "tmp").exists())
        self.assertFalse((destination / ".cursor").exists())
        self.assertEqual(manifest["file_count"], 5)
        self.assertFalse(manifest["contains_git_history"])

    def test_release_preserves_existing_destination(self):
        destination = self.root / "public-release/project"
        destination.mkdir(parents=True)
        (destination / "keep.txt").write_text("keep")
        with self.assertRaises(FileExistsError):
            prepare(self.root, destination)
        self.assertEqual((destination / "keep.txt").read_text(), "keep")

    def test_release_refuses_destination_outside_release_directory(self):
        with self.assertRaises(ValueError):
            prepare(self.root, self.root / "backend")

    def test_public_release_excludes_local_materials_but_keeps_test_logic(self):
        self.write(".gitignore", "public-release/\ndata/test-materials/\n")
        for name in ("README.md", "LICENSE", ".env.example"):
            self.write(name, "")
        self.write("scripts/test_example.py", "assert True\n")
        for name in ("data/eval/local.json", "data/raw/sample.md", "benchmarks/fixtures/local.json"):
            self.write(name, "local-only")
        self.write("data/test-materials/test_example.json", "{}")
        destination = self.root / "public-release/project"
        manifest = prepare(self.root, destination)
        self.assertTrue((destination / "scripts/test_example.py").exists())
        self.assertFalse((destination / "data").exists())
        self.assertFalse((destination / "benchmarks").exists())
        self.assertFalse(manifest["test_materials_included"])

    def test_public_check_rejects_materials_even_when_tracked(self):
        self.write("data/eval/local.json", "{}")
        self.git("add", "data/eval/local.json")
        self.assertEqual(check(self.root)[1], [])
        for staged in (False, True):
            self.assertTrue(
                any("concrete test materials" in message for message in check(self.root, staged, public=True)[1])
            )


if __name__ == "__main__":
    unittest.main()
