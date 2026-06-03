#!/usr/bin/env python3
"""
Git Server Client for connecting to external Gitea instance.

This module provides a lightweight client for interacting with a shared
Gitea service, optimized for task-based isolation where multiple environment
instances share the same Gitea server but have isolated workspaces.
"""

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse


@dataclass
class RepoInfo:
    """Information about a repository."""

    name: str
    url: str
    commit: str
    clone_url: str


class GitServerClient:
    """
    Client for connecting to an external Gitea server.

    This client is optimized for task-based isolation where:
    - Multiple tasks share the same Gitea instance
    - Each task has its own isolated workspace
    - Fast reset() via git operations (no server restart)
    - Repos are pre-migrated to Gitea once

    Args:
        gitea_url: URL of the Gitea server (e.g., "http://gitea:3000")
        username: Gitea username for authentication
        password: Gitea password for authentication
        workspace_dir: Local workspace directory for cloning repos

    Examples:

        ```python
        import os

        # Connect to shared Gitea (credentials from environment)
        client = GitServerClient(
            gitea_url=os.getenv("GITEA_URL"),
            username=os.getenv("GITEA_USERNAME"),
            password=os.getenv("GITEA_PASSWORD")
        )
        client.wait_for_ready()
        # Clone repo to workspace
        path = client.clone_to_workspace("my-repo", commit="abc123")
        # Fast reset to base state
        client.reset_workspace("my-repo", commit="abc123")
        ```
    """

    def __init__(
        self,
        gitea_url: str,
        username: str,
        password: str,
        workspace_dir: str = "/workspace",
    ):
        """Initialize Git Server Client."""
        self.gitea_url = gitea_url.rstrip("/")
        self.username = username
        self.password = password
        self.workspace_dir = Path(workspace_dir)
        self.is_ready = False

        # Parse Gitea URL
        parsed = urlparse(self.gitea_url)
        self.domain = parsed.hostname or "localhost"
        self.port = parsed.port or 3000

        # Ensure workspace exists
        os.makedirs(self.workspace_dir, exist_ok=True)

        # Configure git credentials
        self._configure_git()

    def _configure_git(self):
        """Configure git credentials for automatic authentication."""
        home_dir = Path.home()

        # Git config
        git_config = f"""[user]
    name = {self.username}
    email = {self.username}@local.env
[init]
    defaultBranch = main
[credential]
    helper = store
"""
        gitconfig_path = home_dir / ".gitconfig"
        gitconfig_path.write_text(git_config)

        # Git credentials
        git_credentials = (
            f"http://{self.username}:{self.password}@{self.domain}:{self.port}\n"
        )
        gitcreds_path = home_dir / ".git-credentials"
        gitcreds_path.write_text(git_credentials)
        gitcreds_path.chmod(0o600)

    def wait_for_ready(self, timeout: int = 30) -> bool:
        """
        Wait for Gitea server to be ready.

        Args:
            timeout: Maximum seconds to wait

        Returns:
            True if server is ready, False otherwise
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                result = subprocess.run(
                    ["curl", "-sf", f"{self.gitea_url}/"],
                    capture_output=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    self.is_ready = True
                    return True
            except subprocess.TimeoutExpired:
                pass
            except Exception:
                pass

            time.sleep(1)

        return False

    def list_repositories(self) -> list[dict[str, str]]:
        """
        List all repositories in Gitea.

        Returns:
            List of repository information dictionaries
        """
        if not self.is_ready:
            raise RuntimeError("Gitea server is not ready")

        result = subprocess.run(
            [
                "curl",
                "-s",
                f"{self.gitea_url}/api/v1/user/repos",
                "-u",
                f"{self.username}:{self.password}",
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            return []

        try:
            repos = json.loads(result.stdout)
            return [
                {
                    "name": repo["name"],
                    "full_name": repo["full_name"],
                    "clone_url": repo["clone_url"],
                    "description": repo.get("description", ""),
                }
                for repo in repos
            ]
        except (json.JSONDecodeError, KeyError):
            return []

    def clone_to_workspace(
        self, repo_name: str, target_dir: str | None = None, commit: str = "main"
    ) -> str:
        """
        Clone a repository to the workspace at a specific commit.

        This creates a fresh clone optimized for task isolation.

        Args:
            repo_name: Name of repository to clone
            target_dir: Target directory name (defaults to repo_name)
            commit: Commit hash or branch to check out

        Returns:
            Path to cloned repository

        Raises:
            RuntimeError: If clone fails
        """
        if not self.is_ready:
            raise RuntimeError("Gitea server is not ready")

        target_dir = target_dir or repo_name
        target_path = self.workspace_dir / target_dir

        # Remove existing directory if present
        if target_path.exists():
            shutil.rmtree(target_path)

        clone_url = f"{self.gitea_url}/{self.username}/{repo_name}.git"

        # Clone repository
        result = subprocess.run(
            ["git", "clone", clone_url, str(target_path)],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(f"Clone failed: {result.stderr}")

        # Checkout specific commit
        if commit != "main":
            result = subprocess.run(
                ["git", "checkout", commit],
                cwd=str(target_path),
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                raise RuntimeError(f"Checkout failed: {result.stderr}")

        return str(target_path)

    def reset_workspace(self, repo_name: str, commit: str = "main") -> bool:
        """
        Fast reset of workspace to base state (optimized for task resets).

        This is much faster than re-cloning. It:
        1. Checks out the target commit
        2. Resets to that commit (hard)
        3. Cleans untracked files

        Args:
            repo_name: Name of repository (directory in workspace)
            commit: Commit hash or branch to reset to

        Returns:
            True if reset successful

        Raises:
            RuntimeError: If reset fails
        """
        repo_path = self.workspace_dir / repo_name

        if not repo_path.exists():
            raise RuntimeError(f"Repository not found in workspace: {repo_name}")

        # Fetch latest (in case commit is new)
        subprocess.run(
            ["git", "fetch", "--all"],
            cwd=str(repo_path),
            capture_output=True,
        )

        # Checkout and hard reset to commit
        result = subprocess.run(
            ["git", "checkout", commit],
            cwd=str(repo_path),
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(f"Checkout failed: {result.stderr}")

        result = subprocess.run(
            [
                "git",
                "reset",
                "--hard",
                f"origin/{commit}" if commit != "main" else commit,
            ],
            cwd=str(repo_path),
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            # Try without origin/ prefix
            result = subprocess.run(
                ["git", "reset", "--hard", commit],
                cwd=str(repo_path),
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Reset failed: {result.stderr}")

        # Clean untracked files and directories
        subprocess.run(
            ["git", "clean", "-fdx"],
            cwd=str(repo_path),
            capture_output=True,
        )

        return True

    def execute_git_command(
        self, command: str, working_dir: str = ""
    ) -> tuple[int, str, str]:
        """
        Execute a git command in the workspace.

        Args:
            command: Git command to execute (without 'git' prefix)
            working_dir: Working directory relative to workspace

        Returns:
            Tuple of (exit_code, stdout, stderr)
        """
        work_path = (
            self.workspace_dir / working_dir if working_dir else self.workspace_dir
        )

        if not work_path.exists():
            return (1, "", f"Working directory does not exist: {work_path}")

        # Split command safely
        cmd_parts = ["git"] + command.split()

        result = subprocess.run(
            cmd_parts,
            cwd=str(work_path),
            capture_output=True,
            text=True,
        )

        return (result.returncode, result.stdout, result.stderr)

    def get_current_commit(self, repo_name: str) -> str:
        """
        Get current commit hash of a workspace repository.

        Args:
            repo_name: Name of repository in workspace

        Returns:
            Commit hash
        """
        repo_path = self.workspace_dir / repo_name

        if not repo_path.exists():
            raise RuntimeError(f"Repository not found: {repo_name}")

        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_path),
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(f"Failed to get commit: {result.stderr}")

        return result.stdout.strip()

    def workspace_exists(self, repo_name: str) -> bool:
        """Check if a repository exists in workspace."""
        return (self.workspace_dir / repo_name).exists()
