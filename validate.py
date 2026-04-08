#!/usr/bin/env python3
"""
validate.py — OpenEnv Submission Validator (Cross-platform)

Checks that your project passes pre-submission requirements:
  1. HF Space is live and responds to /reset
  2. Docker image builds successfully
  3. openenv validate passes

Usage:
  python validate.py <ping_url> [repo_dir]
  
  Examples:
    python validate.py https://my-team.hf.space
    python validate.py https://my-team.hf.space .
"""

import sys
import os
import subprocess
import time
import json
from pathlib import Path
from typing import Tuple
import socket
from urllib.parse import urlparse

try:
    import requests
except ImportError:
    print("Error: requests library required. Install it with: pip install requests")
    sys.exit(1)


# Color codes
class Color:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BOLD = '\033[1m'
    NC = '\033[0m'
    
    @classmethod
    def disable(cls):
        """Disable colors if not a terminal."""
        cls.RED = cls.GREEN = cls.YELLOW = cls.BOLD = cls.NC = ''


def log(msg: str, level: str = "INFO"):
    """Log with timestamp."""
    timestamp = time.strftime("%H:%M:%S")
    if level == "PASS":
        print(f"[{timestamp}] {Color.GREEN}PASSED{Color.NC} -- {msg}")
    elif level == "FAIL":
        print(f"[{timestamp}] {Color.RED}FAILED{Color.NC} -- {msg}")
    elif level == "BOLD":
        print(f"[{timestamp}] {Color.BOLD}{msg}{Color.NC}")
    else:
        print(f"[{timestamp}] {msg}")


def hint(msg: str):
    """Print a hint."""
    print(f"  {Color.YELLOW}Hint:{Color.NC} {msg}")


def stop_at(step: str):
    """Stop validation and exit."""
    print(f"\n{Color.RED}{Color.BOLD}Validation stopped at {step}.{Color.NC} Fix the above before continuing.\n")
    sys.exit(1)


def check_hf_space(ping_url: str) -> bool:
    """Check if HF Space is live."""
    log(f"Step 1/3: Pinging HF Space ({ping_url}/reset) ...", "BOLD")
    
    timeout_secs = 30
    try:
        response = requests.post(
            f"{ping_url}/reset",
            json={},
            timeout=timeout_secs,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            log("HF Space is live and responds to /reset", "PASS")
            return True
        else:
            log(f"HF Space /reset returned HTTP {response.status_code} (expected 200)", "FAIL")
            hint("Make sure your Space is running and the URL is correct.")
            hint(f"Try opening {ping_url} in your browser first.")
            return False
    except requests.exceptions.Timeout:
        log(f"HF Space not reachable (timeout after {timeout_secs}s)", "FAIL")
        hint("Check your network connection and that the Space is running.")
        hint(f"Try opening {ping_url} in your browser.")
        return False
    except requests.exceptions.ConnectionError as e:
        log("HF Space not reachable (connection failed)", "FAIL")
        hint(f"Error: {e}")
        hint(f"Check that {ping_url} is correct and the Space is deployed.")
        return False
    except Exception as e:
        log(f"Error checking HF Space: {e}", "FAIL")
        return False


def check_docker_build(repo_dir: str) -> bool:
    """Check if Docker image builds."""
    log("Step 2/3: Running docker build ...", "BOLD")
    
    if not _which("docker"):
        log("docker command not found", "FAIL")
        hint("Install Docker: https://docs.docker.com/get-docker/")
        return False
    
    dockerfile_path = None
    if Path(repo_dir, "Dockerfile").exists():
        dockerfile_path = Path(repo_dir) / "Dockerfile"
    elif Path(repo_dir, "server", "Dockerfile").exists():
        dockerfile_path = Path(repo_dir, "server") / "Dockerfile"
    
    if not dockerfile_path:
        log("No Dockerfile found in repo root or server/ directory", "FAIL")
        return False
    
    docker_context = dockerfile_path.parent
    log(f"  Found Dockerfile in {docker_context}")
    
    timeout_secs = 600
    try:
        start = time.time()
        result = subprocess.run(
            ["docker", "build", str(docker_context)],
            capture_output=True,
            text=True,
            timeout=timeout_secs
        )
        elapsed = time.time() - start
        
        if result.returncode == 0:
            log(f"Docker build succeeded ({elapsed:.1f}s)", "PASS")
            return True
        else:
            log(f"Docker build failed", "FAIL")
            print("\n--- Last 20 lines of build output ---")
            lines = result.stderr.split("\n") if result.stderr else result.stdout.split("\n")
            for line in lines[-20:]:
                if line.strip():
                    print(f"  {line}")
            return False
    except subprocess.TimeoutExpired:
        log(f"Docker build timed out (>{timeout_secs}s)", "FAIL")
        hint("Your Dockerfile may be too complex or downloading dependencies is slow.")
        return False
    except Exception as e:
        log(f"Error running docker build: {e}", "FAIL")
        return False


def check_openenv_validate(repo_dir: str) -> bool:
    """Check if openenv validate passes."""
    log("Step 3/3: Running openenv validate ...", "BOLD")
    
    if not _which("openenv"):
        log("openenv command not found", "FAIL")
        hint("Install it: pip install openenv-core")
        return False
    
    try:
        result = subprocess.run(
            ["openenv", "validate"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            log("openenv validate passed", "PASS")
            if result.stdout:
                log(f"  {result.stdout.strip()}")
            return True
        else:
            log("openenv validate failed", "FAIL")
            print("\n--- openenv validate output ---")
            print(result.stderr if result.stderr else result.stdout)
            return False
    except subprocess.TimeoutExpired:
        log("openenv validate timed out", "FAIL")
        return False
    except Exception as e:
        log(f"Error running openenv validate: {e}", "FAIL")
        return False


def _which(cmd: str) -> str:
    """Find command path (cross-platform)."""
    return subprocess.run(
        ["where" if sys.platform == "win32" else "which", cmd],
        capture_output=True,
        text=True
    ).stdout.strip()


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <ping_url> [repo_dir]")
        print()
        print("  ping_url   Your HuggingFace Space URL (e.g. https://your-space.hf.space)")
        print("  repo_dir   Path to your repo (default: current directory)")
        print()
        print("Examples:")
        print(f"  python {sys.argv[0]} https://my-team.hf.space")
        print(f"  python {sys.argv[0]} https://my-team.hf.space .")
        sys.exit(1)
    
    # Check if terminal supports color
    if not sys.stdout.isatty():
        Color.disable()
    
    ping_url = sys.argv[1].rstrip("/")
    repo_dir = sys.argv[2] if len(sys.argv) > 2 else "."
    
    # Resolve repo_dir
    try:
        repo_dir = str(Path(repo_dir).resolve())
        if not Path(repo_dir).is_dir():
            print(f"Error: directory '{repo_dir}' not found")
            sys.exit(1)
    except Exception as e:
        print(f"Error resolving directory: {e}")
        sys.exit(1)
    
    print()
    print(f"{Color.BOLD}========================================{Color.NC}")
    print(f"{Color.BOLD}  OpenEnv Submission Validator{Color.NC}")
    print(f"{Color.BOLD}========================================{Color.NC}")
    log(f"Repo:     {repo_dir}")
    log(f"Ping URL: {ping_url}")
    print()
    
    # Run checks
    checks = [
        ("HF Space", check_hf_space(ping_url)),
        ("Docker build", check_docker_build(repo_dir)),
        ("openenv validate", check_openenv_validate(repo_dir)),
    ]
    
    passed = sum(1 for _, result in checks if result)
    total = len(checks)
    
    # Report
    print()
    print(f"{Color.BOLD}========================================{Color.NC}")
    if passed == total:
        print(f"{Color.GREEN}{Color.BOLD}  All {total}/{total} checks passed!{Color.NC}")
        print(f"{Color.GREEN}{Color.BOLD}  Your submission is ready to submit.{Color.NC}")
        print(f"{Color.BOLD}========================================{Color.NC}")
        print()
        return 0
    else:
        print(f"{Color.RED}{Color.BOLD}  {passed}/{total} checks passed.{Color.NC}")
        print(f"{Color.RED}{Color.BOLD}  Fix failures above and try again.{Color.NC}")
        print(f"{Color.BOLD}========================================{Color.NC}")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
