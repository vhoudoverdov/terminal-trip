"""Client-side version checker and dual-mode self-updater for terminal-trip.

Supports:
1. Remote version discovery via GitHub Releases API -> GitHub Tags API -> raw pyproject.toml fallback.
2. Non-blocking 24h-cached startup update checks for the HUD toast notification.
3. Dual-mode self-updating (`--update`):
   - Git working tree mode (`git fetch --tags` + `git pull --ff-only`, with dirty-tree protection)
   - Installed package mode (`python -m pip install --upgrade git+https://...@vX.Y.Z`)
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

__version__ = "0.2.1"

REPO_OWNER = "vhoudoverdov"
REPO_NAME = "terminal-trip"
RELEASES_API_URL = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/releases/latest"
TAGS_API_URL = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/tags"
RAW_PYPROJECT_URL = f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/main/pyproject.toml"
GIT_REPO_URL = f"https://github.com/{REPO_OWNER}/{REPO_NAME}.git"

_SEMVER_RE = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)$")
_PYPROJECT_VER_RE = re.compile(r'^\s*version\s*=\s*"(\d+\.\d+\.\d+)"', re.MULTILINE)


def parse_version(version_str: str) -> tuple[int, int, int]:
    """Validate and parse a semantic version string ('0.2.1' or 'v0.2.1') into an int 3-tuple."""
    m = _SEMVER_RE.match(str(version_str).strip())
    if not m:
        raise ValueError(f"invalid semantic version: {version_str!r}")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def format_version(version_str: str) -> str:
    """Return canonical 'X.Y.Z' string after validating semantic version."""
    major, minor, patch = parse_version(version_str)
    return f"{major}.{minor}.{patch}"


def _default_fetch(url: str, timeout: float) -> str:
    if not url.startswith(("https://api.github.com/", "https://raw.githubusercontent.com/")):
        raise ValueError(f"disallowed update check URL: {url}")
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": f"terminal-trip/{__version__}",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec B310 - strict https allowlist above
        return resp.read(256 * 1024).decode("utf-8", errors="replace")


def fetch_latest_version(timeout: float = 2.5, fetcher=None) -> tuple[str, str]:
    """Discover the latest available version from GitHub Releases, Tags, or raw pyproject.toml.

    Returns ``(canonical_version, source_name)``.
    """
    fetch = fetcher or _default_fetch
    errors = []

    # 1. Primary: GitHub Latest Release API
    try:
        body = fetch(RELEASES_API_URL, timeout)
        data = json.loads(body)
        if isinstance(data, dict) and "tag_name" in data:
            return format_version(data["tag_name"]), "github-release"
    except Exception as e:
        errors.append(f"releases: {e}")

    # 2. Fallback: GitHub Repository Tags API (highest semantic version tag)
    try:
        body = fetch(TAGS_API_URL, timeout)
        data = json.loads(body)
        if isinstance(data, list):
            valid_tags = []
            for item in data:
                if isinstance(item, dict) and "name" in item:
                    try:
                        v_str = format_version(item["name"])
                        valid_tags.append((parse_version(v_str), v_str))
                    except ValueError:
                        continue
            if valid_tags:
                valid_tags.sort(reverse=True)
                return valid_tags[0][1], "github-tag"
    except Exception as e:
        errors.append(f"tags: {e}")

    # 3. Fallback: Raw pyproject.toml on main
    try:
        body = fetch(RAW_PYPROJECT_URL, timeout)
        m = _PYPROJECT_VER_RE.search(body)
        if m:
            return format_version(m.group(1)), "pyproject-main"
    except Exception as e:
        errors.append(f"pyproject: {e}")

    raise RuntimeError("unable to query latest version (" + "; ".join(errors) + ")")


def check_for_update(current_version: str = __version__, timeout: float = 2.5, fetcher=None) -> dict:
    """Compare ``current_version`` against the latest version available on GitHub."""
    cur_norm = format_version(current_version)
    latest_norm, source = fetch_latest_version(timeout=timeout, fetcher=fetcher)
    is_newer = parse_version(latest_norm) > parse_version(cur_norm)
    return {
        "current": cur_norm,
        "latest": latest_norm,
        "update_available": is_newer,
        "source": source,
    }


def get_default_cache_path() -> Path:
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "terminal-trip" / "update_check.json"


def check_for_update_cached(
    current_version: str = __version__,
    ttl_seconds: float = 86400.0,
    timeout: float = 2.0,
    cache_path: Path | None = None,
    fetcher=None,
    now_fn=time.time,
) -> dict | None:
    """Cached non-throwing update check for background startup notifications."""
    cpath = cache_path or get_default_cache_path()
    cur_norm = format_version(current_version)
    now = now_fn()

    try:
        if cpath.is_file():
            cached = json.loads(cpath.read_text(encoding="utf-8"))
            if (
                isinstance(cached, dict)
                and cached.get("current") == cur_norm
                and isinstance(cached.get("checked_at"), (int, float))
                and (now - cached["checked_at"]) < ttl_seconds
                and "latest" in cached
            ):
                latest_norm = format_version(cached["latest"])
                return {
                    "current": cur_norm,
                    "latest": latest_norm,
                    "update_available": parse_version(latest_norm) > parse_version(cur_norm),
                    "source": cached.get("source", "cache"),
                    "cached": True,
                }
    except Exception:
        pass

    try:
        info = check_for_update(current_version=cur_norm, timeout=timeout, fetcher=fetcher)
    except Exception:
        return None

    try:
        cpath.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "current": info["current"],
            "latest": info["latest"],
            "source": info["source"],
            "checked_at": now,
        }
        cpath.write_text(json.dumps(payload), encoding="utf-8")
    except Exception:
        pass

    info["cached"] = False
    return info


def is_git_checkout(repo_dir: Path | None = None) -> bool:
    root = repo_dir or Path(__file__).resolve().parent
    return (root / ".git").exists()


def perform_update(
    current_version: str = __version__,
    repo_dir: Path | None = None,
    fetcher=None,
    runner=subprocess.run,
) -> tuple[bool, str]:
    """Check for a newer version and update either via git pull or pip install --upgrade."""
    root = repo_dir or Path(__file__).resolve().parent
    try:
        info = check_for_update(current_version=current_version, fetcher=fetcher)
    except Exception as e:
        return False, f"Update check failed: {e}"

    if not info["update_available"]:
        return True, f"terminal-trip is already up to date (v{info['current']})."

    latest = info["latest"]
    tag = f"v{latest}"

    if is_git_checkout(root):
        # Verify clean working tree before pulling
        st = runner(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=False,
        )
        if st.returncode != 0:
            return False, f"git status failed: {st.stderr.strip() or st.stdout.strip()}"
        if st.stdout.strip():
            return (
                False,
                f"Update v{latest} is available, but local git working tree has uncommitted changes. "
                "Please commit or stash your changes before running --update.",
            )

        fetch_res = runner(
            ["git", "-C", str(root), "fetch", "--tags", "origin"],
            capture_output=True,
            text=True,
            check=False,
        )
        if fetch_res.returncode != 0:
            return False, f"git fetch failed: {fetch_res.stderr.strip() or fetch_res.stdout.strip()}"

        pull_res = runner(
            ["git", "-C", str(root), "pull", "--ff-only"],
            capture_output=True,
            text=True,
            check=False,
        )
        if pull_res.returncode != 0:
            return False, f"git pull --ff-only failed: {pull_res.stderr.strip() or pull_res.stdout.strip()}"

        return True, f"Successfully updated terminal-trip v{info['current']} -> v{latest} via git."

    # Package installation mode (pip / pipx / venv)
    target_spec = f"git+{GIT_REPO_URL}@{tag}" if info["source"] in ("github-release", "github-tag") else f"git+{GIT_REPO_URL}"
    pip_res = runner(
        [sys.executable, "-m", "pip", "install", "--upgrade", target_spec],
        capture_output=True,
        text=True,
        check=False,
    )
    if pip_res.returncode != 0:
        return False, f"pip upgrade failed: {pip_res.stderr.strip() or pip_res.stdout.strip()}"

    return True, f"Successfully updated terminal-trip v{info['current']} -> v{latest} via pip."
