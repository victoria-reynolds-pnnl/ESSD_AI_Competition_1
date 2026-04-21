"""
r_config.py — R environment configuration for rpy2

HOW TO FIND YOUR R_HOME
-----------------------
1. Open RStudio
2. In the R console, run:  R.home()
3. Copy the path that is printed (e.g. "C:/Program Files/R/R-4.3.1")
4. Paste it as the value of R_HOME below (use forward slashes or double backslashes)
5. Save this file

This file is imported by train.py, evaluate.py, and interpretability.py
BEFORE rpy2 is loaded, so the env var is set in time.

ALTERNATIVE (no file edit needed)
----------------------------------
Set the environment variable in PowerShell before running Python:
    $env:R_HOME = "C:/Program Files/R/R-4.3.1"
    python Scripts/train.py

If R_HOME is already set in your OS environment, you do not need to change
anything in this file — it will be detected automatically.
"""

import os
import pathlib
import sys

# ── Paste your R home path here (or leave as None to use env var / auto-detect) ──
R_HOME_OVERRIDE: str | None = "C:/Program Files/R/R-4.5.3"

# ── Common Windows install patterns to try if the above is None ──────────────
_COMMON_PATHS = [
    r"C:\Program Files\R",
    r"C:\Program Files (x86)\R",
]


def _find_r_home() -> str:
    """Return R_HOME string, or raise RuntimeError with actionable instructions."""

    # 1. Explicit override in this file
    if R_HOME_OVERRIDE:
        return R_HOME_OVERRIDE

    # 2. Already set in the process environment
    env_val = os.environ.get("R_HOME", "").strip()
    if env_val and pathlib.Path(env_val).is_dir():
        return env_val

    # 3. Scan common Windows install directories for the highest R version
    candidates = []
    for base in _COMMON_PATHS:
        base_p = pathlib.Path(base)
        if base_p.is_dir():
            for child in sorted(base_p.iterdir(), reverse=True):
                if child.is_dir() and child.name.startswith("R-"):
                    r_exe = child / "bin" / "R.exe"
                    if r_exe.exists():
                        candidates.append(str(child))
    if candidates:
        return candidates[0]

    # 4. Check if Rscript is on PATH
    import shutil
    rscript = shutil.which("Rscript")
    if rscript:
        # Rscript lives in R_HOME/bin/Rscript.exe  →  parent is bin  →  grandparent is R_HOME
        return str(pathlib.Path(rscript).resolve().parents[1])

    # 5. Try registry (may not exist on corporate deployments)
    try:
        import winreg
        for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
            for subkey in (r"SOFTWARE\R-core\R64", r"SOFTWARE\R-core\R"):
                try:
                    with winreg.OpenKey(hive, subkey) as k:
                        val, _ = winreg.QueryValueEx(k, "InstallPath")
                        if val and pathlib.Path(val).is_dir():
                            return val
                except OSError:
                    pass
    except ImportError:
        pass  # not on Windows

    # 6. Give up with a helpful message
    raise RuntimeError(
        "\n"
        "╔══════════════════════════════════════════════════════════════════╗\n"
        "║  R_HOME not found — please do ONE of the following:             ║\n"
        "╠══════════════════════════════════════════════════════════════════╣\n"
        "║  OPTION A (one-time file edit):                                  ║\n"
        "║    1. Open RStudio and run:  R.home()                           ║\n"
        "║    2. Copy the printed path                                      ║\n"
        '║    3. Edit Scripts/r_config.py and set R_HOME_OVERRIDE to that  ║\n'
        "║       path.  Example:                                            ║\n"
        '║         R_HOME_OVERRIDE = "C:/Program Files/R/R-4.3.1"         ║\n'
        "║                                                                  ║\n"
        "║  OPTION B (per-session, no file edit):                          ║\n"
        "║    In PowerShell, before running Python:                         ║\n"
        '║      $env:R_HOME = "C:/Program Files/R/R-4.3.1"                ║\n'
        "║      python Scripts/train.py                                     ║\n"
        "╚══════════════════════════════════════════════════════════════════╝\n"
    )


def get_rscript_path() -> str:
    """
    Return the full path to Rscript.exe.
    Adds R/bin to os.environ['PATH'] so subsequent subprocess calls also work.
    """
    import shutil

    # 1. Already on PATH
    rscript = shutil.which("Rscript") or shutil.which("Rscript.exe")
    if rscript:
        return rscript

    # 2. Derive from R_HOME (calls configure() which finds R_HOME)
    r_home = _find_r_home()
    r_bin   = pathlib.Path(r_home) / "bin"
    rscript = r_bin / "Rscript.exe"
    if not rscript.exists():
        raise FileNotFoundError(
            f"Rscript.exe not found at {rscript}. "
            "Ensure R is correctly installed."
        )
    # Add to PATH so shutil.which('Rscript') works for the rest of the process
    bin_str = str(r_bin)
    if bin_str not in os.environ.get("PATH", ""):
        os.environ["PATH"] = bin_str + os.pathsep + os.environ.get("PATH", "")
    return str(rscript)


def configure() -> str:
    """Set os.environ['R_HOME'] and add R/bin to PATH.  Call before importing rpy2."""
    r_home = _find_r_home()

    os.environ["R_HOME"] = r_home

    # Also add R/bin to PATH so the R DLL is found by rpy2 on Windows
    r_bin = str(pathlib.Path(r_home) / "bin" / "x64")
    r_bin_base = str(pathlib.Path(r_home) / "bin")
    for p in (r_bin, r_bin_base):
        if p not in os.environ.get("PATH", ""):
            os.environ["PATH"] = p + os.pathsep + os.environ.get("PATH", "")

    print(f"R_HOME set to: {r_home}")
    return r_home
