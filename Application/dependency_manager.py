"""
Dependency Manager - Checks and installs required packages

Install strategy (tried in order, stops at first success):
  1. Normal pip install              — standard Python from python.org
  2. --break-system-packages         — uv / PEP 668 managed Python
  3. --user                          — no write permission to system site-packages
  4. --user --break-system-packages  — uv + permission combo
  If all four fail, a clear human-readable error is returned.

Pip output is streamed live line-by-line so the user sees real
download progress instead of a frozen window.
"""

import sys
import subprocess
import importlib
import threading
from typing import List, Tuple, Dict, Optional, Callable


class DependencyManager:
    """Manages Python package dependencies"""

    # Required packages: import_name -> pip install spec
    REQUIRED_PACKAGES = {
        'customtkinter': 'customtkinter>=5.2.0',
        'PIL':           'pillow>=10.0.0',
        'cv2':           'opencv-python>=4.8.0',
        'numpy':         'numpy>=1.24.0',
        'pandas':        'pandas>=2.0.0',
        'ultralytics':   'ultralytics>=8.0.0',
        'supervision':   'supervision>=0.17.0',
    }

    # Install strategies tried in order (extra flags appended to base pip command)
    _STRATEGIES = [
        [],                                      # 1. plain pip install
        ["--break-system-packages"],             # 2. PEP 668 / uv managed env
        ["--user"],                              # 3. no system write permission
        ["--user", "--break-system-packages"],   # 4. uv + permission combo
    ]

    _STRATEGY_LABELS = [
        "standard",
        "managed environment (PEP 668)",
        "user install",
        "user install + managed environment",
    ]

    def __init__(self):
        self.missing_packages: List[str] = []
        self.installed_packages: List[str] = []
        self.status_callback: Optional[Callable[[str], None]] = None

    # ── Dependency checking ───────────────────────────────────────────────────

    def check_dependencies(self) -> Tuple[List[str], List[str]]:
        """
        Check which dependencies are installed.

        Returns:
            Tuple of (installed_package_names, missing_pip_specs)
        """
        installed, missing = [], []

        for import_name, pip_spec in self.REQUIRED_PACKAGES.items():
            try:
                importlib.import_module(import_name)
                installed.append(pip_spec.split('>=')[0])
            except ImportError:
                missing.append(pip_spec)

        self.installed_packages = installed
        self.missing_packages = missing
        return installed, missing

    def get_status_report(self) -> str:
        """Return a formatted status report string."""
        installed, missing = self.check_dependencies()

        lines = ["DEPENDENCY STATUS", "=" * 50, ""]

        if installed:
            lines.append("[OK] INSTALLED:")
            for pkg in installed:
                lines.append(f"  + {pkg}")
            lines.append("")

        if missing:
            lines.append("[!!] MISSING:")
            for pkg in missing:
                lines.append(f"  - {pkg}")
            lines.append("")

        if not missing:
            lines.append("All dependencies are installed!")
        else:
            lines.append(f"Need to install {len(missing)} package(s)")

        return "\n".join(lines) + "\n"

    # ── Live streaming install ────────────────────────────────────────────────

    def _run_pip_streaming(
        self,
        cmd: List[str],
        stream_callback: Optional[Callable[[str], None]] = None
    ) -> Tuple[int, str]:
        """
        Run a pip command and stream output line-by-line to stream_callback.

        Args:
            cmd:             Full command list e.g. ['python', '-m', 'pip', ...]
            stream_callback: Called with each output line as it arrives

        Returns:
            Tuple of (returncode, full_stderr)
        """
        stderr_lines = []

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                errors='replace',
                bufsize=1                    # line-buffered
            )

            # Stream stdout live
            for line in process.stdout:
                line = line.rstrip()
                if line:                     # skip blank lines from pip
                    if stream_callback:
                        stream_callback(f"    {line}")

            # Collect stderr (usually only filled on error)
            for line in process.stderr:
                line = line.rstrip()
                if line:
                    stderr_lines.append(line)
                    # Also stream stderr so student sees any warnings live
                    if stream_callback:
                        stream_callback(f"    {line}")

            process.wait(timeout=300)
            return process.returncode, "\n".join(stderr_lines)

        except subprocess.TimeoutExpired:
            process.kill()
            return -1, "Installation timed out after 5 minutes"
        except Exception as e:
            return -1, str(e)

    # ── Smart installation ────────────────────────────────────────────────────

    def install_package(self, package: str) -> Tuple[bool, str]:
        """
        Install a single package with live output streaming.

        Tries 4 strategies in order, stops at first success:
          1. pip install <pkg>
          2. pip install <pkg> --break-system-packages
          3. pip install <pkg> --user
          4. pip install <pkg> --user --break-system-packages

        Args:
            package: pip install spec, e.g. 'numpy>=1.24.0'

        Returns:
            Tuple of (success: bool, summary_message: str)
        """
        cb = self.status_callback
        if cb:
            cb(f">>> Installing {package} ...")
            cb("")

        base_cmd = [sys.executable, "-m", "pip", "install", package,
                    "--no-warn-script-location"]   # suppress noisy PATH warnings

        all_errors: List[str] = []

        for flags, label in zip(self._STRATEGIES, self._STRATEGY_LABELS):
            cmd = base_cmd + flags

            if cb:
                strategy_num = self._STRATEGIES.index(flags) + 1
                if flags:
                    cb(f"  [Strategy {strategy_num}] Trying: {label} ...")
                else:
                    cb(f"  [Strategy {strategy_num}] Trying: standard install ...")

            returncode, stderr = self._run_pip_streaming(cmd, stream_callback=cb)

            if returncode == 0:
                strategy_note = f" via {label}" if flags else ""
                if cb:
                    cb("")
                    cb(f"  [OK] {package} installed successfully{strategy_note}")
                    cb("")
                return True, f"[OK] {package} installed successfully{strategy_note}"

            # Strategy failed — is it worth trying the next one?
            all_errors.append(f"Strategy {self._STRATEGIES.index(flags)+1} ({label}): {stderr[:200]}")

            # If it's a network/download error, no point trying other strategies
            network_errors = ["connection", "timeout", "network", "resolve", "ssl"]
            if any(e in stderr.lower() for e in network_errors):
                if cb:
                    cb("")
                    cb("  [!!] Network error detected - check your internet connection.")
                    cb("")
                break

            if cb:
                cb(f"  [!!] Strategy {self._STRATEGIES.index(flags)+1} failed, trying next ...")
                cb("")

        # All strategies failed
        error_summary = "\n  ".join(all_errors)
        fail_msg = (
            f"[FAIL] Could not install {package}\n"
            f"  Tried all install methods. Errors:\n"
            f"  {error_summary}\n"
            f"  Manual fix: pip install {package}"
        )
        if cb:
            cb("")
            cb(f"  [FAIL] All strategies failed for {package}")
            cb(f"  Manual fix: pip install {package}")
            cb("")
        return False, fail_msg

    def install_all_missing(self, progress_callback=None) -> Dict[str, Tuple[bool, str]]:
        """
        Install all missing packages with live streaming output.

        Args:
            progress_callback: Optional callable(str) for progress updates

        Returns:
            Dict of {pip_spec: (success, message)}
        """
        self.status_callback = progress_callback
        results = {}

        _, missing = self.check_dependencies()

        if not missing:
            if progress_callback:
                progress_callback("[OK] All packages already installed — nothing to do!")
            return results

        total = len(missing)
        if progress_callback:
            progress_callback(f"Found {total} package(s) to install...")
            progress_callback("=" * 50)
            progress_callback("")

        for i, package in enumerate(missing, 1):
            if progress_callback:
                progress_callback(f"[{i}/{total}] {package}")
                progress_callback("-" * 40)

            success, message = self.install_package(package)
            results[package] = (success, message)

            if progress_callback:
                progress_callback("-" * 40)
                progress_callback("")

        # Final verification
        if progress_callback:
            progress_callback("=" * 50)
            progress_callback("Verifying installation...")

        _, still_missing = self.check_dependencies()

        if not still_missing:
            if progress_callback:
                progress_callback("[OK] All dependencies verified and ready!")
                progress_callback("You can now use all features of the application.")
        else:
            if progress_callback:
                progress_callback(f"[!!] {len(still_missing)} package(s) still missing:")
                for pkg in still_missing:
                    progress_callback(f"  - {pkg}")
                progress_callback("Please restart the application and try again.")

        return results

    def install_all_missing_async(self, progress_callback=None, completion_callback=None):
        """
        Install all missing packages in a background thread.

        Args:
            progress_callback:   Optional callable(str) for live updates
            completion_callback: Optional callable(dict) called when done
        """
        def worker():
            results = self.install_all_missing(progress_callback)
            if completion_callback:
                completion_callback(results)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        return thread


# ── CLI entry point ───────────────────────────────────────────────────────────

def check_dependencies_cli():
    """Command-line interface for checking and installing dependencies."""
    manager = DependencyManager()
    print(manager.get_status_report())

    installed, missing = manager.check_dependencies()

    if not missing:
        print("[OK] Ready to run the application!")
        return True

    response = input("\nInstall missing packages? (y/n): ").strip().lower()
    if response != 'y':
        print("Skipping installation.")
        return False

    print("\nInstalling packages...\n")
    results = manager.install_all_missing(
        progress_callback=lambda msg: print(msg)
    )

    print("\n" + "=" * 50)
    print("INSTALLATION SUMMARY")
    print("=" * 50)

    success_count = sum(1 for ok, _ in results.values() if ok)
    fail_count = len(results) - success_count

    print(f"Successful : {success_count}")
    print(f"Failed     : {fail_count}")

    if fail_count > 0:
        print("\nFailed packages (try installing manually):")
        for pkg, (ok, msg) in results.items():
            if not ok:
                print(f"  - {pkg}")
                print(f"    {msg}")

    return fail_count == 0


if __name__ == "__main__":
    check_dependencies_cli()
