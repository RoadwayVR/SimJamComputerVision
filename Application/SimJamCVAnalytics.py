"""
Traffic Analysis Application - Main GUI
Modern interface for vehicle detection and intersection performance analytics

"""

# ── stdlib only ──────────────────────────────────────────────────────────────
import os
import sys
import subprocess
import importlib
import threading
from tkinter import Tk, messagebox, ttk
import tkinter as tk


# ── Bootstrap: install missing packages before anything else ─────────────────

REQUIRED_PACKAGES = {
    "customtkinter": "customtkinter>=5.2.0",
    "PIL":           "pillow>=10.0.0",
    "cv2":           "opencv-python>=4.8.0",
    "numpy":         "numpy>=1.24.0",
    "pandas":        "pandas>=2.0.0",
    "ultralytics":   "ultralytics>=8.0.0",
    "supervision":   "supervision>=0.17.0",
}


def _check_missing():
    missing = []
    for import_name, pip_spec in REQUIRED_PACKAGES.items():
        try:
            importlib.import_module(import_name)
        except ImportError:
            missing.append(pip_spec)
    return missing


def _install(pip_spec, log_fn=print):
    """Install a package, handling PEP 668 / uv-managed environments."""
    log_fn(f"  Installing {pip_spec} …")
    base_cmd = [sys.executable, "-m", "pip", "install", pip_spec]

    # First attempt: --break-system-packages (required for uv/PEP 668 envs)
    result = subprocess.run(
        base_cmd + ["--break-system-packages"],
        capture_output=True, text=True, timeout=300
    )

    if result.returncode != 0:
        # Second attempt: also add --user (for permission-restricted installs)
        log_fn(f"  Retrying with --user flag …")
        result = subprocess.run(
            base_cmd + ["--user", "--break-system-packages"],
            capture_output=True, text=True, timeout=300
        )

    if result.returncode == 0:
        log_fn(f"  ✓ {pip_spec}")
        return True
    else:
        log_fn(f"  ✗ {pip_spec}: {result.stderr.strip()}")
        return False


class _BootstrapWindow:
    """Minimal tkinter splash shown while packages are installing."""

    def __init__(self, packages):
        self.root = Tk()
        self.root.title("SimJam CV Analytics – First-Run Setup")
        self.root.geometry("540x360")
        self.root.resizable(False, False)
        self.root.configure(bg="#1a1a2e")

        tk.Label(self.root, text="SimJam ComputerVision Analytics",
                 font=("Helvetica", 16, "bold"), fg="white", bg="#1a1a2e").pack(pady=(24, 4))
        tk.Label(self.root, text="Installing required packages – please wait …",
                 font=("Helvetica", 10), fg="#aaaacc", bg="#1a1a2e").pack(pady=(0, 16))

        self.log = tk.Text(self.root, height=10, bg="#0d0d1a", fg="#ccccff",
                           font=("Courier", 9), relief="flat", state="disabled")
        self.log.pack(fill="both", expand=True, padx=20)

        self.bar = ttk.Progressbar(self.root, maximum=len(packages), mode="determinate")
        self.bar.pack(fill="x", padx=20, pady=12)

        self.status = tk.Label(self.root, text="Starting …",
                               font=("Helvetica", 9), fg="#888899", bg="#1a1a2e")
        self.status.pack(pady=(0, 12))

        self.success = False
        self._packages = packages
        self.root.after(100, self._run)

    def _append(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")
        self.root.update_idletasks()

    def _run(self):
        failed = []
        for i, pkg in enumerate(self._packages):
            self.status.configure(text=f"Installing {i+1}/{len(self._packages)}: {pkg.split('>=')[0]}")
            ok = _install(pkg, self._append)
            if not ok:
                failed.append(pkg)
            self.bar["value"] = i + 1
            self.root.update_idletasks()

        if failed:
            self._append(f"\n✗ {len(failed)} package(s) failed – check your internet connection.")
            self.status.configure(text="Some packages failed. The app may not work fully.")
            messagebox.showwarning(
                "Installation Warning",
                f"The following packages could not be installed:\n\n" +
                "\n".join(failed) +
                "\n\nSome features may be unavailable.",
                parent=self.root
            )
        else:
            self._append("\n✓ All packages installed successfully!")
            self.status.configure(text="Done! Launching application …")
            self.success = True

        self.root.after(1200, self.root.destroy)

    def run(self):
        self.root.mainloop()
        return self.success


def bootstrap():
    """Install any missing packages, showing a GUI splash if needed."""
    missing = _check_missing()
    if not missing:
        print("✓ All dependencies already installed.")
        return True

    print(f"Missing {len(missing)} package(s): {', '.join(p.split('>=')[0] for p in missing)}")
    print("Opening installer window …")
    win = _BootstrapWindow(missing)
    win.run()

    # Final verification
    still_missing = _check_missing()
    if still_missing:
        print(f"Still missing: {still_missing}")
        return False
    return True


# ── Run bootstrap BEFORE importing anything third-party ──────────────────────

if __name__ == "__main__" or True:   # also works when imported during testing
    _bootstrap_ok = bootstrap()


# ── Now it is safe to import third-party packages ────────────────────────────

import customtkinter as ctk          # noqa: E402
from tkinter import messagebox       # noqa: E402  (re-import for clarity)
from PIL import Image                # noqa: E402

from dependency_manager import DependencyManager   # noqa: E402
import detection_module                            # noqa: E402
import analytics_module                            # noqa: E402


# ── Main Application ──────────────────────────────────────────────────────────

class TrafficAnalysisApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("SimJamComputerVisionAnalytics")
        self.geometry("1200x800")

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.main_container = ctk.CTkFrame(self, corner_radius=0)
        self.main_container.grid(row=0, column=0, sticky="nsew")
        self.main_container.grid_columnconfigure(0, weight=1)
        self.main_container.grid_rowconfigure(1, weight=1)

        self.create_header()

        self.tabview = ctk.CTkTabview(self.main_container, corner_radius=10)
        self.tabview.grid(row=1, column=0, sticky="nsew", padx=20, pady=(10, 20))

        self.tabview.add("Detection & Tracking")
        self.tabview.add("Performance Analytics")

        for tab_name in ("Detection & Tracking", "Performance Analytics"):
            self.tabview.tab(tab_name).grid_columnconfigure(0, weight=1)
            self.tabview.tab(tab_name).grid_rowconfigure(0, weight=1)

        self.detection_frame = detection_module.DetectionTab(
            self.tabview.tab("Detection & Tracking")
        )
        self.detection_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        self.analytics_frame = analytics_module.AnalyticsTab(
            self.tabview.tab("Performance Analytics")
        )
        self.analytics_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        self.create_status_bar()

    # ── Header ────────────────────────────────────────────────────────────────

    def create_header(self):
        header_frame = ctk.CTkFrame(self.main_container, height=80, corner_radius=0)
        header_frame.grid(row=0, column=0, sticky="ew")
        header_frame.grid_columnconfigure(1, weight=1)
        header_frame.grid_propagate(False)

        left = ctk.CTkFrame(header_frame, fg_color="transparent")
        left.grid(row=0, column=0, padx=30, pady=10, sticky="w", rowspan=2)

        logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "RoadwayVR.jpg")
        if os.path.exists(logo_path):
            try:
                img = Image.open(logo_path).resize((50, 50), Image.Resampling.LANCZOS)
                self.logo_photo = ctk.CTkImage(light_image=img, dark_image=img, size=(50, 50))
                ctk.CTkLabel(left, image=self.logo_photo, text="").pack(side="left", padx=(0, 15))
            except Exception as e:
                print(f"Could not load logo: {e}")

        ctk.CTkLabel(left, text="SimJam ComputerVision Analytics",
                     font=ctk.CTkFont(size=28, weight="bold")).pack(side="left")

        ctk.CTkButton(header_frame, text="ℹ About", width=100,
                      command=self.show_about,
                      fg_color="transparent", border_width=2
                      ).grid(row=0, column=2, rowspan=2, padx=(10, 10), pady=10, sticky="e")

        ctk.CTkButton(header_frame, text="🔧 Dependencies", width=130,
                      command=self.show_dependencies,
                      fg_color=("gray70", "gray30"), border_width=2
                      ).grid(row=0, column=3, rowspan=2, padx=(0, 30), pady=10, sticky="e")

    # ── Status bar ────────────────────────────────────────────────────────────

    def create_status_bar(self):
        self.status_frame = ctk.CTkFrame(self.main_container, height=30, corner_radius=0)
        self.status_frame.grid(row=2, column=0, sticky="ew")
        self.status_frame.grid_propagate(False)

        self.status_label = ctk.CTkLabel(self.status_frame, text="Ready",
                                         font=ctk.CTkFont(size=11))
        self.status_label.pack(side="left", padx=20, pady=5)

    def update_status(self, message):
        self.status_label.configure(text=message)
        self.update_idletasks()

    # ── About dialog ──────────────────────────────────────────────────────────

    def show_about(self):
        w = ctk.CTkToplevel(self)
        w.title("About")
        w.geometry("350x220")
        w.resizable(False, False)
        w.transient(self)
        w.grab_set()

        f = ctk.CTkFrame(w, fg_color="transparent")
        f.pack(fill="both", expand=True, padx=25, pady=20)

        ctk.CTkLabel(f, text="About", font=ctk.CTkFont(size=18, weight="bold")).pack(pady=(0, 12))
        ctk.CTkLabel(f, text="Developed by RoadwayVR",
                     font=ctk.CTkFont(size=13, weight="bold"),
                     text_color="lightblue").pack(pady=(0, 10))
        ctk.CTkLabel(f, text="Application:", font=ctk.CTkFont(size=10),
                     text_color="gray60").pack(pady=(0, 2))
        ctk.CTkLabel(f, text="SimJamCV Analytics",
                     font=ctk.CTkFont(size=12, weight="bold")).pack(pady=(0, 8))
        ctk.CTkLabel(f, text="Version: 1.0",
                     font=ctk.CTkFont(size=11)).pack(pady=(0, 8))
        ctk.CTkLabel(f, text="Traffic analysis tool for\nvehicle detection and analytics",
                     font=ctk.CTkFont(size=10), text_color="gray60",
                     justify="center").pack(pady=(0, 12))
        ctk.CTkButton(f, text="Close", command=w.destroy, width=100, height=28).pack()

    # ── Dependency manager dialog ─────────────────────────────────────────────

    def show_dependencies(self):
        dw = ctk.CTkToplevel(self)
        dw.title("Dependency Manager")
        dw.geometry("600x500")
        dw.transient(self)
        dw.grab_set()
        dw.update_idletasks()
        x = (dw.winfo_screenwidth() // 2) - 300
        y = (dw.winfo_screenheight() // 2) - 250
        dw.geometry(f"+{x}+{y}")

        content = ctk.CTkFrame(dw)
        content.pack(fill="both", expand=True, padx=20, pady=20)

        ctk.CTkLabel(content, text="📦 Package Dependencies",
                     font=ctk.CTkFont(size=20, weight="bold")).pack(pady=(10, 5))
        ctk.CTkLabel(content, text="Check and install required Python packages",
                     font=ctk.CTkFont(size=12), text_color="gray").pack(pady=(0, 15))

        status_text = ctk.CTkTextbox(content, wrap="word",
                                     font=ctk.CTkFont(family="Courier", size=11))
        status_text.pack(fill="both", expand=True, padx=10, pady=10)

        progress = ctk.CTkProgressBar(content)
        progress.pack(fill="x", padx=10, pady=5)
        progress.set(0)

        btn_frame = ctk.CTkFrame(content, fg_color="transparent")
        btn_frame.pack(fill="x", padx=10, pady=10)

        dep_manager = DependencyManager()

        def log(msg):
            status_text.insert("end", f"{msg}\n")
            status_text.see("end")
            dw.update_idletasks()

        def check_deps():
            status_text.delete("1.0", "end")
            log("Checking dependencies…\n")
            status_text.insert("end", dep_manager.get_status_report())
            installed, missing = dep_manager.check_dependencies()
            if missing:
                install_btn.configure(state="normal")
                progress.set(len(installed) / (len(installed) + len(missing)))
            else:
                install_btn.configure(state="disabled")
                progress.set(1.0)

        def install_deps():
            install_btn.configure(state="disabled", text="Installing…")
            check_btn.configure(state="disabled")
            close_btn.configure(state="disabled")
            status_text.delete("1.0", "end")
            log("Starting installation…\n")

            def on_complete(results):
                log("\n" + "=" * 50)
                log("INSTALLATION COMPLETE")
                log("=" * 50 + "\n")
                ok = sum(1 for s, _ in results.values() if s)
                fail = len(results) - ok
                log(f"✓ Successful: {ok}")
                log(f"✗ Failed: {fail}\n")
                if fail:
                    log("Failed packages:")
                    for pkg, (s, _) in results.items():
                        if not s:
                            log(f"  • {pkg}")
                check_deps()
                install_btn.configure(text="⬇ Install Missing")
                check_btn.configure(state="normal")
                close_btn.configure(state="normal")
                if not fail:
                    messagebox.showinfo("Success",
                                        "All dependencies installed successfully!\n\n"
                                        "You can now use all features of the application.",
                                        parent=dw)

            dep_manager.install_all_missing_async(
                progress_callback=log, completion_callback=on_complete)

        check_btn = ctk.CTkButton(btn_frame, text="🔍 Check Status",
                                  command=check_deps, width=150)
        check_btn.pack(side="left", padx=5)

        install_btn = ctk.CTkButton(btn_frame, text="⬇ Install Missing",
                                    command=install_deps, width=150,
                                    fg_color=("green", "darkgreen"))
        install_btn.pack(side="left", padx=5)

        close_btn = ctk.CTkButton(btn_frame, text="Close",
                                  command=dw.destroy, width=100)
        close_btn.pack(side="right", padx=5)

        check_deps()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    app = TrafficAnalysisApp()
    app.mainloop()


if __name__ == "__main__":
    main()
