"""
Detection Module - Vehicle Detection and Tracking Tab
Handles calibration, lane definition, and real-time tracking
"""

import os
import sys
import csv
import inspect
import logging
import traceback
import threading
from collections import defaultdict, deque, Counter
from pathlib import Path

import customtkinter as ctk
from tkinter import filedialog, messagebox, simpledialog
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv

# Import calibration functions from the actual detection code
from calib_and_track_ui import (
    run_tracking, setup_logger
)


class DetectionTab(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent, corner_radius=10)
        
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(0, weight=1)
        
        # Variables
        self.video_path = ctk.StringVar()
        self.weights_path = ctk.StringVar(value="yolo11l.pt")
        self.output_dir = ctk.StringVar()
        self.confidence = ctk.DoubleVar(value=0.5)
        self.nms_iou = ctk.DoubleVar(value=0.5)
        self.imgsz = ctk.IntVar(value=1280)
        self.blur_outside = ctk.BooleanVar(value=True)
        self.show_preview = ctk.BooleanVar(value=True)
        self.force_cpu = ctk.BooleanVar(value=False)
        
        self.is_running = False
        
        # Create UI sections
        self.create_input_section()
        self.create_settings_section()
        
    def create_input_section(self):
        """Create input files and parameters section"""
        input_frame = ctk.CTkFrame(self, corner_radius=10)
        input_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=0)
        input_frame.grid_columnconfigure(0, weight=1)
        
        # Title
        title = ctk.CTkLabel(
            input_frame,
            text="Detection & Tracking Configuration",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        title.grid(row=0, column=0, columnspan=2, padx=20, pady=(20, 10), sticky="w")
        
        # Scrollable frame for inputs
        scroll_frame = ctk.CTkScrollableFrame(input_frame, corner_radius=5)
        scroll_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=20, pady=10)
        input_frame.grid_rowconfigure(1, weight=1)
        scroll_frame.grid_columnconfigure(1, weight=1)
        
        row = 0
        
        # Input Video
        self.create_file_input(
            scroll_frame, row, "Input Video:", self.video_path,
            self.browse_video, "Select video file (MP4, AVI, MOV, MKV)"
        )
        row += 1
        
        # YOLO Weights
        self.create_file_input(
            scroll_frame, row, "YOLO Weights:", self.weights_path,
            self.browse_weights, "Select YOLO weights file (.pt)"
        )
        row += 1
        
        # Output Directory
        self.create_file_input(
            scroll_frame, row, "Output Folder:", self.output_dir,
            self.browse_output, "Select output directory", is_dir=True
        )
        row += 1
        
        # Separator
        sep1 = ctk.CTkFrame(scroll_frame, height=2, fg_color=("gray80", "gray20"))
        sep1.grid(row=row, column=0, columnspan=3, sticky="ew", padx=5, pady=15)
        row += 1
        
        # Detection Parameters Section
        params_label = ctk.CTkLabel(
            scroll_frame,
            text="Detection Parameters",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        params_label.grid(row=row, column=0, columnspan=2, padx=5, pady=(10, 5), sticky="w")
        row += 1
        
        # Confidence Threshold
        self.create_slider_input(
            scroll_frame, row, "Confidence Threshold:", self.confidence, 0.1, 0.95
        )
        row += 1
        
        # NMS IoU Threshold
        self.create_slider_input(
            scroll_frame, row, "NMS IoU Threshold:", self.nms_iou, 0.1, 0.95
        )
        row += 1
        
        # Image Size
        self.create_slider_input(
            scroll_frame, row, "YOLO Image Size:", self.imgsz, 640, 1920, is_int=True
        )
        row += 1
        
        # Separator
        sep2 = ctk.CTkFrame(scroll_frame, height=2, fg_color=("gray80", "gray20"))
        sep2.grid(row=row, column=0, columnspan=3, sticky="ew", padx=5, pady=15)
        row += 1
        
        # Options Section
        options_label = ctk.CTkLabel(
            scroll_frame,
            text="Processing Options",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        options_label.grid(row=row, column=0, columnspan=2, padx=5, pady=(10, 5), sticky="w")
        row += 1
        
        # Checkboxes
        blur_check = ctk.CTkCheckBox(
            scroll_frame,
            text="Blur outside ROI",
            variable=self.blur_outside
        )
        blur_check.grid(row=row, column=0, columnspan=2, padx=5, pady=5, sticky="w")
        row += 1
        
        preview_check = ctk.CTkCheckBox(
            scroll_frame,
            text="Show preview window",
            variable=self.show_preview
        )
        preview_check.grid(row=row, column=0, columnspan=2, padx=5, pady=5, sticky="w")
        row += 1
        
        cpu_check = ctk.CTkCheckBox(
            scroll_frame,
            text="Force CPU (disable GPU)",
            variable=self.force_cpu
        )
        cpu_check.grid(row=row, column=0, columnspan=2, padx=5, pady=5, sticky="w")
        row += 1
        
        # Separator
        sep3 = ctk.CTkFrame(scroll_frame, height=2, fg_color=("gray80", "gray20"))
        sep3.grid(row=row, column=0, columnspan=3, sticky="ew", padx=5, pady=15)
        row += 1
        
        # Output Files Preview
        outputs_label = ctk.CTkLabel(
            scroll_frame,
            text="Generated Output Files",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        outputs_label.grid(row=row, column=0, columnspan=2, padx=5, pady=(10, 5), sticky="w")
        row += 1
        
        self.outputs_text = ctk.CTkTextbox(scroll_frame, height=120, state="disabled")
        self.outputs_text.grid(row=row, column=0, columnspan=3, sticky="ew", padx=5, pady=5)
        row += 1
        
        # Run Button
        self.run_btn = ctk.CTkButton(
            input_frame,
            text="▶ Start Detection & Tracking",
            command=self.run_detection,
            height=50,
            font=ctk.CTkFont(size=16, weight="bold"),
            corner_radius=10
        )
        self.run_btn.grid(row=2, column=0, columnspan=2, padx=20, pady=20, sticky="ew")
        
    def create_settings_section(self):
        """Create settings and info section"""
        settings_frame = ctk.CTkFrame(self, corner_radius=10)
        settings_frame.grid(row=0, column=1, sticky="nsew", padx=0, pady=0)
        settings_frame.grid_columnconfigure(0, weight=1)
        settings_frame.grid_rowconfigure(1, weight=1)
        
        # Title
        title = ctk.CTkLabel(
            settings_frame,
            text="Process Information",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        title.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="w")
        
        # Info tabs
        info_tabs = ctk.CTkTabview(settings_frame, corner_radius=8)
        info_tabs.grid(row=1, column=0, sticky="nsew", padx=20, pady=10)
        
        info_tabs.add("Workflow")
        info_tabs.add("Features")
        info_tabs.add("Log")
        
        # Workflow tab
        workflow_text = """
Process Steps:

1. Video Selection
   • Choose input video file
   • Select YOLO weights
   • Set output directory

2. Calibration
   • Select frame from video
   • Draw 4-point ROI polygon
   • Enter real-world dimensions

3. Lane Definition
   • Draw multiple lane polygons
   • Lanes must not overlap
   • Save lane configurations

4. Detection & Tracking
   • YOLO object detection
   • ByteTrack multi-object tracking
   • Real-time speed calculation
   • Video output with overlays

5. Output Generation
   • Annotated video
   • Vehicle summary CSV
   • Trajectory data CSV
   • Lane configuration CSV
        """
        
        workflow_label = ctk.CTkTextbox(
            info_tabs.tab("Workflow"),
            wrap="word",
            font=ctk.CTkFont(size=11)
        )
        workflow_label.pack(fill="both", expand=True, padx=10, pady=10)
        workflow_label.insert("1.0", workflow_text)
        workflow_label.configure(state="disabled")
        
        # Features tab
        features_text = """
Key Features:

✓ Interactive Calibration
  Real-world coordinate mapping

✓ Multi-Lane Support
  Define unlimited lanes per scene

✓ Advanced Tracking
  ByteTrack algorithm for robust
  vehicle following across frames

✓ Speed Estimation
  Real-time km/h calculation
  based on calibration

✓ Stationary Detection
  Handles stopped vehicles at
  traffic signals correctly

✓ Export Options
  • Annotated video output
  • Per-vehicle statistics
  • Frame-by-frame trajectories
  • Lane polygon coordinates

✓ Performance
  GPU acceleration support
  Batch processing capability
        """
        
        features_label = ctk.CTkTextbox(
            info_tabs.tab("Features"),
            wrap="word",
            font=ctk.CTkFont(size=11)
        )
        features_label.pack(fill="both", expand=True, padx=10, pady=10)
        features_label.insert("1.0", features_text)
        features_label.configure(state="disabled")
        
        # Log tab
        self.log_text = ctk.CTkTextbox(
            info_tabs.tab("Log"),
            wrap="word",
            font=ctk.CTkFont(family="Courier", size=10)
        )
        self.log_text.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Progress bar
        self.progress = ctk.CTkProgressBar(settings_frame)
        self.progress.grid(row=2, column=0, sticky="ew", padx=20, pady=10)
        self.progress.set(0)
        
        # Status label
        self.status_label = ctk.CTkLabel(
            settings_frame,
            text="Ready to process",
            font=ctk.CTkFont(size=12)
        )
        self.status_label.grid(row=3, column=0, padx=20, pady=(0, 20))
        
    def create_file_input(self, parent, row, label, variable, command, tooltip, is_dir=False):
        """Create a file input row"""
        lbl = ctk.CTkLabel(parent, text=label, font=ctk.CTkFont(size=12))
        lbl.grid(row=row, column=0, padx=5, pady=8, sticky="w")
        
        entry = ctk.CTkEntry(parent, textvariable=variable)
        entry.grid(row=row, column=1, padx=5, pady=8, sticky="ew")
        
        btn = ctk.CTkButton(
            parent, text="Browse", command=command, width=80
        )
        btn.grid(row=row, column=2, padx=5, pady=8)
        
    def create_slider_input(self, parent, row, label, variable, min_val, max_val, is_int=False):
        """Create a slider input row"""
        lbl = ctk.CTkLabel(parent, text=label, font=ctk.CTkFont(size=12))
        lbl.grid(row=row, column=0, padx=5, pady=8, sticky="w")
        
        value_label = ctk.CTkLabel(parent, text=f"{variable.get():.2f}" if not is_int else f"{variable.get()}")
        value_label.grid(row=row, column=2, padx=5, pady=8)
        
        def update_label(val):
            if is_int:
                value_label.configure(text=f"{int(float(val))}")
            else:
                value_label.configure(text=f"{float(val):.2f}")
        
        slider = ctk.CTkSlider(
            parent, from_=min_val, to=max_val, variable=variable,
            command=update_label
        )
        slider.grid(row=row, column=1, padx=5, pady=8, sticky="ew")
        
    def browse_video(self):
        """Browse for video file"""
        path = filedialog.askopenfilename(
            title="Select Input Video",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                ("All files", "*.*")
            ]
        )
        if path:
            self.video_path.set(path)
            if not self.output_dir.get():
                self.output_dir.set(os.path.dirname(path))
            self.update_outputs()
            
    def browse_weights(self):
        """Browse for weights file"""
        path = filedialog.askopenfilename(
            title="Select YOLO Weights",
            filetypes=[
                ("PyTorch weights", "*.pt"),
                ("All files", "*.*")
            ]
        )
        if path:
            self.weights_path.set(path)
            
    def browse_output(self):
        """Browse for output directory"""
        path = filedialog.askdirectory(title="Select Output Folder")
        if path:
            self.output_dir.set(path)
            self.update_outputs()
            
    def update_outputs(self):
        """Update output files preview"""
        out_dir = self.output_dir.get()
        if not out_dir:
            return
            
        outputs = [
            "📹 vehicles-result.mp4",
            "📊 vehicles.csv",
            "📍 vehicle_tracks_xy.csv",
            "🛣️ lanes.csv",
            "🖼️ calib_screens/ (folder)",
            "📋 runtime.log"
        ]
        
        self.outputs_text.configure(state="normal")
        self.outputs_text.delete("1.0", "end")
        self.outputs_text.insert("1.0", "\n".join(outputs))
        self.outputs_text.configure(state="disabled")
        
    def log(self, message):
        """Add message to log"""
        self.log_text.insert("end", f"{message}\n")
        self.log_text.see("end")
        self.update_idletasks()
        
    def run_detection(self):
        """Run detection process"""
        if self.is_running:
            messagebox.showwarning("Already Running", "Detection process is already running!")
            return
            
        # Validate inputs
        if not self.video_path.get() or not os.path.exists(self.video_path.get()):
            messagebox.showerror("Error", "Please select a valid input video.")
            return
            
        if not self.weights_path.get() or not os.path.exists(self.weights_path.get()):
            messagebox.showerror("Error", "Please select valid YOLO weights.")
            return
            
        if not self.output_dir.get():
            messagebox.showerror("Error", "Please select an output directory.")
            return
            
        # Run in thread
        self.is_running = True
        self.run_btn.configure(state="disabled", text="⏳ Processing...")
        self.progress.set(0.1)
        self.status_label.configure(text="Starting detection process...")
        
        thread = threading.Thread(target=self._run_detection_thread, daemon=True)
        thread.start()
        
    def _run_detection_thread(self):
        """Detection thread worker"""
        try:
            self.log("=== Starting Detection Process ===")
            
            # Prepare paths
            out_dir = self.output_dir.get()
            os.makedirs(out_dir, exist_ok=True)
            
            video_out = os.path.join(out_dir, "vehicles-result.mp4")
            csv_sum = os.path.join(out_dir, "vehicles.csv")
            csv_tracks = os.path.join(out_dir, "vehicle_tracks_xy.csv")
            csv_lanes = os.path.join(out_dir, "lanes.csv")
            calib_dir = os.path.join(out_dir, "calib_screens")
            log_path = os.path.join(out_dir, "runtime.log")
            
            setup_logger(log_path)
            
            self.log(f"Input: {os.path.basename(self.video_path.get())}")
            self.log(f"Weights: {os.path.basename(self.weights_path.get())}")
            self.log(f"Output: {out_dir}")
            
            self.progress.set(0.3)
            self.status_label.configure(text="Running calibration...")
            
            # Create a hidden tkinter root for calibration dialogs
            import tkinter as tk
            hidden_root = tk.Tk()
            hidden_root.withdraw()
            
            # Run the actual tracking with your code
            run_tracking(
                source_video_path=self.video_path.get(),
                target_video_path=video_out,
                csv_summary_path=csv_sum,
                csv_tracks_path=csv_tracks,
                csv_lanes_path=csv_lanes,
                calib_save_dir=calib_dir,
                calib_display_w=1280,
                weights_path=self.weights_path.get(),
                confidence_threshold=self.confidence.get(),
                nms_iou=self.nms_iou.get(),
                imgsz=self.imgsz.get(),
                blur_outside=self.blur_outside.get(),
                show_preview=self.show_preview.get(),
                force_cpu=self.force_cpu.get(),
                root=hidden_root
            )
            
            hidden_root.destroy()
            
            self.progress.set(1.0)
            self.status_label.configure(text="✓ Detection completed successfully!")
            self.log("=== Detection Process Completed ===")
            
            messagebox.showinfo(
                "Success",
                "Detection and tracking completed!\n\n"
                "Next step: Run Analytics tab for performance metrics."
            )
            
        except Exception as e:
            self.log(f"ERROR: {str(e)}")
            self.log(traceback.format_exc())
            self.status_label.configure(text="✗ Error occurred")
            messagebox.showerror("Error", f"Detection failed:\n{str(e)}")
            
        finally:
            self.is_running = False
            self.run_btn.configure(state="normal", text="▶ Start Detection & Tracking")
            self.progress.set(0)
