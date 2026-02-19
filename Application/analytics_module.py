"""
Analytics Module - Intersection Performance Analysis Tab
Handles trajectory processing, lane assignment, and metrics calculation
"""

import os
import sys
import csv
import threading
import traceback
from pathlib import Path

import customtkinter as ctk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import cv2

# Import the modified analyzer
from cv_intersection_performance_analysis import CVIntersectionAnalyzer


class AnalyticsTab(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent, corner_radius=10)
        
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(0, weight=1)
        
        # Variables
        self.data_dir = ctk.StringVar()
        self.lanes_csv = ctk.StringVar()
        self.tracks_csv = ctk.StringVar()
        self.video_path = ctk.StringVar()
        
        # Optional metrics
        self.include_delay = ctk.BooleanVar(value=False)
        self.include_los = ctk.BooleanVar(value=False)
        
        self.is_running = False
        
        # Create UI sections
        self.create_input_section()
        self.create_results_section()
        
    def create_input_section(self):
        """Create input section"""
        input_frame = ctk.CTkFrame(self, corner_radius=10)
        input_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=0)
        input_frame.grid_columnconfigure(0, weight=1)
        
        # Title
        title = ctk.CTkLabel(
            input_frame,
            text="Performance Analytics Configuration",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        title.grid(row=0, column=0, columnspan=2, padx=20, pady=(20, 10), sticky="w")
        
        # Scrollable frame
        scroll_frame = ctk.CTkScrollableFrame(input_frame, corner_radius=5)
        scroll_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=20, pady=10)
        input_frame.grid_rowconfigure(1, weight=1)
        scroll_frame.grid_columnconfigure(1, weight=1)
        
        row = 0
        
        # Data Directory
        lbl = ctk.CTkLabel(scroll_frame, text="Data Directory:", font=ctk.CTkFont(size=12))
        lbl.grid(row=row, column=0, padx=5, pady=8, sticky="w")
        
        entry = ctk.CTkEntry(scroll_frame, textvariable=self.data_dir)
        entry.grid(row=row, column=1, padx=5, pady=8, sticky="ew")
        
        btn = ctk.CTkButton(scroll_frame, text="Browse", command=self.browse_data_dir, width=80)
        btn.grid(row=row, column=2, padx=5, pady=8)
        row += 1
        
        # Auto-load button
        auto_btn = ctk.CTkButton(
            scroll_frame,
            text="📂 Auto-Load Files from Directory",
            command=self.auto_load_files,
            fg_color=("gray70", "gray30")
        )
        auto_btn.grid(row=row, column=0, columnspan=3, padx=5, pady=5, sticky="ew")
        row += 1
        
        # Separator
        sep1 = ctk.CTkFrame(scroll_frame, height=2, fg_color=("gray80", "gray20"))
        sep1.grid(row=row, column=0, columnspan=3, sticky="ew", padx=5, pady=15)
        row += 1
        
        # Or manual selection
        manual_label = ctk.CTkLabel(
            scroll_frame,
            text="Or Select Files Manually",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=("gray50", "gray50")
        )
        manual_label.grid(row=row, column=0, columnspan=3, padx=5, pady=(5, 10))
        row += 1
        
        # Lanes CSV
        self.create_file_input(
            scroll_frame, row, "Lanes CSV:", self.lanes_csv, self.browse_lanes
        )
        row += 1
        
        # Tracks CSV
        self.create_file_input(
            scroll_frame, row, "Tracks CSV:", self.tracks_csv, self.browse_tracks
        )
        row += 1
        
        # Video (optional)
        self.create_file_input(
            scroll_frame, row, "Video (optional):", self.video_path, self.browse_video
        )
        row += 1
        
        # Separator
        sep2 = ctk.CTkFrame(scroll_frame, height=2, fg_color=("gray80", "gray20"))
        sep2.grid(row=row, column=0, columnspan=3, sticky="ew", padx=5, pady=15)
        row += 1
        
        # Info Section
        info_label = ctk.CTkLabel(
            scroll_frame,
            text="📊 Analysis will output minute-by-minute data:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        info_label.grid(row=row, column=0, columnspan=3, padx=5, pady=(10, 5), sticky="w")
        row += 1
        
        metrics_info = ctk.CTkLabel(
            scroll_frame,
            text="""
Output CSV Format: Minute, lane_id, n_vehicles, avg_speed_kmh

• Minute - Time interval (1, 2, 3, ...)
• lane_id - Lane identifier
• n_vehicles - NEW unique vehicles in this minute in THIS LANE
  (each vehicle counted only once per lane)
• avg_speed_kmh - Average speed in lane during this minute
            """,
            font=ctk.CTkFont(size=11),
            justify="left",
            text_color=("gray30", "gray70")
        )
        metrics_info.grid(row=row, column=0, columnspan=3, padx=20, pady=5, sticky="w")
        row += 1
        
        # Separator
        sep3 = ctk.CTkFrame(scroll_frame, height=2, fg_color=("gray80", "gray20"))
        sep3.grid(row=row, column=0, columnspan=3, sticky="ew", padx=5, pady=15)
        row += 1
        
        # Optional Metrics Section
        optional_label = ctk.CTkLabel(
            scroll_frame,
            text="📊 Optional Metrics (per minute per lane):",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        optional_label.grid(row=row, column=0, columnspan=3, padx=5, pady=(10, 5), sticky="w")
        row += 1
        
        # Checkboxes for optional metrics
        check1 = ctk.CTkCheckBox(
            scroll_frame,
            text="avg_delay_s - Average control delay",
            variable=self.include_delay
        )
        check1.grid(row=row, column=0, columnspan=3, padx=20, pady=3, sticky="w")
        row += 1
        
        check2 = ctk.CTkCheckBox(
            scroll_frame,
            text="los - Level of Service (A-F)",
            variable=self.include_los
        )
        check2.grid(row=row, column=0, columnspan=3, padx=20, pady=3, sticky="w")
        row += 1
        
        # Run Button
        self.run_btn = ctk.CTkButton(
            input_frame,
            text="📊 Run Performance Analysis",
            command=self.run_analytics,
            height=50,
            font=ctk.CTkFont(size=16, weight="bold"),
            corner_radius=10,
            fg_color=("#2CC985", "#2FA572")
        )
        self.run_btn.grid(row=2, column=0, columnspan=2, padx=20, pady=20, sticky="ew")
        
    def create_results_section(self):
        """Create results display section"""
        results_frame = ctk.CTkFrame(self, corner_radius=10)
        results_frame.grid(row=0, column=1, sticky="nsew", padx=0, pady=0)
        results_frame.grid_columnconfigure(0, weight=1)
        results_frame.grid_rowconfigure(1, weight=1)
        
        # Title
        title = ctk.CTkLabel(
            results_frame,
            text="Analysis Results",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        title.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="w")
        
        # Results tabs
        results_tabs = ctk.CTkTabview(results_frame, corner_radius=8)
        results_tabs.grid(row=1, column=0, sticky="nsew", padx=20, pady=10)
        
        results_tabs.add("Summary")
        results_tabs.add("Log")
        results_tabs.add("Help")
        
        # Summary tab
        self.summary_text = ctk.CTkTextbox(
            results_tabs.tab("Summary"),
            wrap="word",
            font=ctk.CTkFont(size=11)
        )
        self.summary_text.pack(fill="both", expand=True, padx=10, pady=10)
        self.summary_text.insert("1.0", "Run analysis to see results...")
        
        # Log tab
        self.log_text = ctk.CTkTextbox(
            results_tabs.tab("Log"),
            wrap="word",
            font=ctk.CTkFont(family="Courier", size=10)
        )
        self.log_text.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Help tab
        help_text = """
Performance Analytics Guide

Input Requirements:
━━━━━━━━━━━━━━━━━━━━
✓ lanes.csv: Lane polygon coordinates
✓ vehicle_tracks_xy.csv: Trajectory data with lane assignments

Output Format (Minute-by-Minute):
━━━━━━━━━━━━━━━━━━━━
CSV with columns: Minute, lane_id, n_vehicles, avg_speed_kmh
[+ optional: avg_delay_s, los]

Required Columns:
• Minute
  Time interval: 1, 2, 3, ... (starts from 1)

• lane_id
  Lane identifier (from lanes.csv)

• n_vehicles
  NEW unique vehicles entering this minute in THIS LANE
  Important: Each vehicle counted only ONCE per lane
  (not repeated in subsequent minutes in same lane)

• avg_speed_kmh
  Average speed of vehicles in this lane
  during this minute

Optional Columns (if enabled):
• avg_delay_s
  Average control delay in seconds
  (actual travel time - free flow time)

• los
  Level of Service: A (best) to F (worst)
  Based on control delay

Example Output:
━━━━━━━━━━━━━━━━━━━━
Minute,lane_id,n_vehicles,avg_speed_kmh
1,1,5,32.4
1,2,3,45.2
2,1,7,28.9
2,2,4,42.1

With optional metrics:
Minute,lane_id,n_vehicles,avg_speed_kmh,avg_delay_s,los
1,1,5,32.4,8.5,C
1,2,3,45.2,3.2,A

Output Files:
━━━━━━━━━━━━━━━━━━━━
📋 lane_metrics.csv
   Minute-by-minute traffic data

📋 cv_metrics.json
   Complete analysis results

📋 analysis_summary.txt
   Text report

Tips:
━━━━━━━━━━━━━━━━━━━━
→ Run detection & tracking first
→ Use auto-load for convenience
→ Each vehicle counted once per lane
→ Enable optional metrics as needed
→ Data suitable for time-series analysis
        """
        
        help_label = ctk.CTkTextbox(
            results_tabs.tab("Help"),
            wrap="word",
            font=ctk.CTkFont(size=11)
        )
        help_label.pack(fill="both", expand=True, padx=10, pady=10)
        help_label.insert("1.0", help_text)
        help_label.configure(state="disabled")
        
        # Progress
        self.progress = ctk.CTkProgressBar(results_frame)
        self.progress.grid(row=2, column=0, sticky="ew", padx=20, pady=10)
        self.progress.set(0)
        
        # Status
        self.status_label = ctk.CTkLabel(
            results_frame,
            text="Ready to analyze",
            font=ctk.CTkFont(size=12)
        )
        self.status_label.grid(row=3, column=0, padx=20, pady=(0, 20))
        
    def create_file_input(self, parent, row, label, variable, command):
        """Create file input row"""
        lbl = ctk.CTkLabel(parent, text=label, font=ctk.CTkFont(size=12))
        lbl.grid(row=row, column=0, padx=5, pady=8, sticky="w")
        
        entry = ctk.CTkEntry(parent, textvariable=variable)
        entry.grid(row=row, column=1, padx=5, pady=8, sticky="ew")
        
        btn = ctk.CTkButton(parent, text="Browse", command=command, width=80)
        btn.grid(row=row, column=2, padx=5, pady=8)
        
    def browse_data_dir(self):
        """Browse for data directory"""
        path = filedialog.askdirectory(title="Select Data Directory")
        if path:
            self.data_dir.set(path)
            
    def browse_lanes(self):
        """Browse for lanes CSV"""
        path = filedialog.askopenfilename(
            title="Select Lanes CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if path:
            self.lanes_csv.set(path)
            
    def browse_tracks(self):
        """Browse for tracks CSV"""
        path = filedialog.askopenfilename(
            title="Select Tracks CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if path:
            self.tracks_csv.set(path)
            
    def browse_video(self):
        """Browse for video file"""
        path = filedialog.askopenfilename(
            title="Select Video (Optional)",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                ("All files", "*.*")
            ]
        )
        if path:
            self.video_path.set(path)
            
    def auto_load_files(self):
        """Auto-load files from data directory"""
        data_dir = self.data_dir.get()
        if not data_dir or not os.path.exists(data_dir):
            messagebox.showerror("Error", "Please select a valid data directory first.")
            return
            
        # Look for expected files
        lanes = os.path.join(data_dir, "lanes.csv")
        tracks = os.path.join(data_dir, "vehicle_tracks_xy.csv")
        
        found = []
        if os.path.exists(lanes):
            self.lanes_csv.set(lanes)
            found.append("lanes.csv")
        if os.path.exists(tracks):
            self.tracks_csv.set(tracks)
            found.append("vehicle_tracks_xy.csv")
            
        # Look for video
        video_patterns = ["vehicles-result.mp4", "*.mp4", "*.avi", "*.mov"]
        for pattern in video_patterns:
            if '*' in pattern:
                import glob
                matches = glob.glob(os.path.join(data_dir, pattern))
                if matches:
                    self.video_path.set(matches[0])
                    found.append(os.path.basename(matches[0]))
                    break
            else:
                video = os.path.join(data_dir, pattern)
                if os.path.exists(video):
                    self.video_path.set(video)
                    found.append(pattern)
                    break
                    
        if found:
            msg = "Auto-loaded files:\n\n" + "\n".join(f"✓ {f}" for f in found)
            messagebox.showinfo("Files Loaded", msg)
            self.log(f"Auto-loaded {len(found)} file(s)")
        else:
            messagebox.showwarning("No Files Found", "Could not find expected files in directory.")
            
    def log(self, message):
        """Add to log"""
        self.log_text.insert("end", f"{message}\n")
        self.log_text.see("end")
        self.update_idletasks()
        
    def run_analytics(self):
        """Run analytics process"""
        if self.is_running:
            messagebox.showwarning("Already Running", "Analysis is already running!")
            return
            
        # Validate
        if not self.lanes_csv.get() or not os.path.exists(self.lanes_csv.get()):
            messagebox.showerror("Error", "Please select lanes.csv file.")
            return
            
        if not self.tracks_csv.get() or not os.path.exists(self.tracks_csv.get()):
            messagebox.showerror("Error", "Please select vehicle_tracks_xy.csv file.")
            return
            
        # Run in thread
        self.is_running = True
        self.run_btn.configure(state="disabled", text="⏳ Analyzing...")
        self.progress.set(0.1)
        self.status_label.configure(text="Starting analysis...")
        
        self.summary_text.delete("1.0", "end")
        self.summary_text.insert("1.0", "Processing...\n")
        
        thread = threading.Thread(target=self._run_analytics_thread, daemon=True)
        thread.start()
        
    def _run_analytics_thread(self):
        """Analytics thread worker"""
        try:
            self.log("=== Starting Performance Analysis ===")
            
            lanes_csv = self.lanes_csv.get()
            tracks_csv = self.tracks_csv.get()
            video_path = self.video_path.get() if self.video_path.get() else None
            
            output_dir = os.path.dirname(tracks_csv)
            
            self.log(f"Lanes: {os.path.basename(lanes_csv)}")
            self.log(f"Tracks: {os.path.basename(tracks_csv)}")
            self.log(f"Output: {output_dir}")
            
            self.progress.set(0.2)
            self.status_label.configure(text="Loading data...")
            
            # Load and set lanes
            self.log("Loading lanes...")
            with open(lanes_csv, 'r') as f:
                import csv as csv_module
                reader = csv_module.DictReader(f)
                lanes = {}
                for row in reader:
                    lane_id = int(row['lane_id'])
                    lanes[lane_id] = [
                        [float(row['p1_x']), float(row['p1_y'])],
                        [float(row['p2_x']), float(row['p2_y'])],
                        [float(row['p3_x']), float(row['p3_y'])],
                        [float(row['p4_x']), float(row['p4_y'])]
                    ]
            self.log(f"Loaded {len(lanes)} lanes: {sorted(lanes.keys())}")
            
            self.progress.set(0.3)
            self.status_label.configure(text="Assigning lanes to trajectories...")
            
            # First, assign lanes to the trajectory data
            self.log("Assigning lanes to trajectory points...")
            from cv_intersection_performance_analysis import assign_lane_to_point
            
            # Read the tracks CSV
            tracks_df = pd.read_csv(tracks_csv)
            
            self.log(f"Original trajectory points: {len(tracks_df)}")
            
            # Assign lane_id to each point
            lane_ids = []
            for _, row in tracks_df.iterrows():
                lane_id = assign_lane_to_point(row['img_x'], row['img_y'], lanes)
                lane_ids.append(lane_id if lane_id is not None else -1)
            
            tracks_df['lane_id'] = lane_ids
            
            # Filter out points with invalid lane_id
            points_before = len(tracks_df)
            tracks_df = tracks_df[tracks_df['lane_id'] >= 0].copy()
            points_after = len(tracks_df)
            removed_points = points_before - points_after
            
            self.log(f"Filtered out {removed_points} points with invalid lane_id")
            self.log(f"Remaining points: {points_after}")
            
            # Filter out tracks that are not in the defined lanes
            valid_lane_ids = set(lanes.keys())
            tracks_before = len(tracks_df)
            tracks_df = tracks_df[tracks_df['lane_id'].isin(valid_lane_ids)].copy()
            tracks_after = len(tracks_df)
            removed_outside_lanes = tracks_before - tracks_after
            
            if removed_outside_lanes > 0:
                self.log(f"Filtered out {removed_outside_lanes} points outside defined lanes")
            
            self.log(f"Final trajectory points: {len(tracks_df)}")
            self.log(f"Unique vehicles: {tracks_df['vehicle_id'].nunique()}")
            
            # Save the filtered tracks with lane assignments to the output directory
            tracks_with_lanes = os.path.join(output_dir, "vehicle_tracks_with_lanes.csv")
            tracks_df.to_csv(tracks_with_lanes, index=False)
            self.log(f"Saved filtered tracks with lane assignments to {os.path.basename(tracks_with_lanes)}")
            
            self.progress.set(0.4)
            self.status_label.configure(text="Initializing analyzer...")
            
            # Now initialize the analyzer with the lane-assigned data
            self.log("Initializing analyzer with lane-assigned data...")
            analyzer = CVIntersectionAnalyzer(
                trajectory_file=tracks_with_lanes,
                free_flow_speed_kmh=50.0,  # Default free-flow speed
                verbose=False
            )
            
            analyzer.lane_polygons = lanes
            self.log(f"Analyzer ready with {len(analyzer.df)} points")
            
            self.progress.set(0.5)
            self.status_label.configure(text="Running analysis...")
            
            # Run the analysis (this does lane assignment, metrics, everything)
            self.log("Running complete analysis...")
            analyzer.analyze(print_results=False)
            self.log("Analysis complete")
            
            self.progress.set(0.8)
            self.status_label.configure(text="Exporting results...")
            
            # Export results to the same output directory
            self.log("Exporting results...")
            results_dir = output_dir  # Use the same output directory, not a subdirectory
            
            # Use the analyzer's export method with optional metrics
            csv_file = os.path.join(results_dir, "lane_metrics.csv")
            analyzer.export_csv(
                csv_file,
                include_delay=self.include_delay.get(),
                include_los=self.include_los.get()
            )
            self.log(f"Saved lane metrics to {os.path.basename(csv_file)}")
            
            # Also create a summary text file
            summary_file = os.path.join(results_dir, "analysis_summary.txt")
            with open(summary_file, 'w') as f:
                f.write("INTERSECTION PERFORMANCE ANALYSIS\n")
                f.write("=" * 70 + "\n\n")
                
                # Write lane group metrics
                import dataclasses
                for lane_id, metrics in analyzer.lane_metrics.items():
                    f.write(f"\nLane {lane_id}\n")
                    f.write("-" * 40 + "\n")
                    if dataclasses.is_dataclass(metrics):
                        metrics_dict = dataclasses.asdict(metrics)
                        for key, value in metrics_dict.items():
                            f.write(f"  {key}: {value}\n")
                
                # Write global metrics if available
                if hasattr(analyzer, 'global_metrics') and analyzer.global_metrics:
                    f.write(f"\n\nGLOBAL METRICS\n")
                    f.write("-" * 40 + "\n")
                    for key, value in analyzer.global_metrics.items():
                        f.write(f"  {key}: {value}\n")
            
            self.log(f"Saved summary to {summary_file}")
            
            self.progress.set(1.0)
            self.status_label.configure(text="✓ Analysis completed!")
            
            # Display summary
            summary = "INTERSECTION PERFORMANCE ANALYSIS\n"
            summary += "=" * 50 + "\n\n"
            
            import dataclasses
            for lane_id, metrics in analyzer.lane_metrics.items():
                summary += f"Lane {lane_id}\n"
                summary += "-" * 30 + "\n"
                if dataclasses.is_dataclass(metrics):
                    metrics_dict = dataclasses.asdict(metrics)
                    for key, value in metrics_dict.items():
                        if isinstance(value, float):
                            summary += f"  {key}: {value:.2f}\n"
                        else:
                            summary += f"  {key}: {value}\n"
                summary += "\n"
            
            self.summary_text.delete("1.0", "end")
            self.summary_text.insert("1.0", summary)
            
            self.log("=== Analysis Complete ===")
            self.log(f"Results saved to: {results_dir}")
            
            messagebox.showinfo(
                "Success",
                f"Analysis completed successfully!\n\n"
                f"Results saved to:\n{results_dir}"
            )
            
        except Exception as e:
            self.log(f"ERROR: {str(e)}")
            self.log(traceback.format_exc())
            self.status_label.configure(text="✗ Error occurred")
            self.summary_text.delete("1.0", "end")
            self.summary_text.insert("1.0", f"Error:\n{str(e)}")
            messagebox.showerror("Error", f"Analysis failed:\n{str(e)}")
            
        finally:
            self.is_running = False
            self.run_btn.configure(state="normal", text="📊 Run Performance Analysis")
            if self.progress.get() < 1.0:
                self.progress.set(0)
                
    def _format_summary(self, metrics):
        """Format metrics summary"""
        summary = "INTERSECTION PERFORMANCE ANALYSIS\n"
        summary += "=" * 50 + "\n\n"
        
        if not metrics:
            return summary + "No metrics calculated."
            
        for lane_id, lane_metrics in metrics.items():
            summary += f"Lane {lane_id}\n"
            summary += "-" * 30 + "\n"
            
            for metric_name, value in lane_metrics.items():
                if isinstance(value, (int, float)):
                    summary += f"  {metric_name}: {value:.2f}\n"
                else:
                    summary += f"  {metric_name}: {value}\n"
            summary += "\n"
            
        return summary