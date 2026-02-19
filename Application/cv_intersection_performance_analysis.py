"""
CV INTERSECTION ANALYSIS - MODIFIED VERSION
Changes:
1. avg_delay_s = Lane Average Control Delay (time_loss for all vehicles)
2. avg_stopped_delay_s = Average time spent completely stopped (for all vehicles)
3. Removed columns: measured_sat_flow_veh_h, estimated_capacity_veh_h, green_time_s, green_ratio, vc_ratio
4. Ensures: avg_delay_s >= avg_stopped_delay_s (always)
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import json
import csv
import os
import sys
import threading
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict
from pathlib import Path

# Fix Windows console encoding issues
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

# ============================================================================
# CONFIGURATION
# ============================================================================

STOPPED_SPEED_THRESHOLD = 0.5  # m/s (< 1.8 km/h considered stopped)
QUEUE_SPEED_THRESHOLD = 1.0    # m/s
FREE_FLOW_PERCENTILE = 85
MEASUREMENT_START_TIME = 0.0
MIN_TRACK_POINTS = 3
MIN_FRAMES_FOR_COUNTING = 10  # Minimum frames required to count a vehicle
MIN_DETECTION_TIME = 0.5  # Minimum detection time in seconds to count a vehicle


# ============================================================================
# GEOMETRY FUNCTIONS FOR LANE ASSIGNMENT
# ============================================================================

def point_in_polygon(x, y, poly_points):
    """
    Check if point (x, y) is inside polygon defined by poly_points.
    Uses ray casting algorithm.
    
    Args:
        x, y: Point coordinates
        poly_points: List of tuples [(x1,y1), (x2,y2), ...]
    
    Returns:
        bool: True if point is inside polygon
    """
    n = len(poly_points)
    inside = False
    
    p1x, p1y = poly_points[0]
    for i in range(1, n + 1):
        p2x, p2y = poly_points[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside


def assign_lane_to_point(img_x, img_y, lane_polygons):
    """
    Assign a lane ID to a trajectory point based on its image coordinates.
    
    Args:
        img_x, img_y: Image coordinates of the point
        lane_polygons: Dict of {lane_id: [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]}
    
    Returns:
        int or None: Lane ID if point is inside a lane polygon, None otherwise
    """
    for lane_id, polygon in lane_polygons.items():
        if point_in_polygon(img_x, img_y, polygon):
            return lane_id
    return None


# ============================================================================
# CORE ANALYSIS FUNCTIONS
# ============================================================================

def calculate_LOS_from_delay(control_delay):
    """HCM 2016 Table 19-8"""
    if control_delay <= 10: return 'A'
    elif control_delay <= 20: return 'B'
    elif control_delay <= 35: return 'C'
    elif control_delay <= 55: return 'D'
    elif control_delay <= 80: return 'E'
    else: return 'F'


@dataclass
class VehicleMetrics:
    vehicle_id: int
    lane_id: int
    first_seen: float
    last_seen: float
    total_time: float
    total_distance: float
    avg_speed: float
    max_speed: float
    stopped_delay: float      # CHANGED: Time spent stopped (for stopped delay calculation)
    control_delay: float      # CHANGED: Total control delay (actual - free flow)
    queue_time: float
    free_flow_time: float
    completed: bool
    num_stops: int
    trajectory_points: int


@dataclass
class LaneGroupMetrics:
    lane_id: int
    throughput: int
    n_vehicles_completed: int
    n_vehicles_total: int
    avg_control_delay: float      # CHANGED: Average control delay
    avg_stopped_delay: float      # CHANGED: Average stopped delay
    avg_speed: float
    measured_flow: float
    los: str
    completion_rate: float


def convert_to_python_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {convert_to_python_types(k): convert_to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_python_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_python_types(item) for item in obj)
    return obj


# ============================================================================
# CV INTERSECTION ANALYZER
# ============================================================================

class CVIntersectionAnalyzer:
    
    def __init__(self, trajectory_file: str, 
                 free_flow_speed_kmh: float = 50.0,
                 verbose: bool = True):
        """
        Args:
            trajectory_file: Path to CSV with vehicle tracks
            free_flow_speed_kmh: Assumed free-flow speed for delay calculation (default 50 km/h)
            verbose: Print analysis info to console
        """
        self.trajectory_file = trajectory_file
        self.free_flow_speed_mps = free_flow_speed_kmh / 3.6  # Convert to m/s
        self.verbose = verbose
        
        if self.verbose:
            print(f"\n{'='*80}")
            print("CV INTERSECTION ANALYSIS - MODIFIED")
            print(f"{'='*80}")
            print(f"File: {os.path.basename(trajectory_file)}")
            print(f"Free-flow speed: {free_flow_speed_kmh:.1f} km/h ({self.free_flow_speed_mps:.2f} m/s)")
            print(f"{'='*80}\n")
        
        self.df = pd.read_csv(trajectory_file)
        
        # Filter out any points with negative lane_id
        if 'lane_id' in self.df.columns:
            points_before = len(self.df)
            self.df = self.df[self.df['lane_id'] >= 0].copy()
            points_after = len(self.df)
            if points_before != points_after and self.verbose:
                print(f"[INFO] Filtered out {points_before - points_after} points with invalid lane_id")
        
        self._preprocess_data()
        
        self.vehicle_metrics: Dict[int, VehicleMetrics] = {}
        self.lane_metrics: Dict[int, LaneGroupMetrics] = {}
        self.global_metrics: Dict = {}
        
    def _preprocess_data(self):
        if self.verbose:
            print(f"  Points: {len(self.df)}")
            print(f"  Duration: {self.df['time_s'].max() - self.df['time_s'].min():.2f} s")
            print(f"  Vehicles: {self.df['vehicle_id'].nunique()}")
            print(f"  Lanes: {sorted(self.df['lane_id'].unique())}")
        
        self.df = self.df.sort_values(['vehicle_id', 'time_s']).reset_index(drop=True)
        
        self.df['dx'] = self.df.groupby('vehicle_id')['x_m'].diff()
        self.df['dy'] = self.df.groupby('vehicle_id')['y_m'].diff()
        self.df['dt'] = self.df.groupby('vehicle_id')['time_s'].diff()
        
        self.df['distance'] = np.sqrt(self.df['dx']**2 + self.df['dy']**2)
        self.df['speed_ms'] = self.df['distance'] / self.df['dt']
        self.df['speed_kmh'] = self.df['speed_ms'] * 3.6
        
        self.df['speed_ms'] = self.df['speed_ms'].fillna(0)
        self.df['speed_kmh'] = self.df['speed_kmh'].fillna(0)
        
        max_realistic_speed = 30.0
        self.df.loc[self.df['speed_ms'] > max_realistic_speed, 'speed_ms'] = np.nan
        self.df.loc[self.df['speed_kmh'] > max_realistic_speed * 3.6, 'speed_kmh'] = np.nan
        
        self.df['speed_ms'] = self.df.groupby('vehicle_id')['speed_ms'].ffill()
        self.df['speed_kmh'] = self.df.groupby('vehicle_id')['speed_kmh'].ffill()
        
        self.df['is_stopped'] = self.df['speed_ms'] < STOPPED_SPEED_THRESHOLD
        self.df['is_queued'] = self.df['speed_ms'] < QUEUE_SPEED_THRESHOLD
        
        if self.verbose:
            print(f"  Mean speed: {self.df['speed_ms'].mean():.2f} m/s ({self.df['speed_ms'].mean()*3.6:.1f} km/h)")
        
    def analyze(self, print_results=True):
        if self.verbose:
            print("\nAnalyzing...")
        self._analyze_vehicles()
        self._analyze_lane_groups()
        self._calculate_global_metrics()
        
        if print_results and self.verbose:
            self._print_results()
        
        return self.global_metrics
    
    def _analyze_vehicles(self):
        track_lengths = self.df.groupby('vehicle_id').size()
        valid_tracks = track_lengths[track_lengths >= MIN_TRACK_POINTS].index
        
        if self.verbose:
            total_vehicles = len(valid_tracks)
            print(f"  Total vehicles with >= {MIN_TRACK_POINTS} points: {total_vehicles}")
        
        vehicles_counted = 0
        vehicles_filtered_frames = 0
        vehicles_filtered_time = 0
        
        for vehicle_id in valid_tracks:
            vehicle_data = self.df[self.df['vehicle_id'] == vehicle_id].copy()
            
            # Check 1: Skip vehicles with less than MIN_FRAMES_FOR_COUNTING frames
            if len(vehicle_data) < MIN_FRAMES_FOR_COUNTING:
                vehicles_filtered_frames += 1
                continue
            
            first_seen = vehicle_data['time_s'].min()
            last_seen = vehicle_data['time_s'].max()
            total_time = last_seen - first_seen
            
            # Check 2: Skip vehicles detected for less than MIN_DETECTION_TIME seconds
            if total_time < MIN_DETECTION_TIME:
                vehicles_filtered_time += 1
                continue
            
            vehicles_counted += 1
            
            total_distance = vehicle_data['distance'].sum()
            avg_speed = vehicle_data['speed_ms'].mean()
            max_speed = vehicle_data['speed_ms'].max()
            
            # CHANGED: Calculate stopped delay (time completely stopped)
            stopped_delay = vehicle_data.loc[vehicle_data['is_stopped'], 'dt'].sum()
            
            # CHANGED: Calculate control delay (actual time - free flow time)
            # Free flow time based on distance traveled at free flow speed
            if total_distance > 0 and self.free_flow_speed_mps > 0:
                free_flow_time = total_distance / self.free_flow_speed_mps
            else:
                free_flow_time = total_time
            
            # Control delay = actual travel time - free flow time
            control_delay = max(0, total_time - free_flow_time)
            
            queue_time = vehicle_data.loc[vehicle_data['is_queued'], 'dt'].sum()
            
            is_stopped_arr = vehicle_data['is_stopped'].values
            num_stops = np.sum(np.diff(is_stopped_arr.astype(int)) > 0)
            
            # A vehicle is "completed" if it entered after measurement start time
            completed = first_seen >= MEASUREMENT_START_TIME
            
            # Assign vehicle to the lane where it spent most time
            lane_id = int(vehicle_data['lane_id'].mode()[0]) if 'lane_id' in vehicle_data.columns else 0
            
            self.vehicle_metrics[vehicle_id] = VehicleMetrics(
                vehicle_id=vehicle_id,
                lane_id=lane_id,
                first_seen=first_seen,
                last_seen=last_seen,
                total_time=total_time,
                total_distance=total_distance,
                avg_speed=avg_speed,
                max_speed=max_speed,
                stopped_delay=stopped_delay,      # Time spent stopped
                control_delay=control_delay,      # Total control delay
                queue_time=queue_time,
                free_flow_time=free_flow_time,
                completed=completed,
                num_stops=num_stops,
                trajectory_points=len(vehicle_data)
            )
        
        if self.verbose:
            print(f"  Vehicles counted (>= {MIN_FRAMES_FOR_COUNTING} frames AND >= {MIN_DETECTION_TIME}s): {vehicles_counted}")
            if vehicles_filtered_frames > 0:
                print(f"  Filtered out (< {MIN_FRAMES_FOR_COUNTING} frames): {vehicles_filtered_frames}")
            if vehicles_filtered_time > 0:
                print(f"  Filtered out (< {MIN_DETECTION_TIME}s duration): {vehicles_filtered_time}")
    
    def _analyze_lane_groups(self):
        measurement_duration = self.df['time_s'].max() - MEASUREMENT_START_TIME
        
        # Only analyze lanes with positive lane_id
        lanes = [lid for lid in self.df['lane_id'].unique() if lid >= 0]
        
        for lane_id in lanes:
            lane_vehicles = [m for m in self.vehicle_metrics.values() if m.lane_id == lane_id]
            
            if not lane_vehicles:
                continue
            
            # All unique vehicles in this lane
            n_vehicles = len(lane_vehicles)
            
            # Completed vehicles
            completed = [v for v in lane_vehicles if v.completed]
            n_completed = len(completed)
            
            # If no completed vehicles, skip this lane
            if n_completed == 0:
                continue
            
            throughput = n_vehicles
            
            # CHANGED: Calculate average control delay for ALL vehicles (not just completed)
            # This ensures we include all vehicles in the average
            avg_control_delay = np.mean([v.control_delay for v in lane_vehicles])
            
            # CHANGED: Calculate average stopped delay for ALL vehicles
            avg_stopped_delay = np.mean([v.stopped_delay for v in lane_vehicles])
            
            # Other metrics use completed vehicles
            avg_speed = np.mean([v.avg_speed for v in completed])
            
            measured_flow = (n_vehicles / measurement_duration) * 3600 if measurement_duration > 0 else 0
            
            los = calculate_LOS_from_delay(avg_control_delay)
            completion_rate = n_completed / n_vehicles if n_vehicles > 0 else 0
            
            self.lane_metrics[lane_id] = LaneGroupMetrics(
                lane_id=lane_id,
                throughput=throughput,
                n_vehicles_completed=n_completed,
                n_vehicles_total=n_vehicles,
                avg_control_delay=avg_control_delay,    # Average control delay
                avg_stopped_delay=avg_stopped_delay,    # Average stopped delay
                avg_speed=avg_speed,
                measured_flow=measured_flow,
                los=los,
                completion_rate=completion_rate
            )
    
    def _calculate_global_metrics(self):
        completed = [v for v in self.vehicle_metrics.values() if v.completed]
        all_vehicles = list(self.vehicle_metrics.values())
        
        measurement_duration = self.df['time_s'].max() - MEASUREMENT_START_TIME
        
        # Use ALL vehicles for delay calculations (not just completed)
        avg_control_delay = np.mean([v.control_delay for v in all_vehicles]) if all_vehicles else 0
        avg_stopped_delay = np.mean([v.stopped_delay for v in all_vehicles]) if all_vehicles else 0
        
        self.global_metrics = {
            'n_vehicles_total': len(all_vehicles),
            'n_vehicles_completed': len(completed),
            'completion_rate': len(completed) / len(all_vehicles) if all_vehicles else 0,
            'avg_control_delay': avg_control_delay,
            'avg_stopped_delay': avg_stopped_delay,
            'avg_speed_ms': np.mean([v.avg_speed for v in completed]) if completed else 0,
            'avg_speed_kmh': np.mean([v.avg_speed for v in completed]) * 3.6 if completed else 0,
            'total_throughput': len(completed),
            'throughput_per_hour': (len(completed) / measurement_duration * 3600) if measurement_duration > 0 else 0,
            'measurement_duration': measurement_duration,
            'overall_los': calculate_LOS_from_delay(avg_control_delay),
            'lane_metrics': {lid: vars(lm) for lid, lm in self.lane_metrics.items()}
        }
    
    def _print_results(self):
        print("\n" + "="*80)
        print("GLOBAL METRICS")
        print("="*80)
        print(f"Total vehicles: {self.global_metrics['n_vehicles_total']}")
        print(f"Completed vehicles: {self.global_metrics['n_vehicles_completed']} ({self.global_metrics['completion_rate']*100:.1f}%)")
        print(f"Average control delay: {self.global_metrics['avg_control_delay']:.2f} s/veh")
        print(f"Average stopped delay: {self.global_metrics['avg_stopped_delay']:.2f} s/veh")
        print(f"Delay ratio (stopped/control): {(self.global_metrics['avg_stopped_delay']/self.global_metrics['avg_control_delay']*100):.1f}%")
        print(f"Average speed: {self.global_metrics['avg_speed_ms']:.2f} m/s ({self.global_metrics['avg_speed_kmh']:.1f} km/h)")
        print(f"Throughput: {self.global_metrics['throughput_per_hour']:.1f} veh/h")
        print(f"Overall LOS: {self.global_metrics['overall_los']}")
        
        print("\n" + "="*80)
        print("LANE-BY-LANE METRICS")
        print("="*80)
        
        for lane_id, metrics in self.lane_metrics.items():
            delay_ratio = (metrics.avg_stopped_delay / metrics.avg_control_delay * 100) if metrics.avg_control_delay > 0 else 0
            print(f"\nLane {lane_id}:")
            print(f"  Vehicles: {metrics.n_vehicles_completed}/{metrics.n_vehicles_total} ({metrics.completion_rate*100:.1f}%)")
            print(f"  Avg control delay: {metrics.avg_control_delay:.2f} s/veh")
            print(f"  Avg stopped delay: {metrics.avg_stopped_delay:.2f} s/veh ({delay_ratio:.1f}% of control delay)")
            print(f"  Avg speed: {metrics.avg_speed:.2f} m/s ({metrics.avg_speed*3.6:.1f} km/h)")
            print(f"  Flow: {metrics.measured_flow:.1f} veh/h")
            print(f"  LOS: {metrics.los}")
    
    def export_csv(self, output_path: str, include_delay: bool = False, 
                   include_los: bool = False):
        """
        Export minute-by-minute lane metrics to CSV
        
        Output format:
        Minute, lane_id, n_vehicles, avg_speed_kmh [, optional metrics]
        
        Where:
        - Minute: Time interval (1, 2, 3, ...) - starts from 1
        - lane_id: Lane identifier
        - n_vehicles: NEW unique vehicles entering in this minute in THIS LANE (not counted in previous minutes in this lane)
        - avg_speed_kmh: Average speed of vehicles in this lane during this minute
        
        Optional metrics (if enabled):
        - avg_delay_s: Average control delay (actual time - free flow time)
        - los: Level of Service (A-F) based on delay
        """
        
        # Get total duration in seconds
        total_duration = self.df['time_s'].max() - self.df['time_s'].min()
        num_minutes = int(np.ceil(total_duration / 60))
        
        if self.verbose:
            print(f"\nGenerating minute-by-minute analysis...")
            print(f"  Total duration: {total_duration:.1f} seconds ({num_minutes} minutes)")
        
        # Get all lane IDs
        lane_ids = sorted([lid for lid in self.df['lane_id'].unique() if lid >= 0])
        
        # Track which vehicles we've counted PER LANE (separate tracking for each lane)
        counted_vehicles_per_lane = {lane_id: set() for lane_id in lane_ids}
        
        rows = []
        
        for minute_idx in range(num_minutes):
            # Minute number starts from 1 instead of 0
            minute = minute_idx + 1
            
            # Time range for this minute
            start_time = minute_idx * 60
            end_time = (minute_idx + 1) * 60
            
            # Get data for this minute
            minute_data = self.df[(self.df['time_s'] >= start_time) & 
                                  (self.df['time_s'] < end_time)].copy()
            
            if len(minute_data) == 0:
                # No data in this minute, still output 0s for all lanes
                for lane_id in lane_ids:
                    rows.append({
                        'Minute': minute,
                        'lane_id': lane_id,
                        'n_vehicles': 0,
                        'avg_speed_kmh': 0.0
                    })
                continue
            
            # Process each lane
            for lane_id in lane_ids:
                lane_minute_data = minute_data[minute_data['lane_id'] == lane_id]
                
                if len(lane_minute_data) == 0:
                    # No vehicles in this lane during this minute
                    rows.append({
                        'Minute': minute,
                        'lane_id': lane_id,
                        'n_vehicles': 0,
                        'avg_speed_kmh': 0.0
                    })
                    continue
                
                # Get unique vehicles in this lane during this minute
                vehicles_this_minute = set(lane_minute_data['vehicle_id'].unique())
                
                # NEW vehicles = vehicles not counted before IN THIS SPECIFIC LANE
                new_vehicles = vehicles_this_minute - counted_vehicles_per_lane[lane_id]
                n_new_vehicles = len(new_vehicles)
                
                # Add these vehicles to the counted set FOR THIS LANE
                counted_vehicles_per_lane[lane_id].update(new_vehicles)
                
                # Calculate average speed for vehicles in this lane this minute
                if 'speed_kmh' in lane_minute_data.columns:
                    avg_speed = lane_minute_data['speed_kmh'].mean()
                else:
                    # Calculate from speed_ms if available
                    avg_speed = lane_minute_data['speed_ms'].mean() * 3.6 if 'speed_ms' in lane_minute_data.columns else 0.0
                
                # Prepare row data
                row_data = {
                    'Minute': minute,
                    'lane_id': lane_id,
                    'n_vehicles': n_new_vehicles,
                    'avg_speed_kmh': round(avg_speed, 2)
                }
                
                # Calculate optional metrics if requested
                if include_delay or include_los:
                    # Get data for NEW vehicles only (not previously counted)
                    new_vehicles_data = lane_minute_data[lane_minute_data['vehicle_id'].isin(new_vehicles)]
                    
                    if len(new_vehicles_data) > 0 and n_new_vehicles > 0:
                        # Calculate control delay for new vehicles
                        delays = []
                        for vid in new_vehicles:
                            veh_data = new_vehicles_data[new_vehicles_data['vehicle_id'] == vid]
                            if len(veh_data) > 1:
                                # Calculate distance traveled
                                if 'distance' in veh_data.columns:
                                    distance = veh_data['distance'].sum()
                                else:
                                    coords = veh_data[['x_m', 'y_m']].values
                                    if len(coords) > 1:
                                        distance = np.sum(np.sqrt(np.sum(np.diff(coords, axis=0)**2, axis=1)))
                                    else:
                                        distance = 0
                                
                                # Actual time
                                actual_time = veh_data['time_s'].max() - veh_data['time_s'].min()
                                
                                # Free flow time
                                if distance > 0 and self.free_flow_speed_mps > 0:
                                    free_flow_time = distance / self.free_flow_speed_mps
                                    delay = max(0, actual_time - free_flow_time)
                                    delays.append(delay)
                        
                        avg_delay = np.mean(delays) if delays else 0.0
                        if include_delay:
                            row_data['avg_delay_s'] = round(avg_delay, 2)
                        
                        if include_los:
                            row_data['los'] = calculate_LOS_from_delay(avg_delay)
                    else:
                        # No new vehicles, set optional metrics to 0
                        if include_delay:
                            row_data['avg_delay_s'] = 0.0
                        if include_los:
                            row_data['los'] = 'A'
                
                rows.append(row_data)
        
        # Create DataFrame and save
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        
        if self.verbose:
            print(f"\nExported minute-by-minute metrics to: {output_path}")
            print("\nCSV Columns:")
            print("  • Minute: Time interval (1, 2, 3, ...) - starts from 1")
            print("  • lane_id: Lane identifier")
            print("  • n_vehicles: NEW unique vehicles entering this minute in this lane")
            print("  • avg_speed_kmh: Average speed in this lane during this minute")
            
            if include_delay:
                print("  • avg_delay_s: Average control delay (optional)")
            if include_los:
                print("  • los: Level of Service A-F (optional)")
            
            print(f"\nTotal rows: {len(df)}")
            
            # Show per-lane counts
            for lane_id in lane_ids:
                total = len(counted_vehicles_per_lane[lane_id])
                print(f"  Lane {lane_id}: {total} unique vehicles counted")


# ============================================================================
# TAB 1: LANE ASSIGNMENT
# ============================================================================

class LaneAssignmentTab:
    
    def __init__(self, parent, notebook):
        self.parent = parent
        self.notebook = notebook
        
        # Create main frame
        self.frame = ttk.Frame(notebook)
        notebook.add(self.frame, text="1. Lane Assignment")
        
        # Variables
        self.trajectory_path = tk.StringVar()
        self.lanes_path = tk.StringVar()
        self.output_path = tk.StringVar()
        
        self.create_widgets()
    
    def create_widgets(self):
        # Title
        title_frame = ttk.Frame(self.frame)
        title_frame.pack(fill=tk.X, padx=20, pady=10)
        
        title_label = ttk.Label(title_frame, text="Step 1: Assign Lane IDs to Trajectories", 
                                font=('Helvetica', 14, 'bold'))
        title_label.pack()
        
        desc_label = ttk.Label(title_frame, 
                               text="Load trajectory CSV and lane polygon CSV to assign lane IDs to each trajectory point",
                               foreground="gray")
        desc_label.pack()
        
        # Separator
        ttk.Separator(self.frame, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=20, pady=10)
        
        # Input files section
        input_frame = ttk.LabelFrame(self.frame, text="Input Files", padding=15)
        input_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Trajectory file
        traj_frame = ttk.Frame(input_frame)
        traj_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(traj_frame, text="Trajectory CSV:", width=20).pack(side=tk.LEFT)
        ttk.Entry(traj_frame, textvariable=self.trajectory_path, width=50).pack(side=tk.LEFT, padx=5)
        ttk.Button(traj_frame, text="Browse...", command=self.browse_trajectory).pack(side=tk.LEFT)
        
        ttk.Label(traj_frame, text="(frame, time_s, vehicle_id, x_m, y_m, img_x, img_y)", 
                 foreground="gray", font=('Helvetica', 8)).pack(side=tk.LEFT, padx=10)
        
        # Lanes file
        lanes_frame = ttk.Frame(input_frame)
        lanes_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(lanes_frame, text="Lanes CSV:", width=20).pack(side=tk.LEFT)
        ttk.Entry(lanes_frame, textvariable=self.lanes_path, width=50).pack(side=tk.LEFT, padx=5)
        ttk.Button(lanes_frame, text="Browse...", command=self.browse_lanes).pack(side=tk.LEFT)
        
        ttk.Label(lanes_frame, text="(lane_id, p1_x, p1_y, p2_x, p2_y, p3_x, p3_y, p4_x, p4_y)", 
                 foreground="gray", font=('Helvetica', 8)).pack(side=tk.LEFT, padx=10)
        
        # Output section
        output_frame = ttk.LabelFrame(self.frame, text="Output", padding=15)
        output_frame.pack(fill=tk.X, padx=20, pady=10)
        
        out_frame = ttk.Frame(output_frame)
        out_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(out_frame, text="Output CSV:", width=20).pack(side=tk.LEFT)
        ttk.Entry(out_frame, textvariable=self.output_path, width=50).pack(side=tk.LEFT, padx=5)
        ttk.Button(out_frame, text="Browse...", command=self.browse_output).pack(side=tk.LEFT)
        
        # Buttons
        button_frame = ttk.Frame(self.frame)
        button_frame.pack(pady=20)
        
        self.assign_button = ttk.Button(button_frame, text="▶ Assign Lane IDs", 
                                        command=self.assign_lanes, style='Accent.TButton')
        self.assign_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="Clear", command=self.clear_all).pack(side=tk.LEFT, padx=5)
        
        # Progress
        self.progress_label = ttk.Label(self.frame, text="", foreground="blue")
        self.progress_label.pack(pady=5)
        
        # Status bar
        self.status_bar = ttk.Label(self.frame, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def browse_trajectory(self):
        filename = filedialog.askopenfilename(
            title="Select Trajectory CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.trajectory_path.set(filename)
            
            # Auto-suggest output path
            if not self.output_path.get():
                base_dir = os.path.dirname(filename)
                output = os.path.join(base_dir, "vehicle_tracks_with_lanes.csv")
                self.output_path.set(output)
    
    def browse_lanes(self):
        filename = filedialog.askopenfilename(
            title="Select Lanes CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.lanes_path.set(filename)
    
    def browse_output(self):
        filename = filedialog.asksaveasfilename(
            title="Save Output CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.output_path.set(filename)
    
    def assign_lanes(self):
        traj_file = self.trajectory_path.get()
        lanes_file = self.lanes_path.get()
        output_file = self.output_path.get()
        
        if not traj_file or not os.path.exists(traj_file):
            messagebox.showerror("Error", "Please select a valid trajectory CSV file")
            return
        
        if not lanes_file or not os.path.exists(lanes_file):
            messagebox.showerror("Error", "Please select a valid lanes CSV file")
            return
        
        if not output_file:
            messagebox.showerror("Error", "Please specify an output file path")
            return
        
        # Disable button
        self.assign_button.config(state=tk.DISABLED)
        self.status_bar.config(text="Assigning lanes...")
        self.progress_label.config(text="Processing... Please wait...")
        self.parent.update()
        
        # Run in thread
        thread = threading.Thread(target=self.assignment_thread, 
                                 args=(traj_file, lanes_file, output_file))
        thread.daemon = True
        thread.start()
    
    def assignment_thread(self, traj_file, lanes_file, output_file):
        try:
            # Load lanes
            lanes_df = pd.read_csv(lanes_file)
            lane_polygons = {}
            
            for _, row in lanes_df.iterrows():
                lane_id = int(row['lane_id'])
                polygon = [
                    (row['p1_x'], row['p1_y']),
                    (row['p2_x'], row['p2_y']),
                    (row['p3_x'], row['p3_y']),
                    (row['p4_x'], row['p4_y'])
                ]
                lane_polygons[lane_id] = polygon
            
            # Load trajectories
            traj_df = pd.read_csv(traj_file)
            
            # Assign lanes
            lane_ids = []
            total = len(traj_df)
            
            for idx, row in traj_df.iterrows():
                img_x = row['img_x']
                img_y = row['img_y']
                
                lane_id = assign_lane_to_point(img_x, img_y, lane_polygons)
                lane_ids.append(lane_id if lane_id is not None else -1)
                
                # Update progress every 1000 rows
                if idx % 1000 == 0:
                    progress = (idx / total) * 100
                    msg = f"Processing... {progress:.1f}% ({idx}/{total})"
                    self.parent.after(0, self.progress_label.config, {'text': msg})
            
            # Add lane_id column
            traj_df['lane_id'] = lane_ids
            
            # Save
            traj_df.to_csv(output_file, index=False)
            
            # Count statistics
            n_assigned = sum(1 for lid in lane_ids if lid >= 0)
            n_unassigned = len(lane_ids) - n_assigned
            
            # Update GUI
            self.parent.after(0, self.assignment_complete, output_file, n_assigned, n_unassigned)
            
        except Exception as e:
            self.parent.after(0, self.assignment_error, str(e))
    
    def assignment_complete(self, output_file, n_assigned, n_unassigned):
        self.progress_label.config(text="Complete!", foreground="green")
        self.status_bar.config(text="Assignment complete")
        self.assign_button.config(state=tk.NORMAL)
        
        summary = f"Lane assignment complete!\n\n"
        summary += f"Output: {os.path.basename(output_file)}\n\n"
        summary += f"Points assigned: {n_assigned:,}\n"
        summary += f"Points unassigned: {n_unassigned:,}\n\n"
        summary += "Now proceed to Tab 2 for performance analysis."
        
        messagebox.showinfo("Success", summary)
    
    def assignment_error(self, error_msg):
        self.progress_label.config(text="Failed", foreground="red")
        self.status_bar.config(text="Error")
        self.assign_button.config(state=tk.NORMAL)
        
        messagebox.showerror("Error", f"Lane assignment failed:\n\n{error_msg}")
    
    def clear_all(self):
        self.trajectory_path.set("")
        self.lanes_path.set("")
        self.output_path.set("")
        self.progress_label.config(text="", foreground="black")
        self.status_bar.config(text="Ready")


# ============================================================================
# TAB 2: PERFORMANCE ANALYSIS
# ============================================================================

class PerformanceAnalysisTab:
    
    def __init__(self, parent, notebook):
        self.parent = parent
        self.notebook = notebook
        
        # Create main frame
        self.frame = ttk.Frame(notebook)
        notebook.add(self.frame, text="2. Performance Analysis")
        
        # Variables
        self.csv_path = tk.StringVar()
        self.free_flow_speed = tk.StringVar(value="50")  # Default 50 km/h
        
        # Lane detection
        self.num_lanes = 0
        self.lane_ids = []
        
        self.create_widgets()
    
    def create_widgets(self):
        # Title
        title_frame = ttk.Frame(self.frame)
        title_frame.pack(fill=tk.X, padx=20, pady=10)
        
        title_label = ttk.Label(title_frame, text="Step 2: Analyze Intersection Performance", 
                                font=('Helvetica', 14, 'bold'))
        title_label.pack()
        
        desc_label = ttk.Label(title_frame, 
                               text="Calculate lane-by-lane performance metrics from trajectory data with lane assignments",
                               foreground="gray")
        desc_label.pack()
        
        # Separator
        ttk.Separator(self.frame, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=20, pady=10)
        
        # Input section
        input_frame = ttk.LabelFrame(self.frame, text="Input File", padding=15)
        input_frame.pack(fill=tk.X, padx=20, pady=10)
        
        csv_frame = ttk.Frame(input_frame)
        csv_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(csv_frame, text="Trajectory CSV:", width=20).pack(side=tk.LEFT)
        ttk.Entry(csv_frame, textvariable=self.csv_path, width=50).pack(side=tk.LEFT, padx=5)
        ttk.Button(csv_frame, text="Browse...", command=self.browse_csv).pack(side=tk.LEFT)
        
        ttk.Label(csv_frame, text="(with lane_id column)", 
                 foreground="gray", font=('Helvetica', 8)).pack(side=tk.LEFT, padx=10)
        
        # Info label
        self.info_label = ttk.Label(input_frame, text="No file loaded yet", 
                                    foreground="gray", font=('Helvetica', 9))
        self.info_label.pack(pady=5)
        
        # Parameters section
        param_frame = ttk.LabelFrame(self.frame, text="Analysis Parameters", padding=15)
        param_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Free flow speed
        ff_frame = ttk.Frame(param_frame)
        ff_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(ff_frame, text="Free-Flow Speed:", width=20).pack(side=tk.LEFT)
        ttk.Entry(ff_frame, textvariable=self.free_flow_speed, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Label(ff_frame, text="km/h", foreground="gray").pack(side=tk.LEFT)
        ttk.Label(ff_frame, text="(Used to calculate control delay)", 
                 foreground="gray", font=('Helvetica', 8)).pack(side=tk.LEFT, padx=10)
        
        # Info text
        info_text = """
ℹ️  Delay Calculation Method:
   • Control Delay = Actual Travel Time - Free-Flow Travel Time
   • Stopped Delay = Time spent completely stopped (speed < 1.8 km/h)
   • Control delay includes stopped delay plus slow-moving time
   • Always: avg_delay_s ≥ avg_stopped_delay_s
        """
        
        info_label = ttk.Label(param_frame, text=info_text, 
                              foreground="blue", font=('Helvetica', 9),
                              justify=tk.LEFT)
        info_label.pack(pady=10, padx=10)
        
        # Buttons
        button_frame = ttk.Frame(self.frame)
        button_frame.pack(pady=20)
        
        self.run_button = ttk.Button(button_frame, text="📊 Run Performance Analysis", 
                                     command=self.run_analysis, style='Accent.TButton',
                                     state=tk.DISABLED)
        self.run_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="Clear", command=self.clear_all).pack(side=tk.LEFT, padx=5)
        
        # Progress
        self.progress_label = ttk.Label(self.frame, text="", foreground="blue")
        self.progress_label.pack(pady=5)
        
        # Status bar
        self.status_bar = ttk.Label(self.frame, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def browse_csv(self):
        filename = filedialog.askopenfilename(
            title="Select Trajectory CSV with Lane IDs",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.csv_path.set(filename)
            self.load_file_info(filename)
    
    def load_file_info(self, filename):
        try:
            df = pd.read_csv(filename)
            
            # Check for required columns
            required = ['vehicle_id', 'lane_id', 'time_s']
            missing = [col for col in required if col not in df.columns]
            
            if missing:
                self.info_label.config(
                    text=f"❌ Missing columns: {', '.join(missing)}", 
                    foreground="red"
                )
                self.run_button.config(state=tk.DISABLED)
                return
            
            # Get lane info
            self.lane_ids = sorted([lid for lid in df['lane_id'].unique() if lid >= 0])
            self.num_lanes = len(self.lane_ids)
            
            n_points = len(df)
            n_vehicles = df['vehicle_id'].nunique()
            duration = df['time_s'].max() - df['time_s'].min()
            
            info_text = f"✓ Loaded: {n_points:,} points, {n_vehicles} vehicles, {duration:.1f}s\n"
            info_text += f"✓ Detected {self.num_lanes} lanes: {self.lane_ids}"
            
            self.info_label.config(text=info_text, foreground="darkgreen")
            self.run_button.config(state=tk.NORMAL)
            
        except Exception as e:
            self.info_label.config(text=f"❌ Error loading file: {str(e)}", foreground="red")
            self.run_button.config(state=tk.DISABLED)
    
    def run_analysis(self):
        csv_file = self.csv_path.get()
        
        if not csv_file or not os.path.exists(csv_file):
            messagebox.showerror("Error", "Please select a valid CSV file")
            return
        
        if self.num_lanes == 0:
            messagebox.showerror("Error", "Please load file and detect lanes first")
            return
        
        # Get free flow speed
        try:
            ff_speed = float(self.free_flow_speed.get())
            if ff_speed <= 0:
                messagebox.showerror("Error", "Free-flow speed must be > 0")
                return
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number for free-flow speed")
            return
        
        # Disable run button
        self.run_button.config(state=tk.DISABLED)
        self.status_bar.config(text="Running analysis...")
        self.progress_label.config(text="Analyzing... Please wait...")
        self.parent.update()
        
        # Run in thread
        thread = threading.Thread(target=self.analysis_thread, args=(csv_file, ff_speed))
        thread.daemon = True
        thread.start()
    
    def analysis_thread(self, csv_file, ff_speed):
        """Run analysis in separate thread"""
        try:
            # Run analysis
            analyzer = CVIntersectionAnalyzer(
                trajectory_file=csv_file,
                free_flow_speed_kmh=ff_speed,
                verbose=False
            )
            
            metrics = analyzer.analyze(print_results=False)
            
            # Export results
            output_dir = os.path.dirname(csv_file)
            csv_path = os.path.join(output_dir, "cv_lane_metrics.csv")
            analyzer.export_csv(csv_path)
            
            json_path = os.path.join(output_dir, "cv_metrics.json")
            with open(json_path, 'w') as f:
                metrics_converted = convert_to_python_types(metrics)
                json.dump(metrics_converted, f, indent=2)
            
            # Update GUI
            self.parent.after(0, self.analysis_complete, csv_path, json_path, metrics)
            
        except Exception as e:
            self.parent.after(0, self.analysis_error, str(e))
    
    def analysis_complete(self, csv_path, json_path, metrics):
        """Called when analysis completes"""
        self.progress_label.config(text="Analysis complete!", foreground="green")
        self.status_bar.config(text="Analysis complete")
        self.run_button.config(state=tk.NORMAL)
        
        # Show summary
        summary = f"Analysis Complete!\n\n"
        summary += f"Results saved to:\n"
        summary += f"  • {os.path.basename(csv_path)}\n"
        summary += f"  • {os.path.basename(json_path)}\n\n"
        summary += f"Summary:\n"
        summary += f"  • Vehicles analyzed: {metrics['n_vehicles_completed']}/{metrics['n_vehicles_total']}\n"
        summary += f"  • Avg Control Delay: {metrics['avg_control_delay']:.2f} s/veh\n"
        summary += f"  • Avg Stopped Delay: {metrics['avg_stopped_delay']:.2f} s/veh\n"
        summary += f"  • Overall LOS: {metrics['overall_los']}\n"
        summary += f"  • Throughput: {metrics['throughput_per_hour']:.1f} veh/h\n\n"
        summary += "Open CSV file to see detailed lane-by-lane results."
        
        messagebox.showinfo("Success", summary)
        
        # Ask to open folder
        response = messagebox.askyesno("Open Results", "Would you like to open the results folder?")
        if response:
            output_dir = os.path.dirname(csv_path)
            if sys.platform == 'win32':
                os.startfile(output_dir)
            elif sys.platform == 'darwin':
                os.system(f'open "{output_dir}"')
            else:
                os.system(f'xdg-open "{output_dir}"')
    
    def analysis_error(self, error_msg):
        """Called when analysis fails"""
        self.progress_label.config(text="Analysis failed", foreground="red")
        self.status_bar.config(text="Error")
        self.run_button.config(state=tk.NORMAL)
        
        messagebox.showerror("Analysis Error", f"Analysis failed:\n\n{error_msg}")
    
    def clear_all(self):
        """Clear all inputs"""
        self.csv_path.set("")
        self.free_flow_speed.set("50")
        self.num_lanes = 0
        self.lane_ids = []
        
        self.info_label.config(text="No file loaded yet", foreground="gray")
        
        self.run_button.config(state=tk.DISABLED)
        self.progress_label.config(text="")
        self.status_bar.config(text="Ready")


# ============================================================================
# MAIN APPLICATION
# ============================================================================

class CVIntersectionApp:
    
    def __init__(self, root):
        self.root = root
        self.root.title("CV Intersection Analysis - MODIFIED (Correct Delay Calculation)")
        self.root.geometry("900x750")
        
        # Configure style
        style = ttk.Style()
        
        # Try to use a nice theme
        available_themes = style.theme_names()
        if 'clam' in available_themes:
            style.theme_use('clam')
        elif 'alt' in available_themes:
            style.theme_use('alt')
        
        # Define accent button style
        style.configure('Accent.TButton', font=('Helvetica', 10, 'bold'))
        
        # Create notebook (tabs)
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create tabs
        self.lane_tab = LaneAssignmentTab(root, self.notebook)
        self.analysis_tab = PerformanceAnalysisTab(root, self.notebook)
        
        # Status bar
        self.status_bar = ttk.Label(root, text="Ready - MODIFIED VERSION with correct delay calculations", 
                                    relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main function"""
    root = tk.Tk()
    app = CVIntersectionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()