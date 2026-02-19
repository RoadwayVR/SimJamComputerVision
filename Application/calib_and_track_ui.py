# calib_and_track_ui.py
# Source-only app:
# - UI for selecting video/weights/output + parameters
# - Interactive calibration (pick 4-point polygon + real-world lengths)
# - Interactive lane definition (draw multiple 4-point lanes) - saved for post-processing
# - YOLO detection + ByteTrack tracking inside ROI
# - Exports:
#   1) vehicles-result.mp4 (with lane overlays)
#   2) vehicles.csv (summary per track)
#   3) vehicle_tracks_xy.csv (per-frame positions WITH img_x, img_y for lane post-processing)
#   4) lanes.csv (lane polygon coordinates for post-processing)
#
# Lane assignment is done in SEPARATE post-processing step (lane_postprocess.py) for speed.

import os, sys, csv, inspect, logging, traceback
from collections import defaultdict, deque, Counter

import cv2
import numpy as np

# UI
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter import simpledialog

# ML / CV
from ultralytics import YOLO
import supervision as sv


# ============================
# Logging
# ============================
def setup_logger(log_path):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout)
        ]
    )


# ============================
# Colors (BGR)
# ============================
GREY   = (128, 128, 128)
YELLOW = (0, 255, 255)
WHITE  = (255, 255, 255)
BLACK  = (0, 0, 0)

LANE_COLORS = [
    (255, 0, 0),    # Blue
    (0, 255, 0),    # Green
    (0, 0, 255),    # Red
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Magenta
    (0, 165, 255),  # Orange
    (128, 0, 128),  # Purple
    (0, 128, 128),  # Olive
]


# ============================
# ----- CALIBRATION PART -----
# ============================
def grab_four_frames(cap):
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = [0, max(0, total // 4), max(0, total // 2), max(0, 3 * total // 4)]
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, f = cap.read()
        frames.append(f if ok else None)
    return frames, idxs


def choose_frame(frames, rescale=960):
    thumbs = []
    for f in frames:
        if f is None:
            thumbs.append(None); continue
        h, w = f.shape[:2]
        s = rescale / w
        thumbs.append(cv2.resize(f, (int(w * s), int(h * s))))

    labeled = []
    for i, t in enumerate(thumbs):
        if t is None:
            labeled.append(None); continue
        img = t.copy()
        cv2.rectangle(img, (8, 8), (60, 48), BLACK, -1)
        cv2.putText(img, str(i + 1), (18, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, 2)
        labeled.append(img)

    w = max(x.shape[1] for x in labeled if x is not None)
    h = max(x.shape[0] for x in labeled if x is not None)

    def pad(x):
        if x is None:
            return np.zeros((h, w, 3), dtype=np.uint8)
        ph, pw = h - x.shape[0], w - x.shape[1]
        return cv2.copyMakeBorder(x, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=(30, 30, 30))

    tl, tr, bl, br = map(pad, labeled)
    grid = np.vstack([np.hstack([tl, tr]), np.hstack([bl, br])])

    title = "Pick a frame (1/2/3/4, q quit)"
    cv2.imshow(title, grid)
    while True:
        k = cv2.waitKey(0) & 0xFF
        if k in (ord('1'), ord('2'), ord('3'), ord('4')):
            cv2.destroyWindow(title)
            return k - ord('1')
        if k in (ord('q'), 27):
            cv2.destroyAllWindows()
            sys.exit(0)


def collect_polygon(frame, display_w=1280):
    win = "Draw ROI polygon (L-click add 4 pts, R-click undo, Enter confirm, r reset, q quit)"
    orig_h, orig_w = frame.shape[:2]
    scale = float(display_w) / float(orig_w)
    display_h = int(round(orig_h * scale))
    disp_frame = cv2.resize(frame, (display_w, display_h), interpolation=cv2.INTER_AREA)
    pts_orig = []

    def on_mouse(event, x, y, flags, param):
        nonlocal pts_orig
        if event == cv2.EVENT_LBUTTONDOWN and len(pts_orig) < 4:
            ox = int(round(x / scale))
            oy = int(round(y / scale))
            ox = max(0, min(orig_w - 1, ox))
            oy = max(0, min(orig_h - 1, oy))
            pts_orig.append((ox, oy))
        elif event == cv2.EVENT_RBUTTONDOWN and pts_orig:
            pts_orig.pop()

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, display_w, display_h)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        vis = disp_frame.copy()
        pts_disp = [(int(round(px * scale)), int(round(py * scale))) for px, py in pts_orig]
        if len(pts_disp) >= 3:
            overlay = vis.copy()
            cv2.fillPoly(overlay, [np.array(pts_disp, np.int32)], GREY)
            cv2.addWeighted(overlay, 0.45, vis, 0.55, 0, vis)
        if len(pts_disp) >= 2:
            closed = (len(pts_disp) == 4)
            cv2.polylines(vis, [np.array(pts_disp, np.int32)], closed, YELLOW, 2)
        for i, p in enumerate(pts_disp):
            cv2.circle(vis, p, 6, YELLOW, -1)
            cv2.putText(vis, str(i + 1), (p[0] + 6, p[1] - 6), cv2.FONT_HERSHEY_DUPLEX, 0.7, BLACK, 2, cv2.LINE_AA)
        cv2.imshow(win, vis)
        k = cv2.waitKey(20) & 0xFF
        if k in (13, 10) and len(pts_orig) == 4:
            cv2.destroyWindow(win)
            return np.array(pts_orig, dtype=np.float32)
        elif k == ord('r'):
            pts_orig = []
        elif k in (ord('q'), 27):
            cv2.destroyAllWindows()
            sys.exit(0)


# ============================
# ----- LANE DRAWING -----
# ============================
def collect_lanes(frame, roi_polygon, display_w=1280):
    """Draw multiple lane polygons (4 points each, clockwise)."""
    win = "Draw Lanes (L-click 4 pts clockwise, Enter=next lane, D=done, R=reset, Q=quit)"
    orig_h, orig_w = frame.shape[:2]
    scale = float(display_w) / float(orig_w)
    display_h = int(round(orig_h * scale))
    disp_frame = cv2.resize(frame, (display_w, display_h), interpolation=cv2.INTER_AREA)
    lanes_orig = []
    current_pts_orig = []

    def on_mouse(event, x, y, flags, param):
        nonlocal current_pts_orig
        if event == cv2.EVENT_LBUTTONDOWN and len(current_pts_orig) < 4:
            ox = int(round(x / scale))
            oy = int(round(y / scale))
            ox = max(0, min(orig_w - 1, ox))
            oy = max(0, min(orig_h - 1, oy))
            current_pts_orig.append((ox, oy))
        elif event == cv2.EVENT_RBUTTONDOWN and current_pts_orig:
            current_pts_orig.pop()

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, display_w, display_h)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        vis = disp_frame.copy()
        roi_disp = [(int(round(px * scale)), int(round(py * scale))) for px, py in roi_polygon]
        cv2.polylines(vis, [np.array(roi_disp, np.int32)], True, YELLOW, 2)

        for lane_idx, lane in enumerate(lanes_orig):
            color = LANE_COLORS[lane_idx % len(LANE_COLORS)]
            lane_disp = [(int(round(px * scale)), int(round(py * scale))) for px, py in lane]
            overlay = vis.copy()
            cv2.fillPoly(overlay, [np.array(lane_disp, np.int32)], color)
            cv2.addWeighted(overlay, 0.3, vis, 0.7, 0, vis)
            cv2.polylines(vis, [np.array(lane_disp, np.int32)], True, color, 2)
            cx = int(np.mean([p[0] for p in lane_disp]))
            cy = int(np.mean([p[1] for p in lane_disp]))
            cv2.putText(vis, f"Lane {lane_idx + 1}", (cx - 30, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)

        current_lane_idx = len(lanes_orig)
        current_color = LANE_COLORS[current_lane_idx % len(LANE_COLORS)]
        pts_disp = [(int(round(px * scale)), int(round(py * scale))) for px, py in current_pts_orig]
        if len(pts_disp) >= 3:
            overlay = vis.copy()
            cv2.fillPoly(overlay, [np.array(pts_disp, np.int32)], current_color)
            cv2.addWeighted(overlay, 0.3, vis, 0.7, 0, vis)
        if len(pts_disp) >= 2:
            closed = (len(pts_disp) == 4)
            cv2.polylines(vis, [np.array(pts_disp, np.int32)], closed, current_color, 2)
        for i, p in enumerate(pts_disp):
            cv2.circle(vis, p, 6, current_color, -1)
            cv2.putText(vis, str(i + 1), (p[0] + 6, p[1] - 6), cv2.FONT_HERSHEY_DUPLEX, 0.6, BLACK, 2, cv2.LINE_AA)

        # Draw status text at top-left with better font
        status = f"Drawing Lane {current_lane_idx + 1} | Points: {len(current_pts_orig)}/4 | Completed: {len(lanes_orig)} lanes"
        # Black background for better readability
        cv2.rectangle(vis, (5, 5), (650, 45), BLACK, -1)
        # Use FONT_HERSHEY_DUPLEX for cleaner look
        cv2.putText(vis, status, (15, 32), cv2.FONT_HERSHEY_DUPLEX, 0.7, WHITE, 1, cv2.LINE_AA)
        
        # Draw instructions at top-right
        instr = "Enter=confirm | D=done | R=reset | Q=quit"
        # Get text size to position from right
        (text_w, text_h), _ = cv2.getTextSize(instr, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
        x_pos = display_w - text_w - 15
        cv2.rectangle(vis, (x_pos - 10, 5), (display_w - 5, 45), BLACK, -1)
        cv2.putText(vis, instr, (x_pos, 32), cv2.FONT_HERSHEY_DUPLEX, 0.6, YELLOW, 1, cv2.LINE_AA)

        cv2.imshow(win, vis)
        k = cv2.waitKey(20) & 0xFF

        if k in (13, 10) and len(current_pts_orig) == 4:
            new_lane = np.array(current_pts_orig, dtype=np.float32)
            is_valid, overlap_msg = check_lane_overlap(new_lane, lanes_orig)
            if is_valid:
                lanes_orig.append(new_lane)
                logging.info("Lane %d defined: %s", len(lanes_orig), current_pts_orig)
                current_pts_orig = []
            else:
                logging.warning("Lane overlap detected: %s", overlap_msg)
                cv2.putText(vis, f"OVERLAP! {overlap_msg}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow(win, vis)
                cv2.waitKey(1000)
        elif k == ord('r') or k == ord('R'):
            current_pts_orig = []
        elif k == ord('d') or k == ord('D'):
            cv2.destroyWindow(win)
            logging.info("Lane drawing complete. Total lanes: %d", len(lanes_orig))
            return lanes_orig
        elif k in (ord('q'), 27):
            cv2.destroyAllWindows()
            sys.exit(0)


def check_lane_overlap(new_lane, existing_lanes):
    if len(existing_lanes) == 0:
        return True, ""
    new_poly = new_lane.astype(np.float32)
    for i, existing in enumerate(existing_lanes):
        existing_poly = existing.astype(np.float32)
        for pt in new_poly:
            if cv2.pointPolygonTest(existing_poly, (float(pt[0]), float(pt[1])), False) > 0:
                return False, f"overlaps with Lane {i+1}"
        for pt in existing_poly:
            if cv2.pointPolygonTest(new_poly, (float(pt[0]), float(pt[1])), False) > 0:
                return False, f"overlaps with Lane {i+1}"
        if polygons_intersect(new_poly, existing_poly):
            return False, f"edges cross Lane {i+1}"
    return True, ""


def polygons_intersect(poly1, poly2):
    def ccw(A, B, C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
    def segments_intersect(A, B, C, D):
        return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)
    n1, n2 = len(poly1), len(poly2)
    for i in range(n1):
        A, B = poly1[i], poly1[(i+1) % n1]
        for j in range(n2):
            C, D = poly2[j], poly2[(j+1) % n2]
            if segments_intersect(A, B, C, D):
                return True
    return False


def save_lanes_csv(lanes, csv_path):
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["lane_id", "p1_x", "p1_y", "p2_x", "p2_y", "p3_x", "p3_y", "p4_x", "p4_y"])
        for i, lane in enumerate(lanes, start=1):
            pts = lane.flatten().tolist()
            writer.writerow([i] + [int(p) for p in pts])
    logging.info("Lanes saved to %s", csv_path)


def prompt_positive_float_window(root, msg):
    while True:
        v = simpledialog.askstring("Calibration Input", msg, parent=root)
        if v is None:
            raise RuntimeError("Calibration cancelled by user.")
        try:
            f = float(v)
            if f > 0:
                return f
        except Exception:
            pass
        messagebox.showerror("Invalid input", "Please enter a positive number.", parent=root)


def run_calibration(video_path, save_dir, root, display_w=1280):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("[!] Could not open calibration video.")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS)
    logging.info(f"Video info: {W}x{H} @ {FPS:.2f} fps")
    os.makedirs(save_dir, exist_ok=True)

    frames, idxs = grab_four_frames(cap)
    for i, (f, idx) in enumerate(zip(frames, idxs)):
        if f is not None:
            cv2.imwrite(os.path.join(save_dir, f"snap_{i+1}_frame{idx}.png"), f)

    chosen = choose_frame(frames)
    chosen_frame = frames[chosen].copy()

    poly = collect_polygon(chosen_frame, display_w=display_w)
    poly_str = f"np.array([{', '.join([str([int(x), int(y)]) for x, y in poly.tolist()])}])"
    logging.info("Selected polygon coordinates (ORIGINAL pixels): %s", poly_str)

    L12_m = prompt_positive_float_window(root, "Length between P1 and P2 (meters):")
    L23_m = prompt_positive_float_window(root, "Length between P2 and P3 (meters):")
    logging.info("Stored lengths: L12=%.6f m, L23=%.6f m", L12_m, L23_m)

    messagebox.showinfo("Lane Drawing",
                        "Next step: Draw lanes within the ROI.\n\n"
                        "- Click 4 points clockwise for each lane\n"
                        "- Press Enter to confirm each lane\n"
                        "- Press D when done\n"
                        "- Lanes should not overlap",
                        parent=root)
    lanes = collect_lanes(chosen_frame, poly, display_w=display_w)
    logging.info("Total lanes defined: %d", len(lanes))

    out_txt = os.path.join(save_dir, "calibration_output.txt")
    with open(out_txt, "w", encoding="utf-8") as fh:
        fh.write(f"Video: {video_path}\n")
        fh.write(f"Frame size: {W}x{H} @ {FPS:.2f} fps\n")
        fh.write(f"Chosen snapshot index: {chosen}\n\n")
        fh.write("ROI Polygon (image pixels):\n")
        for i, (x, y) in enumerate(poly.tolist(), 1):
            fh.write(f"  P{i}: [{int(x)}, {int(y)}]\n")
        fh.write("\nPolygon as numpy: " + poly_str + "\n\n")
        fh.write("Known real-world lengths (meters):\n")
        fh.write(f"  L12 (P1-P2): {L12_m:.6f}\n")
        fh.write(f"  L23 (P2-P3): {L23_m:.6f}\n\n")
        fh.write(f"Number of lanes: {len(lanes)}\n")
        for i, lane in enumerate(lanes, 1):
            fh.write(f"\nLane {i} (image pixels):\n")
            for j, (x, y) in enumerate(lane.tolist(), 1):
                fh.write(f"  P{j}: [{int(x)}, {int(y)}]\n")

    vis = chosen_frame.copy()
    overlay = vis.copy()
    cv2.fillPoly(overlay, [poly.astype(np.int32)], GREY)
    cv2.addWeighted(overlay, 0.25, vis, 0.75, 0, vis)
    cv2.polylines(vis, [poly.astype(np.int32)], True, YELLOW, 2)
    for i, p in enumerate(poly.astype(int).tolist(), 1):
        cv2.circle(vis, (p[0], p[1]), 6, YELLOW, -1)
        cv2.putText(vis, str(i), (p[0] + 6, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, BLACK, 2)
    for lane_idx, lane in enumerate(lanes):
        color = LANE_COLORS[lane_idx % len(LANE_COLORS)]
        lane_pts = lane.astype(np.int32)
        overlay = vis.copy()
        cv2.fillPoly(overlay, [lane_pts], color)
        cv2.addWeighted(overlay, 0.3, vis, 0.7, 0, vis)
        cv2.polylines(vis, [lane_pts], True, color, 2)
        cx = int(np.mean(lane[:, 0]))
        cy = int(np.mean(lane[:, 1]))
        cv2.putText(vis, f"Lane {lane_idx + 1}", (cx - 30, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
    cv2.imwrite(os.path.join(save_dir, "calibration_visual.png"), vis)

    cap.release()
    cv2.destroyAllWindows()
    return poly.astype(np.float32), float(L12_m), float(L23_m), lanes


# ============================
# ----- DETECTION PART -----
# ============================
class PlaneMapper:
    def __init__(self, src_quad: np.ndarray, dst_quad: np.ndarray) -> None:
        self._H = cv2.getPerspectiveTransform(src_quad.astype(np.float32), dst_quad.astype(np.float32))

    def warp_points(self, pts_xy: np.ndarray) -> np.ndarray:
        if pts_xy is None or len(pts_xy) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        pts = pts_xy.reshape(-1, 1, 2).astype(np.float32)
        warped = cv2.perspectiveTransform(pts, self._H)
        return warped.reshape(-1, 2)


def mean_speed_kmh(track_pts, fps):
    if len(track_pts) < 2:
        return None
    (x0, y0) = track_pts[0]
    (x1, y1) = track_pts[-1]
    d_units = float(np.hypot(x1 - x0, y1 - y0))
    dt = len(track_pts) / fps
    if dt <= 0:
        return None
    return (d_units / dt) * 3.6


def iou_xyxy(a, b):
    xA = max(a[0], b[0]); yA = max(a[1], b[1])
    xB = min(a[2], b[2]); yB = min(a[3], b[3])
    inter_w = max(0.0, xB - xA); inter_h = max(0.0, yB - yA)
    inter = inter_w * inter_h
    if inter == 0:
        return 0.0
    area_a = max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1])
    area_b = max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])
    union = area_a + area_b - inter
    return inter / max(union, 1e-6)


def suppress_track_dupes(dets: sv.Detections, iou=0.85) -> sv.Detections:
    if len(dets) <= 1:
        return dets
    boxes = dets.xyxy
    confs = dets.confidence if dets.confidence is not None else np.ones((len(dets),), dtype=float)
    order = np.argsort(-confs)
    keep_mask = np.ones((len(dets),), dtype=bool)
    for i_idx in range(len(order)):
        i = order[i_idx]
        if not keep_mask[i]:
            continue
        for j_idx in range(i_idx + 1, len(order)):
            j = order[j_idx]
            if not keep_mask[j]:
                continue
            ti = None if dets.tracker_id is None else dets.tracker_id[i]
            tj = None if dets.tracker_id is None else dets.tracker_id[j]
            if ti is None or tj is None:
                continue
            if iou_xyxy(boxes[i], boxes[j]) >= iou:
                keep_mask[j] = False
    return dets[keep_mask]


def make_polygon_mask(img_shape, polygon_xy):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [polygon_xy.astype(np.int32)], 255)
    return mask


def filter_detections_to_polygon(dets: sv.Detections, polygon_xy: np.ndarray, anchor: sv.Position = sv.Position.CENTER):
    if len(dets) == 0:
        return dets, np.zeros((0,), dtype=bool)
    anchors = dets.get_anchors_coordinates(anchor=anchor).astype(np.float32)
    poly = polygon_xy.astype(np.float32)
    inside = np.array([cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0 for (x, y) in anchors], dtype=bool)
    return dets[inside], inside


def stylize_display(base_frame, polygon_xy, lanes, ln_px, blur_outside=True):
    mask = make_polygon_mask(base_frame.shape, polygon_xy)
    if blur_outside:
        overlay = base_frame.copy()
        cv2.fillPoly(overlay, [polygon_xy.astype(np.int32)], color=(0, 255, 255))
        tinted = base_frame.copy()
        cv2.addWeighted(overlay, 0.15, tinted, 0.85, 0, tinted)
        blurred = cv2.GaussianBlur(tinted, (31, 31), 0)
        display = blurred.copy()
        mask3 = cv2.merge([mask, mask, mask])
        display[mask3 == 255] = tinted[mask3 == 255]
    else:
        display = base_frame.copy()

    for lane_idx, lane in enumerate(lanes):
        color = LANE_COLORS[lane_idx % len(LANE_COLORS)]
        lane_pts = lane.astype(np.int32)
        overlay = display.copy()
        cv2.fillPoly(overlay, [lane_pts], color)
        cv2.addWeighted(overlay, 0.15, display, 0.85, 0, display)
        cv2.polylines(display, [lane_pts], True, color, max(1, ln_px - 1))

    cv2.polylines(display, [polygon_xy.astype(np.int32)], True, (0, 255, 255), ln_px)
    return display, mask


def run_tracking(
    source_video_path, target_video_path, csv_summary_path, csv_tracks_path, csv_lanes_path,
    calib_save_dir, calib_display_w, weights_path, confidence_threshold, nms_iou, imgsz,
    blur_outside, show_preview, force_cpu=False, preview_w=960, preview_x=100, preview_y=100, root=None
):
    logging.info("UI run started")

    CAL_SRC, W_m, H_m, lanes = run_calibration(video_path=source_video_path, save_dir=calib_save_dir, root=root, display_w=calib_display_w)
    CAL_DST = np.array([[0, 0], [W_m, 0], [W_m, H_m], [0, H_m]], dtype=np.float32)
    logging.info("Calibration finished. CAL_SRC=%s W_m=%.4f H_m=%.4f, Lanes=%d", CAL_SRC.tolist(), W_m, H_m, len(lanes))

    save_lanes_csv(lanes, csv_lanes_path)

    vidmeta = sv.VideoInfo.from_video_path(video_path=source_video_path)
    logging.info("Tracking video meta: %s, FPS=%.3f", vidmeta.resolution_wh, vidmeta.fps)
    frames = sv.get_video_frames_generator(source_path=source_video_path)

    yolo = YOLO(weights_path)
    try:
        import torch
        DEVICE = "cpu" if force_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    except Exception:
        DEVICE = "cpu"
    logging.info("Running on device: %s (force_cpu=%s)", DEVICE, force_cpu)

    dummy = np.zeros((vidmeta.resolution_wh[1], vidmeta.resolution_wh[0], 3), dtype=np.uint8)
    _ = yolo(dummy, imgsz=imgsz, device=DEVICE, verbose=False)[0]
    logging.info("Warming up model... done.")

    init_params = set(inspect.signature(sv.ByteTrack.__init__).parameters.keys())
    try:
        if {"track_thresh", "match_thresh", "track_buffer"} <= init_params:
            tracker = sv.ByteTrack(frame_rate=vidmeta.fps, track_thresh=0.25, match_thresh=0.6, track_buffer=int(vidmeta.fps * 8))
        elif {"track_activation_threshold"} <= init_params:
            tracker = sv.ByteTrack(frame_rate=vidmeta.fps, track_activation_threshold=confidence_threshold)
        else:
            tracker = sv.ByteTrack(frame_rate=vidmeta.fps)
    except TypeError:
        tracker = sv.ByteTrack(frame_rate=vidmeta.fps, track_activation_threshold=confidence_threshold)

    mapper = PlaneMapper(src_quad=CAL_SRC, dst_quad=CAL_DST)
    ln_px = sv.calculate_optimal_line_thickness(resolution_wh=vidmeta.resolution_wh)
    base_txt_scale = sv.calculate_optimal_text_scale(resolution_wh=vidmeta.resolution_wh)
    txt_scale = base_txt_scale * 0.6
    draw_boxes = sv.BoxAnnotator(thickness=ln_px)
    draw_labels = sv.LabelAnnotator(text_scale=txt_scale, text_thickness=max(1, ln_px - 1), text_position=sv.Position.TOP_CENTER)
    draw_traces = sv.TraceAnnotator(thickness=ln_px, trace_length=int(vidmeta.fps * 2), position=sv.Position.CENTER)

    trails_roi = defaultdict(lambda: deque(maxlen=int(vidmeta.fps)))
    class_votes = defaultdict(Counter)
    first_frame = {}
    last_frame = {}

    stationary_tracks = {}
    STATIONARY_THRESHOLD = 2.0
    STATIONARY_GRACE = int(vidmeta.fps * 15)
    MOVING_GRACE = int(max(1, 0.5 * vidmeta.fps))
    logging.info("Stationary vehicle handling: threshold=%.2fm, grace=%d/%d frames", STATIONARY_THRESHOLD, STATIONARY_GRACE, MOVING_GRACE)

    os.makedirs(os.path.dirname(csv_summary_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(csv_tracks_path) or ".", exist_ok=True)

    csv_sum_fh = open(csv_summary_path, "w", newline="", encoding="utf-8")
    csv_sum_out = csv.writer(csv_sum_fh)
    csv_sum_out.writerow(["vehicle_id", "label", "avg_speed_kmh", "start_frame", "end_frame", "start_time_s", "end_time_s"])

    csv_trk_fh = open(csv_tracks_path, "w", newline="", encoding="utf-8")
    csv_trk_out = csv.writer(csv_trk_fh)
    csv_trk_out.writerow(["frame", "time_s", "vehicle_id", "x_m", "y_m", "img_x", "img_y"])

    if show_preview:
        aspect = vidmeta.resolution_wh[1] / vidmeta.resolution_wh[0]
        disp_w = int(preview_w)
        disp_h = int(disp_w * aspect)
        cv2.namedWindow("preview", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("preview", disp_w, disp_h)
        cv2.moveWindow("preview", preview_x, preview_y)

    f_idx = 0
    try:
        with sv.VideoSink(target_video_path, vidmeta) as vsink:
            for frame in frames:
                yout = yolo(frame, imgsz=imgsz, device=DEVICE, verbose=False)[0]
                raw_dets = sv.Detections.from_ultralytics(yout)
                dets = raw_dets
                if dets.confidence is not None:
                    dets = dets[dets.confidence > 0.15]
                try:
                    dets = dets.with_nms(threshold=nms_iou, class_agnostic=True)
                except TypeError:
                    dets = dets.with_nms(threshold=nms_iou)

                dets_roi, _ = filter_detections_to_polygon(dets, CAL_SRC, anchor=sv.Position.CENTER)
                dets_tracked = tracker.update_with_detections(detections=dets_roi)
                dets_tracked = suppress_track_dupes(dets_tracked, iou=0.85)

                bc_xy = dets_tracked.get_anchors_coordinates(anchor=sv.Position.CENTER)
                bc_warp = mapper.warp_points(pts_xy=bc_xy).astype(float)

                class_names = getattr(yout, "names", None)
                def id_to_name(c):
                    if class_names and c is not None and int(c) in class_names:
                        return class_names[int(c)]
                    return str(int(c)) if c is not None else "unknown"

                ids = dets_tracked.tracker_id
                cls = dets_tracked.class_id
                if ids is None:
                    ids = np.empty((0,), dtype=int)

                for idx in range(len(ids)):
                    tid = ids[idx]
                    if tid is None:
                        continue
                    x_m, y_m = float(bc_warp[idx][0]), float(bc_warp[idx][1])
                    img_x, img_y = float(bc_xy[idx][0]), float(bc_xy[idx][1])

                    if tid in stationary_tracks:
                        prev_pos = stationary_tracks[tid]['last_pos']
                        dist_moved = np.hypot(x_m - prev_pos[0], y_m - prev_pos[1])
                        if dist_moved < STATIONARY_THRESHOLD:
                            stationary_tracks[tid]['frames_stationary'] += 1
                            stationary_tracks[tid]['last_seen'] = f_idx
                        else:
                            stationary_tracks[tid]['frames_stationary'] = 0
                            stationary_tracks[tid]['last_pos'] = (x_m, y_m)
                            stationary_tracks[tid]['last_seen'] = f_idx
                    else:
                        stationary_tracks[tid] = {'last_pos': (x_m, y_m), 'frames_stationary': 0, 'last_seen': f_idx}

                    trails_roi[tid].append((x_m, y_m))
                    last_frame[tid] = f_idx
                    if tid not in first_frame:
                        first_frame[tid] = f_idx

                    cid = None
                    if cls is not None and idx < len(cls):
                        cid = cls[idx]
                    class_votes[tid][id_to_name(cid)] += 1

                    t_s = f_idx / vidmeta.fps
                    csv_trk_out.writerow([f_idx, round(t_s, 3), tid, round(x_m, 3), round(y_m, 3), round(img_x, 1), round(img_y, 1)])
                csv_trk_fh.flush()

                lbls = []
                disp_ids = dets_tracked.tracker_id
                if disp_ids is None:
                    disp_ids = np.empty((0,), dtype=int)
                for i_disp in range(len(disp_ids)):
                    tid = disp_ids[i_disp]
                    pts_roi = trails_roi.get(tid, deque())
                    v = mean_speed_kmh(pts_roi, vidmeta.fps)
                    lbls.append(f"#{tid}" if v is None else f"#{tid} {int(v)} km/h")

                display, _ = stylize_display(frame, CAL_SRC, lanes, ln_px, blur_outside=blur_outside)
                display = draw_traces.annotate(scene=display, detections=dets_tracked)
                display = draw_boxes.annotate(scene=display, detections=dets_tracked)
                display = draw_labels.annotate(scene=display, detections=dets_tracked, labels=lbls)
                vsink.write_frame(display)

                to_close = []
                for tid in list(last_frame.keys()):
                    frames_missing = f_idx - last_frame.get(tid, -10**9)
                    if tid in stationary_tracks and stationary_tracks[tid]['frames_stationary'] > int(vidmeta.fps):
                        grace_period = STATIONARY_GRACE
                    else:
                        grace_period = MOVING_GRACE
                    if frames_missing > grace_period:
                        to_close.append(tid)

                for tid in to_close:
                    pts = trails_roi.get(tid, deque())
                    if len(pts) >= 2:
                        v = mean_speed_kmh(pts, vidmeta.fps)
                        f0 = first_frame.get(tid, 0)
                        f1 = last_frame.get(tid, f_idx)
                        t0 = f0 / vidmeta.fps
                        t1 = f1 / vidmeta.fps
                        votes = class_votes.get(tid, Counter())
                        if len(votes) == 0:
                            label = "unknown"
                        else:
                            label, _ = max(((k, c) for k, c in votes.items() if k != "unknown"), default=max(votes.items(), key=lambda kv: kv[1]))
                        if v is not None:
                            csv_sum_out.writerow([tid, label, round(v, 1), f0, f1, round(t0, 3), round(t1, 3)])
                            csv_sum_fh.flush()
                    trails_roi.pop(tid, None)
                    class_votes.pop(tid, None)
                    first_frame.pop(tid, None)
                    last_frame.pop(tid, None)
                    stationary_tracks.pop(tid, None)

                if show_preview:
                    disp_frame = cv2.resize(display, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
                    cv2.imshow("preview", disp_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                f_idx += 1
    finally:
        csv_sum_fh.close()
        csv_trk_fh.close()
        cv2.destroyAllWindows()


# ============================
# --------- UI ---------------
# ============================
def launch_ui():
    root = tk.Tk()
    root.title("Calibration + Lane + ROI Tracking UI")
    pad = dict(padx=6, pady=4)
    frm = ttk.Frame(root, padding=8)
    frm.grid(row=0, column=0, sticky="nsew")
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    frm.columnconfigure(1, weight=1)

    video_var = tk.StringVar()
    weights_var = tk.StringVar(value="yolo11l.pt")
    outdir_var = tk.StringVar()
    conf_var = tk.StringVar(value="0.5")
    nms_var = tk.StringVar(value="0.5")
    imgsz_var = tk.StringVar(value="1280")
    blur_var = tk.BooleanVar(value=True)
    preview_var = tk.BooleanVar(value=True)
    force_cpu_var = tk.BooleanVar(value=False)

    def browse_video():
        p = filedialog.askopenfilename(title="Select input video", filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")])
        if p:
            video_var.set(p)
            if not outdir_var.get():
                outdir_var.set(os.path.dirname(p))
            update_outputs()

    def browse_weights():
        p = filedialog.askopenfilename(title="Select YOLO weights (.pt)", filetypes=[("PyTorch weights", "*.pt"), ("All files", "*.*")])
        if p:
            weights_var.set(p)
            update_outputs()

    def browse_outdir():
        p = filedialog.askdirectory(title="Select output folder")
        if p:
            outdir_var.set(p)
            update_outputs()

    def update_outputs(*_):
        out_dir = outdir_var.get().strip()
        if not out_dir:
            return
        video_out_var.set(os.path.join(out_dir, "vehicles-result.mp4"))
        csv_sum_out_var.set(os.path.join(out_dir, "vehicles.csv"))
        csv_trk_out_var.set(os.path.join(out_dir, "vehicle_tracks_xy.csv"))
        csv_lanes_var.set(os.path.join(out_dir, "lanes.csv"))
        calib_dir_var.set(os.path.join(out_dir, "calib_screens"))
        log_var.set(os.path.join(out_dir, "runtime.log"))

    r = 0
    ttk.Label(frm, text="Input Video:").grid(row=r, column=0, sticky="w", **pad)
    ttk.Entry(frm, textvariable=video_var).grid(row=r, column=1, sticky="ew", **pad)
    ttk.Button(frm, text="Browse...", command=browse_video).grid(row=r, column=2, **pad)
    r += 1
    ttk.Label(frm, text="YOLO Weights (.pt):").grid(row=r, column=0, sticky="w", **pad)
    ttk.Entry(frm, textvariable=weights_var).grid(row=r, column=1, sticky="ew", **pad)
    ttk.Button(frm, text="Browse...", command=browse_weights).grid(row=r, column=2, **pad)
    r += 1
    ttk.Label(frm, text="Output Folder:").grid(row=r, column=0, sticky="w", **pad)
    ttk.Entry(frm, textvariable=outdir_var).grid(row=r, column=1, sticky="ew", **pad)
    ttk.Button(frm, text="Browse...", command=browse_outdir).grid(row=r, column=2, **pad)
    r += 1
    ttk.Separator(frm).grid(row=r, column=0, columnspan=3, sticky="ew", pady=8)
    r += 1
    ttk.Label(frm, text="Confidence Threshold:").grid(row=r, column=0, sticky="w", **pad)
    ttk.Entry(frm, textvariable=conf_var, width=8).grid(row=r, column=1, sticky="w", **pad)
    r += 1
    ttk.Label(frm, text="NMS IoU Threshold:").grid(row=r, column=0, sticky="w", **pad)
    ttk.Entry(frm, textvariable=nms_var, width=8).grid(row=r, column=1, sticky="w", **pad)
    r += 1
    ttk.Label(frm, text="YOLO imgsz (long side):").grid(row=r, column=0, sticky="w", **pad)
    ttk.Entry(frm, textvariable=imgsz_var, width=8).grid(row=r, column=1, sticky="w", **pad)
    r += 1
    ttk.Checkbutton(frm, text="Blur outside ROI", variable=blur_var).grid(row=r, column=0, sticky="w", **pad)
    ttk.Checkbutton(frm, text="Show preview window", variable=preview_var).grid(row=r, column=1, sticky="w", **pad)
    ttk.Checkbutton(frm, text="Force CPU", variable=force_cpu_var).grid(row=r, column=2, sticky="w", **pad)
    r += 1
    ttk.Separator(frm).grid(row=r, column=0, columnspan=3, sticky="ew", pady=8)
    r += 1

    video_out_var = tk.StringVar()
    csv_sum_out_var = tk.StringVar()
    csv_trk_out_var = tk.StringVar()
    csv_lanes_var = tk.StringVar()
    calib_dir_var = tk.StringVar()
    log_var = tk.StringVar()

    ttk.Label(frm, text="Auto-generated outputs:").grid(row=r, column=0, columnspan=3, sticky="w", **pad)
    r += 1

    def out_row(label, var):
        nonlocal r
        ttk.Label(frm, text=label).grid(row=r, column=0, sticky="w", **pad)
        ttk.Entry(frm, textvariable=var, state="readonly").grid(row=r, column=1, columnspan=2, sticky="ew", **pad)
        r += 1

    out_row("Output Video:", video_out_var)
    out_row("vehicles.csv:", csv_sum_out_var)
    out_row("vehicle_tracks_xy.csv:", csv_trk_out_var)
    out_row("lanes.csv:", csv_lanes_var)
    out_row("Calibration Folder:", calib_dir_var)
    out_row("runtime.log:", log_var)

    ttk.Separator(frm).grid(row=r, column=0, columnspan=3, sticky="ew", pady=8)
    r += 1
    ttk.Label(frm, text="Run lane_postprocess.py after tracking to add lane assignments.", foreground="gray").grid(row=r, column=0, columnspan=3, sticky="w", **pad)
    r += 1

    def run_clicked():
        try:
            video_path = video_var.get().strip()
            weights_path = weights_var.get().strip()
            out_dir = outdir_var.get().strip()
            if not video_path or not os.path.exists(video_path):
                messagebox.showerror("Missing video", "Please select a valid input video.", parent=root)
                return
            if not weights_path or not os.path.exists(weights_path):
                messagebox.showerror("Missing weights", "Please select valid YOLO .pt weights.", parent=root)
                return
            if not out_dir:
                messagebox.showerror("Missing output folder", "Please select an output folder.", parent=root)
                return
            update_outputs()
            setup_logger(log_var.get())
            conf = float(conf_var.get())
            nms = float(nms_var.get())
            imgsz = int(float(imgsz_var.get()))
            run_tracking(
                source_video_path=video_path, target_video_path=video_out_var.get(),
                csv_summary_path=csv_sum_out_var.get(), csv_tracks_path=csv_trk_out_var.get(),
                csv_lanes_path=csv_lanes_var.get(), calib_save_dir=calib_dir_var.get(),
                calib_display_w=1280, weights_path=weights_path, confidence_threshold=conf,
                nms_iou=nms, imgsz=imgsz, blur_outside=bool(blur_var.get()),
                show_preview=bool(preview_var.get()), force_cpu=bool(force_cpu_var.get()), root=root
            )
            messagebox.showinfo("Done", "Tracking finished!\n\nRun post-processing:\n  python lane_postprocess.py <output_folder>", parent=root)
        except Exception as e:
            log_path = log_var.get() or "(no log path)"
            logging.error("Unhandled exception:\n%s", traceback.format_exc())
            messagebox.showerror("CalibTrackApp crashed", f"{type(e).__name__}: {e}\n\nSee log:\n{log_path}", parent=root)

    ttk.Button(frm, text="Run Calibration + Tracking", command=run_clicked).grid(row=r, column=0, columnspan=3, pady=16)
    video_var.trace_add("write", update_outputs)
    outdir_var.trace_add("write", update_outputs)
    root.mainloop()


if __name__ == "__main__":
    launch_ui()