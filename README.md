## Welcome to SimJam Computer Vision Analytics (Open-Source)

<p align="left">
  <img src="https://img.shields.io/github/contributors/RoadwayVR/SimJam-ComputerVision-Analytics?style=for-the-badge">
  <img src="https://img.shields.io/github/forks/RoadwayVR/SimJam-ComputerVision-Analytics?style=for-the-badge">
  <img src="https://img.shields.io/github/stars/RoadwayVR/SimJam-ComputerVision-Analytics?style=for-the-badge">
  <img src="https://img.shields.io/github/issues/RoadwayVR/SimJam-ComputerVision-Analytics?style=for-the-badge">
  <img src="https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge">
  <a href="https://www.linkedin.com/in/ahmadmohammadi1441/">
    <img src="https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge">
  </a>
</p>

## Introduction
Transportation planning and traffic operations studies increasingly rely on **video data** (dashcam, CCTV, drone, or roadside cameras). However, extracting reliable mobility measures from video often requires custom scripts, repeated manual work, and multiple disconnected tools.

**SimJam Computer Vision Analytics** is an open-source application that turns raw traffic video into planning-ready outputs. It is designed as a practical “video → analytics → export” workflow that supports:

1. **Traffic object detection** (cars, trucks, buses, bicycles, pedestrians, etc.)
2. **Multi-object tracking** to maintain consistent IDs over time
3. **Mobility analytics** including counts, turning movements, trajectories, speed profiles, and basic operational measures
4. **Exportable outputs** (CSV summaries and structured results) that can directly support planning studies, reports, and simulation model inputs

Typical use cases:
- Turning movement counts (TMC) and approach volumes
- Speed estimation and speed distributions
- Trajectory extraction for safety/near-miss analysis
- Before/after studies for traffic calming or signal timing updates
- Data preparation for microsimulation calibration and validation

## Workflow Overview
It only requires three steps:

1) **Load video** (CCTV / drone / roadside / dashcam)  
2) **Detect + track** road users using YOLO-based models  
3) **Export analytics** (counts, speeds, trajectories, summaries)

<p align="center">
  <img src="https://github.com/user-attachments/assets/REPLACE_WITH_YOUR_WORKFLOW_IMAGE" alt="SimJam workflow" width="720">
</p>

## A User-Friendly Tool
SimJam is built for **students, researchers, and practitioners**:
- Minimal setup (works as a standalone workflow once dependencies are installed)
- Clear outputs (CSV + plots + summary tables)
- Designed for planning applications (counts, speeds, movement patterns)

> Note: Replace this section with your exact packaging details (e.g., “GUI app”, “Python script”, “exe”, etc.) once finalized.

## Short Demo Videos (Click to Play)

### 1) Vehicle Detection + Tracking Demo
<p align="center"><em>Short video demo - Click to Play</em></p>
<p align="center">
  <a href="https://youtu.be/REPLACE_WITH_VIDEO_1" target="_blank">
    <img src="https://github.com/user-attachments/assets/REPLACE_WITH_THUMBNAIL_1" alt="Detection + Tracking demo" width="720">
  </a>
</p>

### 2) Analytics + Export Demo (Counts / Speeds / CSV)
<p align="center"><em>Short video demo - Click to Play</em></p>
<p align="center">
  <a href="https://youtu.be/REPLACE_WITH_VIDEO_2" target="_blank">
    <img src="https://github.com/user-attachments/assets/REPLACE_WITH_THUMBNAIL_2" alt="Analytics + Export demo" width="720">
  </a>
</p>

## Getting Started

### 1) Install Python
- Install Python 3.10+ (recommended)
- Create a virtual environment (recommended)

### 2) Install Dependencies
```bash
pip install -r requirements.txt
