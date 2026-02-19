## Welcome to SimJam Computer Vision Analytics (Open-Source)

<p align="left">
  <img src="https://img.shields.io/github/contributors/RoadwayVR/SimJamComputerVision?style=for-the-badge">
  <img src="https://img.shields.io/github/forks/RoadwayVR/SimJamComputerVision?style=for-the-badge">
  <img src="https://img.shields.io/github/stars/RoadwayVR/SimJamComputerVision?style=for-the-badge">
  <img src="https://img.shields.io/github/issues/RoadwayVR/SimJamComputerVision?style=for-the-badge">
  <img src="https://img.shields.io/github/license/RoadwayVR/SimJamComputerVision?style=for-the-badge">
  <a href="https://www.linkedin.com/in/ahmadmohammadi1441/">
    <img src="https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge">
  </a>
</p>

## Introduction
Transportation planning and traffic operations studies increasingly rely on **video data** (CCTV, drone, roadside cameras, dashcams). However, extracting reliable mobility measures from video often requires repeated manual work and disconnected tools.

**SimJam Computer Vision Analytics** is an open-source application that turns raw traffic video into planning-ready outputs. It supports:

1. **Traffic object detection** (cars, trucks, buses, bicycles, pedestrians, etc.)
2. **Multi-object tracking** to keep consistent IDs over time
3. **Mobility analytics** such as counts, trajectories, and speed estimation (when calibration is available)
4. **Exportable outputs** (CSV summaries and structured results) to support planning studies and reporting

Typical use cases:
- Turning movement counts (TMC) and approach volumes
- Speed estimation and speed distributions
- Trajectory extraction for safety/near-miss analysis
- Before/after studies (traffic calming, signal timing, policy changes)
- Data preparation for microsimulation calibration/validation

## Workflow Overview
A practical workflow is:

1) **Load a video** (CCTV / drone / roadside)  
2) **Detect + track** road users using YOLO-based models  
3) **Export analytics** (counts / speeds / trajectories / summaries)

<p align="center">
  <img src="https://github.com/user-attachments/assets/00077e81-e551-42a0-808f-74382595231e"
       alt="SimJam workflow"
       width="720">
</p>

## Short Demo Videos (Click to Play)

### 1) Detection + Tracking Demo
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
1. Clone the repository:
```bash
git clone https://github.com/RoadwayVR/SimJamComputerVision.git
cd SimJamComputerVision
