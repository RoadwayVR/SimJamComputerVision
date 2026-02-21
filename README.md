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

1) **Detect + track** road users using YOLO-based models  
<p align="center">
  <img src="https://github.com/user-attachments/assets/00077e81-e551-42a0-808f-74382595231e"
       alt="Detection + Tracking interface (SimJam Computer Vision)"
       width="720">
</p>
<p align="center"><em>Step 1 — Detection + Tracking</em></p>

2) **Export analytics** (counts / speeds / trajectories / summaries)
<p align="center">
  <img src="https://github.com/user-attachments/assets/8a588cb9-e2b7-463b-986b-f9be19fdafe4"
       alt="Analytics + Export interface (CSV summaries, counts, speeds)"
       width="720">
</p>
<p align="center"><em>Step 2 — Analytics + Export</em></p>

## Short Demo Video (Click to Play)

<p align="center"><em>Short demo video - Click to Play</em></p>
<p align="center">
  <a href="https://youtu.be/ez3ZOUufBHY" target="_blank" rel="noopener noreferrer">
    <img src="https://github.com/user-attachments/assets/6f54524c-8524-4ac8-adaa-4cf4ee3f8eb5"
         alt="SimJam Computer Vision Analytics - Short Demo"
         width="720">
  </a>
</p>


## Getting Started (Video Tutorial)

The easiest way to get started is to follow the step-by-step video tutorial:

🎥 **Getting Started Tutorial (Click to Play)**  
<p align="center">
  <a href="https://youtu.be/By7EyE-WsxI" target="_blank" rel="noopener noreferrer">
    <img src="https://github.com/user-attachments/assets/14896b21-8088-4302-8845-f95d2fc83ea1"
         alt="SimJam Computer Vision Analytics - Getting Started Tutorial"
         width="720">
  </a>
</p>

## Requirement 
Python version 3.12 and higher

Visual Studio code

## License
This project is licensed under the MIT License. It uses Ultralytics YOLO which is licensed under AGPL-3.0. This project is distributed as open-source in compliance with that license.
