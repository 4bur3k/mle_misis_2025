import streamlit as st
import cv2
from src.visual_odometry import VisualOdometry
from src.visualizer import Trajectory2DVisualizer, draw_keypoints
import numpy as np

st.title("SLAM-lite: Visual Odometry from Video")

video_file = st.sidebar.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
focal_length = st.sidebar.slider("Focal Length (px)", 100, 2000, 718)
pp_x = st.sidebar.slider("Principal Point X", 0, 1280, 607)
pp_y = st.sidebar.slider("Principal Point Y", 0, 720, 185)

if video_file:
    tvis = Trajectory2DVisualizer()
    stframe1 = st.empty()
    stframe2 = st.empty()

    vo = VisualOdometry(focal_length=focal_length, pp=(pp_x, pp_y))
    cap = cv2.VideoCapture('data/'+video_file.name)
    
    frame_idx = 0
    print(video_file.name)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        pose, keypoints = vo.process_frame(frame)
        if pose is None:
            continue

        traj_image = tvis.update(pose)
        annotated = draw_keypoints(frame.copy(), keypoints)

        stframe1.image(annotated, caption=f"Frame {frame_idx}", channels="BGR")
        stframe2.image(traj_image, caption="Camera Trajectory", channels="BGR")
        frame_idx += 1
