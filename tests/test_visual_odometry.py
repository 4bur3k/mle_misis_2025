import numpy as np
import cv2
import pytest
from src.visual_odometry import VisualOdometry

def create_blank_frame(width=640, height=480):
    return np.zeros((height, width, 3), dtype=np.uint8)

def test_first_frame_returns_none_and_empty_keypoints():
    vo = VisualOdometry(focal_length=718, pp=(607, 185))
    frame = create_blank_frame()

    pose, keypoints = vo.process_frame(frame)

    assert pose is None, "Первая поза должна быть None"
    assert keypoints == [], "На первом кадре ключевые точки должны отсутствовать"

def test_feature_detection_on_non_empty_frame():
    vo = VisualOdometry(focal_length=718, pp=(607, 185))
    frame = create_blank_frame()
    # Контрастный квадрат — чтобы точки могли быть найдены
    cv2.rectangle(frame, (100, 100), (200, 200), (255, 255, 255), -1)

    pose, keypoints = vo.process_frame(frame)
    
    assert len(keypoints) > 0, "Должны быть обнаружены ключевые точки"

def test_pose_estimation_after_two_frames():
    vo = VisualOdometry(focal_length=718, pp=(607, 185))
    frame1 = create_blank_frame()
    cv2.circle(frame1, (320, 240), 50, (255, 255, 255), -1)

    frame2 = create_blank_frame()
    cv2.circle(frame2, (325, 245), 50, (255, 255, 255), -1)  # немного сдвинули круг

    # Первый кадр
    pose1, kp1 = vo.process_frame(frame1)
    # Второй кадр
    pose2, kp2 = vo.process_frame(frame2)

    assert pose1 is None, "Поза первого кадра должна быть None"
    # После второго кадра должна быть матрица 4x4
    assert pose2 is not None, "Поза второго кадра должна быть вычислена"
    assert pose2.shape == (4, 4), "Поза должна быть 4x4 матрицей"
    assert len(kp2) > 0, "Ключевые точки должны присутствовать и во втором кадре"

def test_reset_works_correctly():
    vo = VisualOdometry(focal_length=718, pp=(607, 185))
    frame = create_blank_frame()

    vo.process_frame(frame)
    vo.reset()
    assert vo.prev_frame is None
    assert vo.prev_pose is None
    assert vo.prev_keypoints == []

