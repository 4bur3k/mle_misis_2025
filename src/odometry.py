import cv2
import numpy as np


class VisualOdometry:
    def __init__(self):
        self.prev_gray = None
        self.prev_pts = None
        self.pose = np.eye(4, dtype=np.float32)

    def track(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_pts = cv2.goodFeaturesToTrack(gray, maxCorners=200, qualityLevel=0.01, minDistance=7)
            return self.pose.copy()

        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.prev_pts, None)

        good_prev = self.prev_pts[status == 1]
        good_next = next_pts[status == 1]

        if len(good_prev) >= 8:
            E, mask = cv2.findEssentialMat(good_next, good_prev, focal=1.0, pp=(0., 0.), method=cv2.RANSAC, threshold=1.0)
            if E is not None:
                _, R, t, mask = cv2.recoverPose(E, good_next, good_prev)

                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3:] = t

                self.pose = self.pose @ np.linalg.inv(T)  # обновляем глобальную позу

        self.prev_gray = gray
        self.prev_pts = good_next.reshape(-1, 1, 2)

        return self.pose.copy()
