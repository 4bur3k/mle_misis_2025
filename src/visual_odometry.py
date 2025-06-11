import numpy as np
import cv2

class VisualOdometry:
    def __init__(self, focal_length, pp):
        self.focal = focal_length
        self.pp = pp
        self.detector = cv2.ORB_create(2000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        self.prev_kp = None
        self.prev_des = None
        self.prev_frame = None

        self.pose = np.eye(4)
        self.R_total = np.eye(3)
        self.t_total = np.zeros((3, 1))

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = self.detector.detectAndCompute(gray, None)

        if self.prev_kp is None:
            self.prev_kp = kp
            self.prev_des = des
            self.prev_frame = gray
            return None, []

        matches = self.matcher.match(self.prev_des, des)
        matches = sorted(matches, key=lambda x: x.distance)[:500]

        pts1 = np.float32([self.prev_kp[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp[m.trainIdx].pt for m in matches])

        E, mask = cv2.findEssentialMat(pts2, pts1, focal=self.focal, pp=self.pp, method=cv2.RANSAC, prob=0.999)
        if E is None:
            return None, []

        _, R, t, _ = cv2.recoverPose(E, pts2, pts1, focal=self.focal, pp=self.pp)

        self.t_total += self.R_total @ t
        self.R_total = R @ self.R_total

        self.prev_kp = kp
        self.prev_des = des
        self.prev_frame = gray

        pose = np.eye(4)
        pose[:3, :3] = self.R_total
        pose[:3, 3] = self.t_total.squeeze()
        return pose, [kp[m.trainIdx].pt for m in matches]
