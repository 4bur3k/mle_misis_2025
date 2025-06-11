import numpy as np
import cv2

class Trajectory2DVisualizer:
    def __init__(self, canvas_size=(600, 600)):
        self.canvas = np.ones((canvas_size[1], canvas_size[0], 3), dtype=np.uint8) * 255
        self.trajectory = []
        self.scale = 5

    def update(self, pose):
        x, z = pose[0, 3], pose[2, 3]
        self.trajectory.append((x, z))

        for i in range(1, len(self.trajectory)):
            x1, z1 = self.trajectory[i - 1]
            x2, z2 = self.trajectory[i]
            pt1 = (int(x1 * self.scale + 300), int(z1 * self.scale + 300))
            pt2 = (int(x2 * self.scale + 300), int(z2 * self.scale + 300))
            cv2.line(self.canvas, pt1, pt2, (0, 0, 255), 2)

        return self.canvas.copy()

def draw_keypoints(image, keypoints):
    for pt in keypoints:
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
    return image
