import numpy as np
import cv2


class DepthFusion:
    def __init__(self, intrinsics: np.ndarray):
        self.intrinsics = intrinsics  # Камерная матрица K: 3x3
        self.points = []  # 3D-точки (облако)

    def add_frame(self, rgb_image: np.ndarray, depth_map: np.ndarray, pose: np.ndarray):
        h, w = depth_map.shape
        fx, fy = self.intrinsics[0, 0], self.intrinsics[1, 1]
        cx, cy = self.intrinsics[0, 2], self.intrinsics[1, 2]

        # Сгенерируем сетку координат
        xs, ys = np.meshgrid(np.arange(w), np.arange(h))
        zs = depth_map.flatten()
        xs = xs.flatten()
        ys = ys.flatten()

        # Отфильтруем слишком маленькие или нулевые глубины
        valid = zs > 0.1
        xs, ys, zs = xs[valid], ys[valid], zs[valid]

        # Обратный проецирование в 3D в координатах камеры
        x = (xs - cx) * zs / fx
        y = (ys - cy) * zs / fy
        z = zs
        pts_camera = np.vstack((x, y, z, np.ones_like(z)))  # shape: (4, N)

        # Перенос в мировую систему координат
        pts_world = pose @ pts_camera  # shape: (4, N)
        self.points.append(pts_world[:3, :].T)  # только XYZ

    def get_point_cloud(self) -> np.ndarray:
        if self.points:
            return np.vstack(self.points)
        return np.empty((0, 3))
