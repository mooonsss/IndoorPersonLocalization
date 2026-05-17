import cv2
import numpy as np


class CoordinateMapper:
    def __init__(self, src_points, dst_points):
        """
        初始化坐标映射器，计算单应性矩阵 (Homography Matrix)

        Args:
            src_points (list): 视频画面中的 4 个参考点坐标，例如 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            dst_points (list): 2D 平面图中对应的 4 个参考点坐标
        """
        # 将传入的列表转换为 OpenCV 需要的 float32 格式的 NumPy 数组
        pts_src = np.array(src_points, dtype=np.float32)
        pts_dst = np.array(dst_points, dtype=np.float32)

        # 计算 3x3 的透视变换矩阵 H
        # cv2.findHomography 比 getPerspectiveTransform 更鲁棒，哪怕以后你给 5 个、6 个点它也能算最优解
        self.H, _ = cv2.findHomography(pts_src, pts_dst)

    def pixel_to_map(self, foot_point):
        """
        将视频画面中的像素坐标映射到平面图坐标

        Args:
            foot_point (list/tuple): 画面中的脚底坐标 (x, y)

        Returns:
            (int, int): 映射到 2D 平面图上的 (X, Y) 坐标
        """
        x, y = foot_point

        # OpenCV 的 perspectiveTransform 函数需要特定的数组形状：(1, 1, 2)
        pt_src = np.array([[[x, y]]], dtype=np.float32)

        # 执行矩阵变换
        pt_dst = cv2.perspectiveTransform(pt_src, self.H)

        # 提取映射后的坐标
        map_x, map_y = pt_dst[0][0]

        return int(map_x), int(map_y)