"""
region.py
--------------------------------
功能：
    定义室内区域（如房间、走廊等），
    根据人员检测结果判断其所在区域。

说明：
    - 区域使用多边形（polygon）表示
    - 人的位置使用 bounding box 中心点表示
    - 不负责进入事件判断（由 tracker 模块完成）
"""

import cv2
import numpy as np


class RegionManager:
    def __init__(self):
        """
        初始化区域管理器
        在这里定义所有室内区域
        """

        # 区域定义（示例，需根据你的视频画面自行调整）
        # 每个区域由若干个点组成，按顺时针或逆时针排列
        self.regions = {
            "Corridor": np.array([
                (0, 888),
                (0, 1080),
                (842, 1080),
                (1830, 0),
                (1450, 0)
            ], dtype=np.int32),
            "1": np.array([
                (0, 520),
                (0, 888),
                (1450, 0),
                (1200, 0)
            ]),
            "2": np.array([
                (842, 1080),
                (1680,1080),
                (1850, 600),
                (1900, 0),
                (1830, 0)

            ])

        }

    def get_bbox_center(self, bbox):
        """
        计算 bounding box 的中心点

        Args:
            bbox (list): [x1, y1, x2, y2]

        Returns:
            (cx, cy)
        """
        x1, y1, x2, y2 = bbox
        cx = int((x1 + x2) / 2)
        cy = int(y1 + (y2 - y1) * 0.7)  #框底部向上30%处
        return cx, cy

    def point_in_region(self, point, region_polygon):
        """
        判断点是否在某个区域多边形内

        Args:
            point (tuple): (x, y)
            region_polygon (ndarray): 区域多边形点集

        Returns:
            True / False
        """
        # cv2.pointPolygonTest:
        # >0: 在内部
        # =0: 在边界
        # <0: 在外部
        result = cv2.pointPolygonTest(region_polygon, point, False)
        return result >= 0

    def locate_person(self, bbox):
        """
        根据 bbox 判断人员所在区域

        Args:
            bbox (list): [x1, y1, x2, y2]

        Returns:
            region_name (str): 区域名称，若不在任何区域则返回 "Unknown"
        """
        center_point = self.get_bbox_center(bbox)

        for region_name, polygon in self.regions.items():
            if self.point_in_region(center_point, polygon):
                return region_name

        return "Unknown"

    def add_region(self, region_name, coordinates):
        """
        手动添加一个新的区域

        Args:
            region_name (str): 新区域的名称
            coordinates (list of tuple): 区域的顶点坐标列表，格式为 [(x1, y1), (x2, y2), ...]
        """
        self.regions[region_name] = np.array(coordinates, dtype=np.int32)

    def draw_regions(self, frame):
        """
        在画面中绘制区域（调试用）

        Args:
            frame (ndarray): 视频帧

        Returns:
            frame
        """
        for name, polygon in self.regions.items():
            cv2.polylines(
                frame,
                [polygon],
                isClosed=True,
                color=(0, 255, 0),
                thickness=2
            )

            # 在区域中心写区域名
            center = polygon.mean(axis=0).astype(int)
            cv2.putText(
                frame,
                name,
                tuple(center),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        return frame
