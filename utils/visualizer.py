import cv2
import numpy as np
from collections import defaultdict
import datetime
from utils import config


class SystemVisualizer:
    def __init__(self, map_path, max_trail_length=50, log_path=config.LOG_PATH):
        """
        初始化可视化工具

        Args:
            map_path (str): 2D 室内平面图的路径 (如 'data/floor_plans/room_map.png')
            max_trail_length (int): 尾迹/轨迹线的最大长度（保存多少个历史点）
            log_path (str): 日志文件保存路径
        """
        # 读取原始平面图
        self.raw_map = cv2.imread(map_path)
        if self.raw_map is None:
            raise FileNotFoundError(f"找不到地图文件: {map_path}")

        # 使用字典记录每个 ID 的历史映射坐标，用于画轨迹线
        # 格式: { id1: [(X1, Y1), (X2, Y2), ...], id2: [...] }
        self.trajectories = defaultdict(list)
        self.max_trail_length = max_trail_length

        # 用于状态追踪和日志记录
        self.log_path = log_path
        self.last_zone_status = defaultdict(lambda: "General_Area")

        # 初始化日志文件，写入启动时间
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"\n--- 系统启动时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")

    def _log_event(self, message):
        """
        内部方法：同步将事件记录到控制台和本地记事本

        Args:
            message (str): 需要记录的文本内容
        """
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        log_entry = f"[{timestamp}] {message}"
        print(f" {log_entry}")
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(log_entry + "\n")

    def draw_results(self, frame, tracked_people, mapper, zones=None):
        """
        在视频帧和地图上绘制结果，并检测人员是否在区域间移动

        Args:
            frame (ndarray): 当前视频帧
            tracked_people (list): 包含 ID、检测框和脚底坐标的列表
            mapper (CoordinateMapper): 坐标转换工具实例
            zones (dict, optional): 电子围栏区域配置

        Returns:
            ndarray: 拼接后的可视化结果图像
        """
        # 每次都拷贝一份干净的地图，防止上一帧画的东西残留
        current_map = self.raw_map.copy()
        # 记录当前帧出现的所有 ID，用于后续清理消失的 ID 轨迹
        current_ids = []

        # 绘制背景区域
        if zones is not None:
            for zone_name, zone_data in zones.items():
                polygon = zone_data["polygon"]
                color = zone_data["color"]

                # 1. 画出多边形轮廓线
                cv2.polylines(current_map, [polygon], True, color, 2)

                # 2. 标上区域名字
                # 获取多边形的第一个点作为文字起点的参考 (x, y)
                # 因为你的区域顶部靠近画面边缘 (y=10)，我们将文字往下偏移 25 个像素，往右偏移 10 个像素，防止出界
                text_x = polygon[0][0] + 10
                text_y = polygon[0][1] + 25

                cv2.putText(current_map, zone_name, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        for person in tracked_people:
            # 提取信息
            x1, y1, x2, y2 = person["bbox"]
            pid = person["id"]
            current_ids.append(pid)

            # 映射坐标
            map_x, map_y = mapper.pixel_to_map(person["foot_point"])

            # 区域检测逻辑
            current_zone = "General_Area"
            person_color = (0, 255, 0)  # 默认人的颜色是绿色 (安全)
            if zones is not None:
                for zone_name, zone_data in zones.items():
                    # 使用 cv2.pointPolygonTest 判断坐标点是否在多边形内部
                    # 返回值 >= 0 表示点在多边形内部或边缘上
                    if cv2.pointPolygonTest(zone_data["polygon"], (map_x, map_y), False) >= 0:
                        current_zone = zone_name
                        person_color = zone_data["color"]   # 人的框变成警报区的颜色
                        break   # 只要进了一个危险区，就触发报警并跳出循环

            # 状态转换检测：判断是否从区域 A 进入了区域 B
            last_zone = self.last_zone_status[pid]
            if current_zone != last_zone:
                self._log_event(f"ID {pid}: 从 {last_zone} 进入了 {current_zone}")
                self.last_zone_status[pid] = current_zone

            # 绘制逻辑
            cv2.rectangle(frame, (x1, y1), (x2, y2), person_color, 2)   # 画边界框 (根据是否报警改变颜色)
            cv2.putText(frame, f"ID: {pid}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, person_color, 2) # 标出 ID

            # 在右侧地图上画轨迹和点
            # 把当前点加入该 ID 的轨迹历史中
            self.trajectories[pid].append((map_x, map_y))
            # 限制轨迹长度，避免线太长把屏幕画花
            if len(self.trajectories[pid]) > self.max_trail_length:
                self.trajectories[pid].pop(0)
            # 画出历史轨迹线
            if len(self.trajectories[pid]) > 1:
                pts = np.array(self.trajectories[pid], np.int32).reshape((-1, 1, 2))
                cv2.polylines(current_map, [pts], False, (255, 0, 0), 2)
            # 画出当前位置的实心圆点 (根据是否报警改变颜色)
            cv2.circle(current_map, (map_x, map_y), 8, person_color, -1)
            # 坐标点设置为 (map_x + 10, map_y - 10)，让文字出现在圆点的右上方
            cv2.putText(current_map, f"ID: {pid}", (map_x + 10, map_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, person_color, 2)

        # 清理已经离开画面的人的轨迹
        for k in list(self.trajectories.keys()):
            if k not in current_ids:
                del self.trajectories[k]
                if k in self.last_zone_status: del self.last_zone_status[k]

        return frame, current_map

