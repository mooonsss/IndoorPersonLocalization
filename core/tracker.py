

class PersonTracker:
    def __init__(self):
        """
        轨迹平滑与锚点提取器
        由于追踪逻辑已交由 YOLOv8 底层 ByteTrack 处理，
        本类的核心职责是：利用历史物理尺寸，推算极其稳定的 2D 映射脚底坐标。
        """

        # 保存每个 ID 的“物理尺寸”历史
        # 格式: { track_id: (stable_width, stable_height) }
        self.body_sizes = {}

        # 保存每个 ID 的平滑后的左上角坐标
        # 格式: { track_id: (x1_smooth, y1_smooth) }
        self.positions = {}

    def track_frame(self, detections):
        """
        处理检测与追踪结果，计算平滑脚底坐标

        Args:
            detections (list): detector 返回的包含 ID 和 bbox 的列表

        Returns:
            tracked_objects (list): 增加了稳定 foot_point 的最终列表
        """
        tracked_objects = []
        current_active_ids = []

        for det in detections:
            track_id = det["id"]
            x1, y1, x2, y2 = det["bbox"]
            current_active_ids.append(track_id)

            # 当前宽高
            current_w = x2 - x1
            current_h = y2 - y1

            # 平滑左上角坐标
            if track_id in self.positions:
                prev_x1, prev_y1 = self.positions[track_id]
                # 平滑系数 0.15，与宽高一致
                x1_smooth = 0.15 * x1 + 0.85 * prev_x1
                y1_smooth = 0.15 * y1 + 0.85 * prev_y1
            else:
                x1_smooth, y1_smooth = x1, y1
            self.positions[track_id] = (x1_smooth, y1_smooth)

            # 平滑宽高
            if track_id in self.body_sizes:
                prev_w, prev_h = self.body_sizes[track_id]
                stable_w = 0.15 * current_w + 0.85 * prev_w
                stable_h = 0.15 * current_h + 0.85 * prev_h
            else:
                stable_w, stable_h = current_w, current_h
            self.body_sizes[track_id] = (stable_w, stable_h)

            # 使用平滑后的值推算脚底点
            stable_foot_x = int(x1_smooth + stable_w / 2)
            stable_foot_y = int(y1_smooth + stable_h)

            tracked_objects.append({
                "id": track_id,
                "bbox": [x1, y1, x2, y2],  # 原始检测框（用于显示）
                "foot_point": [stable_foot_x, stable_foot_y]
            })

        # 清理已离开画面的人的缓存
        active_set = set(current_active_ids)
        self.body_sizes = {k: v for k, v in self.body_sizes.items() if k in active_set}
        self.positions = {k: v for k, v in self.positions.items() if k in active_set}

        return tracked_objects