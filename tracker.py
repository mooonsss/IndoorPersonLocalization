"""
tracker.py
--------------------------------
功能：
    追踪人员的位置，并识别“进入房间”事件。
    在每一帧之间判断人员是否从一个区域进入另一个区域。

说明：
    - 使用 ID 来追踪每个人的位置。
    - 如果一个人在上一帧位于区域 A，当前帧进入区域 B，则标记为进入事件。
"""

from deep_sort_realtime.deepsort_tracker import DeepSort    # 导入 DeepSORT
from collections import deque
from collections import Counter


class PersonTracker:
    def __init__(self, max_age=30, min_hits=3):
        """
        初始化人员追踪器
        使用 DeepSORT 来追踪每个独立人物的 ID 和其在不同帧中的位置
        """
        self.deepsort = DeepSort(max_age=30,
                                n_init=3,
                                embedder="mobilenet",       # 或 "clip_RN50" 等轻量模型
                                half=True,                  # 半精度加速
                                embedder_gpu=True,          # 强制使用 GPU
                                nms_max_overlap=0.5,        # 调整 NMS 最大重叠度
                                max_cosine_distance=0.3     # 可根据实际效果调整
        )
        self.max_age = max_age  # 最大允许不出现的帧数
        self.min_hits = min_hits  # 最少匹配的帧数
        self.entered_regions = {}  # 记录每个 ID 的“进入事件”
        self.region_history = {}  # person_id -> deque of regions
        self.confirm_frames = 3  # 需要连续3帧确认

    def update_tracking(self, frame, frame_detections, region_manager):
        """
        更新人员追踪信息，并检查是否进入新区域

        Args:
            frame (ndarray): 当前帧的图像数据
            frame_detections (list): 当前帧的检测结果，每个检测包括 'bbox' 和 'score'
            region_manager (RegionManager): 区域管理器，用来判断每个人在什么区域

        Returns:
            entered_events (list): 记录所有“进入事件”，格式: [(person_id, from_region, to_region)]
        """
        entered_events = []

        # Step 1: 使用 DeepSORT 进行目标追踪
        detections = []
        for detection in frame_detections:
            bbox = detection['bbox']
            score = detection['score']
            class_id = detection.get('class_id', 0)

            # 转换: [x1, y1, x2, y2] -> [x1, y1, width, height]
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            bbox_xywh = [x1, y1, width, height]  # DeepSORT 要求的格式
            detections.append((bbox_xywh, score, class_id))

        # 获取 DeepSORT 追踪结果，传递 frame 图像和检测结果
        tracks = self.deepsort.update_tracks(detections, frame=frame)

        # Step 2: 根据 DeepSORT 的追踪结果判断人员是否进入新区域
        for track in tracks:
            person_id = track.track_id
            bbox = track.to_tlbr()
            current_region = region_manager.locate_person(bbox)

            # 维护历史队列
            if person_id not in self.region_history:
                self.region_history[person_id] = deque(maxlen=self.confirm_frames)
            self.region_history[person_id].append(current_region)

            # 获取稳定区域（取队列中出现次数最多的区域）
            counter = Counter(self.region_history[person_id])
            stable_region = counter.most_common(1)[0][0]

            # 若稳定区域发生变化且与原记录不同，才触发事件
            previous_region = self.entered_regions.get(person_id)
            if previous_region is None:
                # 首次出现：仅记录区域，不触发事件
                self.entered_regions[person_id] = stable_region
            elif previous_region != stable_region:
                # 区域真正发生变化，记录事件
                entered_events.append((person_id, previous_region, stable_region))
                self.entered_regions[person_id] = stable_region

        return entered_events
