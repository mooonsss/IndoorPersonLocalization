from ultralytics import YOLO


class PersonDetector:
    def __init__(self, model_path, conf_thresh=0.4):
        """
        初始化人员检测器(基于 YOLOv8 内置 ByteTrack)

        Args:
            model_path (str): YOLOv8 模型路径
            conf_thresh (float): 置信度阈值
        """
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh

    def detect_and_track_frame(self, frame):
        """
        对单帧图像进行人员检测与 ByteTrack 追踪

        Args:
            frame (ndarray): 输入的视频帧

        Returns:
            detections (list): 带有 ID 和检测框的字典列表
        """
        # 使用 model.track 代替 model.predict
        # persist=True 表示在视频流中保持轨迹连续性
        # tracker="bytetrack.yaml" 指定使用 ByteTrack 算法 (默认为 BoT-SORT)
        results = self.model.track(frame, persist=True, conf=self.conf_thresh,
                                   classes=[0], tracker="bytetrack.yaml", verbose=False)

        detections = []

        for r in results:
            # 如果没有检测到人，或者还没有分配 ID，直接跳过
            if r.boxes is None or len(r.boxes) == 0 or r.boxes.id is None:
                continue

            # 将数据统一拉取到 CPU 并转为 numpy，方便后续 tracker 或 opencv 处理
            boxes = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            ids = r.boxes.id.cpu().numpy()  # 获取 ByteTrack 分配的 ID

            for box, score, obj_id in zip(boxes, scores, ids):
                x1, y1, x2, y2 = box
                detections.append({
                    "id": int(obj_id),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "score": float(score)
                })

        return detections