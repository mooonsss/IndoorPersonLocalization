from ultralytics import YOLO

class PersonDetector:
    def __init__(self, model_path, conf_thresh=0.4):
        """
        初始化人员检测器

        Args:
            model_path (str): YOLOv8 模型路径
            conf_thresh (float): 置信度阈值
        """
        self.model = YOLO(model_path).cuda()
        self.conf_thresh = conf_thresh

    def detect_frame(self, frame):
        """
        对单帧图像进行人员检测

        Args:
            frame (ndarray): 输入的视频帧

        Returns:
            detections (list): 检测结果，包含 bbox 和置信度
        """
        results = self.model(frame, verbose=False)
        detections = []

        for r in results:
            if r.boxes is None:
                continue

            boxes = r.boxes.xyxy
            classes = r.boxes.cls
            scores = r.boxes.conf

            for box, cls_id, score in zip(boxes, classes, scores):
                if int(cls_id) == 0 and float(score) >= self.conf_thresh:
                    x1, y1, x2, y2 = box.tolist()
                    detections.append({
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "score": float(score)
                    })

        return detections

