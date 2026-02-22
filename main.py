import os
import cv2
from utils.detect import PersonDetector
from utils.region import RegionManager
from utils.tracker import PersonTracker

def process_video(video_path, output_video_path, result_file):
    # Step 1: 初始化相关模块
    detector = PersonDetector(model_path="model/yolov8s.pt", conf_thresh=0.4)
    region_manager = RegionManager()
    tracker = PersonTracker()

    # Step 2: 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video.")
        return

    # 获取视频帧率和尺寸
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 创建输出视频文件
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # 确保输出目录存在
    os.makedirs(os.path.dirname(result_file), exist_ok=True)

    # 以覆盖写入模式打开结果文件（若需追加可改为 'a'）
    result_f = open(result_file, 'w', encoding='utf-8')

    # Step 3: 主线程处理视频
    while True:
        # 1. 读取视频帧
        ret, frame = cap.read()
        if not ret:
            print("视频播放完毕")
            break

        # 2. 目标检测
        detections = detector.detect_frame(frame)

        # 3. 更新追踪器
        entered_events = tracker.update_tracking(frame, detections, region_manager)

        # 4. 可视化绘制
        region_manager.draw_regions(frame)
        for detection in detections:
            bbox = detection['bbox']
            region = region_manager.locate_person(bbox)
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{region}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


        # 5. 显示视频帧
        cv2.imshow("Frame with Regions", frame)

        # 6. 保存每一帧到输出视频
        out.write(frame)

        # 7. 记录进入事件
        for event in entered_events:
            person_id, from_region, to_region = event
            event_str = f"Person {person_id} entered from {from_region} to {to_region}\n"
            result_f.write(event_str)
            # 同时打印到控制台
            print(event_str, end='')

        # 8. 按键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 清理资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    result_f.close()  # 关闭结果文件


if __name__ == "__main__":
    video_path = "videos/003.avi"  # 输入你的视频路径
    output_video_path = "outputs/output_videos.mp4"  # 输出视频路径
    result_file = "outputs/result.txt"  # 输出结果文件路径
    process_video(video_path, output_video_path, result_file)