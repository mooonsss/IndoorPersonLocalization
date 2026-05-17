import cv2
import time
import os
from core.detector import PersonDetector
from core.tracker import PersonTracker
from core.mapper import CoordinateMapper
from utils.visualizer import SystemVisualizer
from utils import config


def main():
    """
    主程序入口：控制检测、跟踪、映射、报警及视频保存的完整流程
    """
    # 确保输出目录存在
    if not os.path.exists(config.OUTPUT_PATH):
        os.makedirs(config.OUTPUT_PATH)
        print("已创建 outputs 文件夹")

    # 初始化所有模块
    detector = PersonDetector(config.MODEL_PATH, config.CONF_THRESH)
    tracker = PersonTracker()
    mapper = CoordinateMapper(config.SOURCE_POINTS, config.DESTINATION_POINTS)
    visualizer = SystemVisualizer(config.MAP_PATH, config.MAX_TRAIL)

    # 加载视频
    cap = cv2.VideoCapture(config.VIDEO_PATH)

    # 获取视频的基本信息
    fps_original = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps_original if fps_original > 0 else 0
    print(f"视频原始帧率: {fps_original:.2f} FPS")
    print(f"总帧数: {total_frames}")
    print(f"视频时长: {duration:.2f} 秒")

    # 视频保存初始化
    video_writer = None
    output_path = "outputs/result_video.mp4"

    # 帧率计算变量
    frame_count = 0
    start_time = time.time()
    prev_time = start_time

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 记录处理开始时间
        process_start_time = time.time()

        # 第一步：检测与初步跟踪 (ByteTrack)
        detections = detector.detect_and_track_frame(frame)

        # 第二步：平滑与脚底坐标推算
        # 不再需要把 frame 传进去给 DeepSORT 切图 大大节省了时间
        tracked_people = tracker.track_frame(detections)

        # 第三步：映射与可视化
        # visualizer 内部会调用 mapper.pixel_to_map
        frame, current_map = visualizer.draw_results(frame, tracked_people, mapper, config.ZONES)

        # 计算当前帧的处理时间
        process_end_time = time.time()
        frame_process_time = process_end_time - process_start_time

        # 更新帧计数
        frame_count += 1

        # 计算实时帧率
        current_time = time.time()
        elapsed_time = current_time - prev_time
        real_time_fps = 1.0 / elapsed_time if elapsed_time > 0 else 0
        avg_fps = frame_count / (current_time - start_time) if (current_time - start_time) > 0 else 0

        # 在帧上显示实时帧率信息
        cv2.putText(frame, f'Real-time FPS: {real_time_fps:.2f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Average FPS: {avg_fps:.2f}', (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Process Time: {frame_process_time * 1000:.2f}ms', (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        '''
        # 同样在地图窗口也显示FPS信息
        cv2.putText(current_map, f'Real-time FPS: {real_time_fps:.2f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(current_map, f'Average FPS: {avg_fps:.2f}', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        '''
        # 第四步：第一次循环时初始化 VideoWriter
        if video_writer is None:
            height, width = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4 格式
            video_writer = cv2.VideoWriter(output_path, fourcc, min(30, fps_original), (width, height))  # 限制最大保存帧率为30
            print(f"视频将保存至: {output_path}")
            print(f"实际处理帧率: {real_time_fps:.2f} FPS (瞬时), {avg_fps:.2f} FPS (平均)")

        # 第四步：写入帧
        video_writer.write(frame)

        # 第五步：展示
        cv2.imshow("Indoor Positioning System - Demo", frame)
        cv2.imshow("Tracking", current_map)

        # 更新前一帧时间
        prev_time = current_time

        # 按 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 计算并显示最终统计信息
    final_avg_fps = frame_count / (time.time() - start_time)
    processing_duration = time.time() - start_time
    print(f"\n=== 处理完成 ===")
    print(f"总处理帧数: {frame_count}")
    print(f"总处理时间: {processing_duration:.2f} 秒")
    print(f"平均处理帧率: {final_avg_fps:.2f} FPS")
    print(f"原始视频帧率: {fps_original:.2f} FPS")
    print(f"性能效率: {final_avg_fps / fps_original * 100:.1f}%")

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
