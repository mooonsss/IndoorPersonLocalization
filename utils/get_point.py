import cv2

# 用来存放鼠标点击的坐标
points = []


def click_event(event, x, y, flags, param):
    global points
    # 监听鼠标左键点击事件
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])
        print(f"记录第 {len(points)} 个点: [{x}, {y}]")

        # 在点击的位置画一个红色的圆点
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

        # 连线：如果已经点了不止一个点，就把当前点和上一个点连起来
        if len(points) > 1:
            cv2.line(frame, tuple(points[-2]), tuple(points[-1]), (0, 255, 0), 2)

        # 当点了 4 个点时，闭合多边形并打印最终结果
        if len(points) == 4:
            cv2.line(frame, tuple(points[-1]), tuple(points[0]), (0, 255, 0), 2)
            print("\n" + "=" * 40)
            print("提取完成！请将以下内容复制到你的config.py中：")
            print(f"SOURCE_POINTS = {points}")
            print("=" * 40 + "\n")
            print("你可以按任意键关闭窗口。")

        cv2.imshow("Click 4 Points (Press any key to exit)", frame)


# 1. 填入你的视频路径 (或者换成单张图片的路径)
VIDEO_PATH = "C:/Users/Moon/Desktop/IndoorPersonLocalization/data/floor_plans/test.png"  # 替换成你下载的测试视频路径

# 读取视频的第一帧
cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()
cap.release()  # 读完第一帧就释放视频
if not ret:
    print(f"无法读取视频: {VIDEO_PATH}，请检查路径是否正确！")
    exit()
print("请在弹出的窗口中，依次点击 4 个顶点 (例如: 左上 -> 右上 -> 右下 -> 左下)")

# 显示图像并绑定鼠标事件
cv2.imshow("Click 4 Points (Press any key to exit)", frame)
cv2.setMouseCallback("Click 4 Points (Press any key to exit)", click_event)

# 等待按键，按任意键退出
cv2.waitKey(0)
cv2.destroyAllWindows()
