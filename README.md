基于 YOLOv8 与单应性变换的室内人员定位追踪系统
📖 项目简介
本项目是一个集成了目标检测、多目标追踪、地理坐标映射及电子围栏报警功能的智能化视觉监控系统。
针对传统安防系统算力开销大、定位轨迹抖动等痛点，本项目在架构与算法上进行了深度优化：

端到端极速追踪：摒弃传统 DeepSORT 繁重的特征提取网络，采用 YOLOv8 内置 ByteTrack 算法。通过低分框二次关联机制，在提升抗遮挡能力的同时，实现了推理帧率（FPS）的大幅提升。

高精度映射：通过单应性变换（Homography Matrix）将视频像素坐标实时、精准地映射至 2D 室内平面图。

轨迹去噪优化：在后处理阶段“头部锚定与几何约束”逻辑，利用指数移动平均（EMA）维护目标物理尺寸，一定程度上消除了行人步态带来的 2D 坐标高频抖动。

电子围栏安防：结合 Point-in-Polygon 算法，实现危险区域入侵的实时变色与日志报警。

🛠️ 环境准备
Python 版本：推荐使用 Python 3.8 或更高版本。

⚙️ 快速上手
在正式运行前，请根据你的本地路径修改 utils/config.py 文件：

配置路径：设置 VIDEO_PATH（视频源）、MAP_PATH（平面图）和 MODEL_PATH（权重文件）。

坐标标定：运行get_point.py,确保 SOURCE_POINTS 与 DESTINATION_POINTS 对应视频和地图中的四个相同物理参考点。

📂 项目结构说明
IndoorPersonLocalization/
├── core/
│   ├── detector.py      # 核心视觉引擎 (集成 YOLOv8s 检测 + ByteTrack 追踪)
│   ├── tracker.py       # 轨迹平滑模块 (执行头部锚定与 EMA 几何尺寸约束)
│   └── mapper.py        # 空间映射引擎 (单应性矩阵变换)
├── utils/
│   ├── config.py        # 全局参数与语义区域 (Geofencing Zones) 配置
│   ├── get_point.py     # 辅助工具：交互式坐标采集脚本 (用于获取标定点坐标)
│   └── visualizer.py    # 结果可视化与跨区域事件日志生成
├── data/
│   ├── floor_plans/     # 存放室内平面图素材
│   └── videos/          # 存放测试视频
├── weights/             # 存放训练好的 .pt 模型权重
├── outputs/             # 存放运行生成的视频结果与 alarm.log.txt 报警日志
├── main.py              # 系统主入口
├── requirements.txt     # 项目依赖库列表
└── README.md            # 项目说明文档
⚠️ 注意事项
模型选择：默认使用 YOLOv8s 权重，若需追求更高的部署实时性，可在 config.py 中更换为 yolov8n.pt。

日志生成：系统运行时，若人员跨越设定的语义区域（如进入危险区或离开画面），会自动在 outputs/alarm.log.txt 中生成带有时间戳的审计日志。
