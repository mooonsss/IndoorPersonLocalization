from ultralytics import YOLO
import cv2
import torch

torch.cuda.empty_cache()

# 1. 加载预训练模型（用yolov8s.pt）
model = YOLO('/root/autodl-tmp/IndoorPersonLocalization/model/yolov8s.pt')

# 2. 执行训练（参数根据你的数据集和设备调整）
# 2. 优化后的训练参数
results = model.train(
    data='crowdhuman.yaml',  # 数据集配置文件
    epochs=200,  # 训练轮数

    # 显存优化核心参数
    batch=16,  # 从16减少到12（更安全），RTX 4090可以尝试12-16
    imgsz=640,  # 图像尺寸，640是安全选择
    rect=True,  # 矩形训练，提升效率

    # 内存优化
    cache='ram',  # 改为False，缓存会占用大量RAM（90GB也容易不够）
    amp=True,  # 混合精度训练

    # 针对密集目标优化的参数
    max_det=300,  # 增加最大检测数量
    mask_ratio=4,  # 保持默认
    overlap_mask=True,  # 保持默认

    # 学习率调整
    lr0=0.01,  # 初始学习率
    lrf=0.01,  # 最终学习率
    warmup_epochs=3,  # 添加学习率热身
    warmup_momentum=0.8,  # 热身期动量
    warmup_bias_lr=0.1,  # 偏置参数学习率

    # 优化器配置
    optimizer='SGD',  # SGD
    momentum=0.937,  # SGD动量
    weight_decay=0.0005,  # 权重衰减

    # 损失权重（针对密集目标调整）
    box=0.05,  # 降低box loss权重
    cls=0.3,  # 降低cls loss权重
    dfl=1.5,  # 保持DFL loss

    # 数据增强优化（针对密集行人）
    augment=True,
    mosaic=0.5,  # 降低mosaic概率到50%，避免目标过小
    mixup=0.0,  # 关闭mixup（密集场景可能不合适）
    copy_paste=0.0,  # 关闭copy-paste
    degrees=10.0,  # 改为10度旋转增强
    translate=0.1,  # 增加平移增强
    scale=0.5,  # 尺度增强
    shear=0.0,  # 添加剪切变换（设为0或小值）
    perspective=0.001,  # 透视变换
    flipud=0.0,
    fliplr=0.5,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    erasing=0.0,  # 随机擦除，可以设为0或小值

    # 系统参数
    device=0,
    workers=8,  # 减少到8个worker（20核CPU，留资源给系统）
    patience=50,  # 减少早停耐心值到50（更早检测过拟合）

    # 训练控制
    val=True,
    save=True,
    save_period=10,
    pretrained=True,
    verbose=True,
    seed=42,
    deterministic=True,

    # 项目配置
    project='crowdhuman_train',
    name='yolov8m_crowdhuman_optimized',
    exist_ok=True,

    # 高级参数
    nbs=64,  # 名义batch size
    close_mosaic=10,  # 最后10个epoch关闭mosaic
    label_smoothing=0.1,  # 标签平滑，防止过拟合
    cos_lr=False,  # 不启用余弦学习率调度
    dropout=0.0,  # 无dropout
)

# 3. 模型评估
val_results = model.val()
print(f"自定义数据集训练完成！")
print(f"mAP@0.5: {val_results.box.map50:.4f}")  # IoU=0.5时的mAP
print(f"mAP@0.5:0.95: {val_results.box.map:.4f}")  # IoU从0.5到0.95的平均mAP

# 4. 模型预测
model = YOLO('custom_train/yolov8n_custom/weights/best.pt')  # 加载最佳模型
test_img_path = 'custom_test.jpg'  # 自定义测试图像路径
results = model(test_img_path, conf=0.25)
annotated_img = results[0].plot()
cv2.imshow('Custom Dataset Detection', annotated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('custom_result.jpg', annotated_img)
print("自定义数据集预测结果已保存为custom_result.jpg")
