# fix_annotations.py
import os
from pathlib import Path
import cv2


def fix_yolo_annotations(data_dir):
    """
    修复YOLO格式标注文件的坐标超出问题
    """
    # 设置路径
    images_dir = Path(data_dir) / 'images/val'
    labels_dir = Path(data_dir) / 'labels/val'

    # 获取所有标注文件
    label_files = list(labels_dir.glob('*.txt'))

    print(f"找到 {len(label_files)} 个标注文件")

    fixed_count = 0
    removed_count = 0

    for label_file in label_files:
        # 对应的图像文件
        img_file = images_dir / f"{label_file.stem}.jpg"

        if not img_file.exists():
            # 尝试其他扩展名
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                img_file = images_dir / f"{label_file.stem}{ext}"
                if img_file.exists():
                    break

        if not img_file.exists():
            print(f"警告: 找不到图像文件 {img_file}")
            continue

        # 读取图像获取尺寸
        img = cv2.imread(str(img_file))
        if img is None:
            print(f"警告: 无法读取图像 {img_file}")
            continue

        img_h, img_w = img.shape[:2]

        # 读取标注文件
        with open(label_file, 'r') as f:
            lines = f.readlines()

        fixed_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 5:  # YOLO格式: class x_center y_center width height
                continue

            cls_id = parts[0]
            try:
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
            except ValueError:
                continue

            # 检查坐标是否超出边界
            if any(coord < 0 or coord > 1 for coord in [x_center, y_center, width, height]):
                # 修复坐标：截断到 [0, 1]
                x_center = max(0.0, min(1.0, x_center))
                y_center = max(0.0, min(1.0, y_center))
                width = max(0.0, min(1.0, width))
                height = max(0.0, min(1.0, height))

                # 检查修复后的边界框是否有效
                if width > 0 and height > 0:
                    fixed_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                    fixed_count += 1
                else:
                    removed_count += 1
                    print(f"移除无效标注: {label_file.name} - 坐标: {x_center}, {y_center}, {width}, {height}")
            else:
                fixed_lines.append(line)

        # 保存修复后的标注
        if fixed_lines:
            with open(label_file, 'w') as f:
                f.write('\n'.join(fixed_lines))
        else:
            # 如果修复后没有有效标注，删除文件
            os.remove(label_file)
            removed_count += len(lines)
            print(f"删除无有效标注的文件: {label_file.name}")

    print(f"\n修复完成!")
    print(f"修复了 {fixed_count} 个标注")
    print(f"移除了 {removed_count} 个无效标注")

    # 验证修复
    verify_annotations(data_dir)


def verify_annotations(data_dir):
    """验证标注文件"""
    labels_dir = Path(data_dir) / 'labels/train'
    label_files = list(labels_dir.glob('*.txt'))

    print(f"\n验证 {len(label_files)} 个标注文件...")

    error_files = []

    for label_file in label_files:
        with open(label_file, 'r') as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 5:
                error_files.append((label_file.name, f"第{i + 1}行: 格式错误"))
                continue

            try:
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
            except ValueError:
                error_files.append((label_file.name, f"第{i + 1}行: 数值转换错误"))
                continue

            # 检查坐标范围
            for coord, name in [(x_center, 'x_center'), (y_center, 'y_center'),
                                (width, 'width'), (height, 'height')]:
                if coord < 0 or coord > 1:
                    error_files.append((label_file.name, f"第{i + 1}行: {name}={coord} 超出范围"))

    if error_files:
        print(f"发现 {len(error_files)} 个错误:")
        for file, error in error_files[:10]:  # 只显示前10个错误
            print(f"  {file}: {error}")
        if len(error_files) > 10:
            print(f"  ... 还有 {len(error_files) - 10} 个错误")
    else:
        print("所有标注文件验证通过!")


def visualize_problematic_annotations(data_dir, num_samples=5):
    """可视化有问题的标注"""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    images_dir = Path(data_dir) / 'images/train'
    labels_dir = Path(data_dir) / 'labels/train'

    label_files = list(labels_dir.glob('*.txt'))

    for label_file in label_files[:num_samples]:
        img_file = images_dir / f"{label_file.stem}.jpg"

        if not img_file.exists():
            continue

        # 读取图像
        img = cv2.imread(str(img_file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_h, img_w = img.shape[:2]

        # 读取标注
        with open(label_file, 'r') as f:
            lines = f.readlines()

        fig, ax = plt.subplots(1, 2, figsize=(15, 6))

        # 原始图像
        ax[0].imshow(img)
        ax[0].set_title(f'原始图像: {img_file.name}')
        ax[0].axis('off')

        # 绘制边界框
        ax[1].imshow(img)
        ax[1].set_title('标注边界框')

        problem_count = 0
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            cls_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            # 转换到像素坐标
            x1 = (x_center - width / 2) * img_w
            y1 = (y_center - height / 2) * img_h
            x2 = (x_center + width / 2) * img_w
            y2 = (y_center + height / 2) * img_h

            # 检查是否超出边界
            color = 'red' if any(coord < 0 or coord > 1 for coord in [x_center, y_center, width, height]) else 'green'

            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                     edgecolor=color, facecolor='none')
            ax[1].add_patch(rect)

            if color == 'red':
                problem_count += 1

        if problem_count > 0:
            ax[1].text(10, 30, f'发现 {problem_count} 个问题标注',
                       bbox=dict(facecolor='red', alpha=0.5), color='white')

        ax[1].axis('off')
        plt.tight_layout()
        plt.savefig(f'annotation_problem_{label_file.stem}.png', dpi=150)
        plt.show()


if __name__ == "__main__":
    # 设置您的数据集路径
    dataset_path = "C:/Users/Moon/Desktop/IndoorPersonLocalization/dataset"

    print("开始修复标注文件...")
    fix_yolo_annotations(dataset_path)

    # 可视化一些有问题的标注（可选）
    # visualize_problematic_annotations(dataset_path, num_samples=3)