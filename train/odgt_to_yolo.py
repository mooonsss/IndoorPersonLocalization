import os
import json
from PIL import Image


def load_func(fpath):
    """加载并解析 ODGT 文件"""
    assert os.path.exists(fpath), f"File not found: {fpath}"
    with open(fpath, 'r', encoding='utf-8') as fid:
        lines = fid.readlines()
    records = [json.loads(line.strip('\n')) for line in lines]
    return records


def convert_crowdhuman_odgt_to_txt(odgt_path, output_dir, images_base_dir=None):
    """
    将 CrowdHuman ODGT 标注转换为 YOLO 格式

    Args:
        odgt_path: ODGT 文件路径
        output_dir: 输出目录
        images_base_dir: 图片基础目录（如果不指定，则假设在 ODGT 同级的 images 文件夹）
    """
    os.makedirs(output_dir, exist_ok=True)

    # 创建类别文件
    with open(os.path.join(output_dir, 'classes.txt'), 'w', encoding='utf-8') as f:
        f.write('person\n')

    # 加载标注
    records = load_func(odgt_path)

    # 用于缓存图片尺寸
    img_size_cache = {}

    for record in records:
        image_id = record['ID']
        txt_filename = f"{image_id}.txt"
        txt_path = os.path.join(output_dir, txt_filename)

        # 构建图片路径
        if images_base_dir is None:
            # 默认假设：odgt文件所在目录的images子文件夹
            images_dir = os.path.join(os.path.dirname(odgt_path), "images")
        else:
            images_dir = images_base_dir

        img_path = os.path.join(images_dir, f"{image_id}.jpg")

        # 获取图片尺寸（使用缓存）
        if image_id not in img_size_cache:
            if not os.path.exists(img_path):
                print(f"警告：图片不存在，跳过 {image_id}: {img_path}")
                continue
            try:
                with Image.open(img_path) as img:
                    img_width, img_height = img.size
                    img_size_cache[image_id] = (img_width, img_height)
            except Exception as e:
                print(f"警告：无法打开图片 {img_path}: {e}")
                continue
        else:
            img_width, img_height = img_size_cache[image_id]

        # 处理每个标注框
        with open(txt_path, 'w', encoding='utf-8') as f:
            for bbox in record.get('gtboxes', []):
                # 跳过mask和忽略的框
                if bbox.get('tag') == 'mask':
                    continue
                if bbox.get('extra', {}).get('ignore', 0) == 1:
                    continue

                # 获取全身框（fbox）
                fbox = bbox.get('fbox')
                if not fbox or len(fbox) != 4:
                    continue

                x, y, w, h = fbox

                # 转换为YOLO格式并归一化
                x_center = (x + w / 2) / img_width
                y_center = (y + h / 2) / img_height
                w_norm = w / img_width
                h_norm = h / img_height

                # 写入文件（类别ID 0 表示person）
                f.write(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")


def main():
    # 配置路径（请根据实际情况修改）
    odgt_path = 'annotation_train.odgt'
    output_dir = '/dataset/labels/val'

    # 可选的：如果图片在另一个目录，指定这里
    images_dir = '/dataset/images/val'

    try:
        convert_crowdhuman_odgt_to_txt(
            odgt_path=odgt_path,
            output_dir=output_dir,
            images_base_dir=images_dir  # 可选参数
        )
        print(f"转换完成！标签文件保存在: {output_dir}")

        # 统计信息
        num_txt_files = len([f for f in os.listdir(output_dir) if f.endswith('.txt')])
        print(f"共生成 {num_txt_files} 个标签文件")

    except Exception as e:
        print(f"转换过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()