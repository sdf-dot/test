import json
import os

# 标签映射（根据实际标注修改）
name2id = {'已投放': 0, '未投放': 1}  # 示例标签，需与LabelMe标注一致


def polygon_to_bbox(points):
    """计算多边形的最小外接矩形（x1,y1,x2,y2）"""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), min(ys), max(xs), max(ys))  # 左上、右下坐标


def rectangle_to_bbox(points):
    """将矩形标注（两个点）转换为边界框（x1,y1,x2,y2）"""
    x1, y1 = points[0]  # 左上角点
    x2, y2 = points[1]  # 右下角点
    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


def convert_to_yolo(img_size, bbox):
    """将包围盒转换为YOLO归一化坐标（x_center, y_center, width, height）"""
    x1, y1, x2, y2 = bbox
    dw, dh = 1.0 / img_size[0], 1.0 / img_size[1]
    x = (x1 + x2) / 2.0 * dw
    y = (y1 + y2) / 2.0 * dh
    w = (x2 - x1) * dw
    h = (y2 - y1) * dh
    return (x, y, w, h)


def process_json(json_path, output_dir):
    """处理单个JSON文件，生成YOLO标注"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    img_w, img_h = data['imageWidth'], data['imageHeight']
    txt_path = os.path.join(output_dir, os.path.basename(json_path)[:-5] + '.txt')

    with open(txt_path, 'w') as txt_file:
        for shape in data['shapes']:
            shape_type = shape['shape_type']
            label = shape['label']

            # 检查标签是否存在
            if label not in name2id:
                print(f"警告：跳过未知标签 '{label}' in {json_path}")
                continue

            # 根据形状类型处理标注
            if shape_type == 'polygon':
                bbox = polygon_to_bbox(shape['points'])
            elif shape_type == 'rectangle':
                bbox = rectangle_to_bbox(shape['points'])
            else:
                print(f"警告：跳过不支持的形状类型 '{shape_type}' in {json_path}")
                continue

            yolo_coords = convert_to_yolo((img_w, img_h), bbox)
            txt_file.write(f"{name2id[label]} {' '.join(f'{v:.6f}' for v in yolo_coords)}\n")

    # 清理空文件（无有效标注时）
    if os.path.getsize(txt_path) == 0:
        os.remove(txt_path)
        print(f"删除空文件：{txt_path}")


if __name__ == "__main__":
    # 使用原始字符串避免反斜杠转义问题
    json_dir = r"D:\\垃圾桶2\\垃圾桶2"  # LabelMe JSON文件目录
    output_dir = r"C:\\yolov5\\yolov5-master\\my_dates\\labels\\train"  # YOLO标注输出目录

    os.makedirs(output_dir, exist_ok=True)

    for json_file in os.listdir(json_dir):
        if json_file.endswith('.json'):
            process_json(os.path.join(json_dir, json_file), output_dir)

    print("转换完成！")