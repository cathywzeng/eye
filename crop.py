"""
裁剪 JPG 中的子图 - 基于投影分析 + 连通组件双重方法
对每张 JPG，自动检测其中的小图并裁剪出来保存为单独的文件。
"""

import os
from PIL import Image

WORK_DIR = r"c:/workspace/eye"
OUTPUT_DIR = os.path.join(WORK_DIR, "cropped")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 参数
# ============================================================
LIGHT_THRESH = 235       # 像素 RGB 全部 > 此值视为"背景"
GAP_THRESH = 0.04        # 行/列中暗像素比例 < 此值视为"分隔缝隙"
MIN_SIZE = 50            # 最小子图尺寸 (宽和高都要 >= 此值)
BORDER_PAD = 1           # 裁剪边距

# ============================================================
# 工具函数
# ============================================================

def make_binary(img):
    """
    将图片二值化: 1 = 内容像素 (暗), 0 = 背景 (亮)
    """
    w, h = img.size
    px = img.load()
    binary = [[0] * w for _ in range(h)]
    for y in range(h):
        for x in range(w):
            r, g, b = px[x, y]
            if r < LIGHT_THRESH or g < LIGHT_THRESH or b < LIGHT_THRESH:
                binary[y][x] = 1
    return binary


def find_gaps(projection, total_size, threshold=GAP_THRESH):
    """
    从投影数据中找到分隔缝隙 (gap bands)。
    projection: 数组, 每个元素是暗像素的数量
    total_size: 投影方向的尺寸 (宽或高)
    threshold: 暗像素比例低于此值为缝隙

    返回: [(gap_start, gap_end), ...] — 缝隙区间列表
    """
    gap_positions = []
    for i, count in enumerate(projection):
        ratio = count / total_size
        if ratio < threshold:
            gap_positions.append(i)

    # 合并连续的缝隙位置为区间
    gaps = []
    if not gap_positions:
        return gaps

    start = gap_positions[0]
    end = gap_positions[0]
    for pos in gap_positions[1:]:
        if pos == end + 1:
            end = pos
        else:
            gaps.append((start, end))
            start = pos
            end = pos
    gaps.append((start, end))

    # 过滤掉太窄的缝隙 (< 2px 的可能是噪声)
    gaps = [(s, e) for s, e in gaps if e - s >= 2]
    return gaps


def gaps_to_bands(gaps, total_size):
    """
    从缝隙区间推导出内容区间 (bands)。
    gaps: [(gap_start, gap_end), ...] — 缝隙区间
    total_size: 总尺寸

    返回: [(band_start, band_end), ...] — 内容区间
    """
    bands = []
    prev_end = -1

    for gs, ge in gaps:
        band_start = prev_end + 1
        band_end = gs - 1
        if band_start <= band_end:
            bands.append((band_start, band_end))
        prev_end = ge

    # 最后一个 gap 之后的内容
    if prev_end < total_size - 1:
        bands.append((prev_end + 1, total_size - 1))

    return bands


def get_grid_boxes(w, h, binary):
    """
    基于投影分析找出网格布局。
    返回: [(left, top, right, bottom), ...]
    """
    # --- 列投影: 每列中有多少暗像素 ---
    col_proj = [0] * w
    for y in range(h):
        for x in range(w):
            if binary[y][x]:
                col_proj[x] += 1

    col_gaps = find_gaps(col_proj, h)
    col_bands = gaps_to_bands(col_gaps, w)

    # --- 行投影: 每行中有多少暗像素 ---
    row_proj = [0] * h
    for y in range(h):
        row_proj[y] = sum(binary[y])

    row_gaps = find_gaps(row_proj, w)
    row_bands = gaps_to_bands(row_gaps, h)

    # --- 生成网格框 ---
    boxes = []
    for top, bottom in row_bands:
        for left, right in col_bands:
            bw = right - left + 1
            bh = bottom - top + 1
            if bw >= MIN_SIZE and bh >= MIN_SIZE:
                boxes.append((left, top, right, bottom))

    return boxes


def find_connected_components_binary(binary, w, h):
    """
    BFS 四连通扫描, 从 binary 找内容组件。
    返回: [(left, top, right, bottom), ...]
    """
    visited = [[False] * w for _ in range(h)]
    components = []

    for sy in range(h):
        for sx in range(w):
            if visited[sy][sx] or not binary[sy][sx]:
                visited[sy][sx] = True
                continue

            # BFS
            stack = [(sx, sy)]
            visited[sy][sx] = True
            min_x = max_x = sx
            min_y = max_y = sy

            while stack:
                x, y = stack.pop()
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < h and not visited[ny][nx]:
                        visited[ny][nx] = True
                        if binary[ny][nx]:
                            min_x = min(min_x, nx)
                            max_x = max(max_x, nx)
                            min_y = min(min_y, ny)
                            max_y = max(max_y, ny)
                            stack.append((nx, ny))

            comp_w = max_x - min_x + 1
            comp_h = max_y - min_y + 1
            if comp_w >= MIN_SIZE and comp_h >= MIN_SIZE:
                components.append((min_x, min_y, max_x, max_y))

    return components


def merge_vertical_boxes(boxes, y_thresh=8, x_overlap_ratio=0.5):
    """
    合并垂直方向上相邻、水平方向上有重叠的框。
    用于将子图与其标签/标题合并。

    y_thresh: 垂直间距小于此值则合并
    x_overlap_ratio: 需要重叠比例大于此值
    """
    if not boxes:
        return []

    # 按 top 排序
    sorted_boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    merged = [sorted_boxes[0]]

    for box in sorted_boxes[1:]:
        last = merged[-1]
        l1, t1, r1, b1 = last
        l2, t2, r2, b2 = box

        # 检查水平重叠: 两个框在水平方向上有足够重叠
        overlap = min(r1, r2) - max(l1, l2)
        w1 = r1 - l1 + 1
        w2 = r2 - l2 + 1
        min_w = min(w1, w2)

        h_gap = t2 - b1  # 垂直间距 (正数 = 有间距)

        if overlap > 0 and h_gap >= 0 and h_gap <= y_thresh:
            # 垂直相邻且水平重叠 → 合并
            merged[-1] = (min(l1, l2), t1, max(r1, r2), max(b1, b2))
        else:
            merged.append(box)

    return merged


def split_wide_box(img, box, binary_w, binary_h, binary):
    """
    对非常宽的框（可能包含多个子图未分割）尝试二次切分。
    """
    left, top, right, bottom = box
    bw = right - left + 1
    bh = bottom - top + 1

    # 只在宽高比 > 2 时尝试切分
    if bw <= bh * 2:
        return [box]

    # 在这个框的区域内重新做列投影
    sub_col_proj = [0] * bw
    for y in range(top, bottom + 1):
        for x in range(left, right + 1):
            if y < binary_h and x < binary_w and binary[y][x]:
                sub_col_proj[x - left] += 1

    sub_gaps = find_gaps(sub_col_proj, bh, threshold=0.03)
    sub_bands = gaps_to_bands(sub_gaps, bw)

    sub_boxes = []
    for band_l, band_r in sub_bands:
        cw = band_r - band_l + 1
        if cw >= MIN_SIZE:
            sub_boxes.append((left + band_l, top, left + band_r, bottom))

    # 如果切分后只有 1 块或没有, 返回原框
    if len(sub_boxes) <= 1:
        return [box]

    return sub_boxes


def crop_and_save(img, boxes, base_name):
    """裁剪并保存"""
    results = []
    for idx, (left, top, right, bottom) in enumerate(boxes):
        # 边距
        left = max(0, left - BORDER_PAD)
        top = max(0, top - BORDER_PAD)
        right = min(img.width - 1, right + BORDER_PAD)
        bottom = min(img.height - 1, bottom + BORDER_PAD)

        crop = img.crop((left, top, right + 1, bottom + 1))
        out_name = f"{base_name}_{idx + 1}.png"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        crop.save(out_path, "PNG")
        results.append((out_name, right - left + 1, bottom - top + 1))
    return results


def process_image(filepath):
    """处理单张图片"""
    base = os.path.splitext(os.path.basename(filepath))[0]
    print(f"\n{'='*50}")
    print(f"处理: {os.path.basename(filepath)}")
    print(f"{'='*50}")

    img = Image.open(filepath).convert("RGB")
    w, h = img.size
    print(f"  尺寸: {w}x{h}")

    binary = make_binary(img)

    # --- 方法1: 投影分析法 (适用于规则网格) ---
    grid_boxes = get_grid_boxes(w, h, binary)
    print(f"  投影分析: 找到 {len(grid_boxes)} 个候选区域")

    # --- 方法2: 连通组件法 (适用于不规则布局) ---
    cc_boxes = find_connected_components_binary(binary, w, h)
    print(f"  连通组件: 找到 {len(cc_boxes)} 个候选区域")

    # --- 选择更可靠的结果 ---
    # 如果投影分析找到了合理数量的框 (> 70% 的组件数), 优先用它
    # 否则使用合并后的结果
    if grid_boxes and len(grid_boxes) >= len(cc_boxes) * 0.7:
        boxes = grid_boxes
        print(f"  → 使用投影分析结果")
    else:
        # 使用连通组件 + 垂直合并
        boxes = merge_vertical_boxes(cc_boxes)
        print(f"  → 使用连通组件结果 (合并后 {len(boxes)} 个)")

    # --- 对非常宽的框尝试再次切分 ---
    final_boxes = []
    for box in boxes:
        split = split_wide_box(img, box, w, h, binary)
        final_boxes.extend(split)
    if len(final_boxes) != len(boxes):
        print(f"  宽框切分: {len(boxes)} → {len(final_boxes)}")

    # --- 裁剪 ---
    results = crop_and_save(img, final_boxes, base)
    for name, cw, ch in results:
        print(f"    => {name}  ({cw}x{ch})")

    return results


def main():
    # 清空输出目录
    for f in os.listdir(OUTPUT_DIR):
        fp = os.path.join(OUTPUT_DIR, f)
        if os.path.isfile(fp):
            os.remove(fp)

    total_files = 0
    total_crops = 0

    for i in range(1, 13):
        fname = f"{i}.jpg"
        fpath = os.path.join(WORK_DIR, fname)
        if os.path.exists(fpath):
            results = process_image(fpath)
            total_files += 1
            total_crops += len(results)

    print(f"\n{'='*50}")
    print(f"处理完成: {total_files} 个文件, 共裁剪出 {total_crops} 张子图")
    print(f"输出目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
