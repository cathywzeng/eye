import argparse
import torch
from torchvision import transforms, models
from PIL import Image, ImageDraw

# ================= 1. 核心配置 =================
MODEL_PATH = "photography_eye_v1.pth"  # 刚才训练生成的模型文件
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ================= 2. 加载模型 =================
def load_model():
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(in_features, 4)
    
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()  # 切换到评估模式
    model.to(DEVICE)
    return model

# ================= 3. 预测并画图 =================
def predict_and_draw(image_path):
    # 1. 加载模型
    model = load_model()
    
    # 2. 加载并预处理图片
    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size
    
    # 3. 【核心】Letterbox 等比缩放 + 居中填充（必须与训练时一致！）
    target_size = 224
    scale = target_size / max(orig_w, orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    
    resized_img = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    padded_img = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    padded_img.paste(resized_img, (pad_x, pad_y))
    
    # 4. 转换为 Tensor 并预测
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(padded_img).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        # 使用 Sigmoid 将输出映射到 [0, 1]
        pred_box = torch.sigmoid(outputs).cpu().numpy()[0]
        
    # 5. 将预测的 [0, 1] 坐标还原为原图的真实像素坐标
    x1 = max(0, (pred_box[0] * target_size - pad_x) / scale)
    y1 = max(0, (pred_box[1] * target_size - pad_y) / scale)
    x2 = min(orig_w, (pred_box[2] * target_size - pad_x) / scale)
    y2 = min(orig_h, (pred_box[3] * target_size - pad_y) / scale)
    
    # 6. 在原图上画出预测的框
    draw = ImageDraw.Draw(image)
    line_width = max(3, int(min(orig_w, orig_h) / 200))
    draw.rectangle([x1, y1, x2, y2], outline="red", width=line_width)
    
    # 显示并保存图片
    image.show()
    save_name = f"predicted_{image_path.split('/')[-1]}"
    image.save(save_name)
    print(f"✅ 预测完成！结果已保存为 {save_name}")
    print(f"📍 预测框坐标: 左上({x1:.0f}, {y1:.0f}), 右下({x2:.0f}, {y2:.0f})")

# ================= 4. 命令行参数解析 =================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="摄影眼模型预测脚本")
    parser.add_argument("image", type=str, help="要预测的图片路径")
    args = parser.parse_args()
    
    predict_and_draw(args.image)