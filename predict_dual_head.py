import argparse
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageDraw
import numpy as np
import subprocess

# ================= 1. 核心配置 =================
MODEL_PATH = "dual_head_photography_eye.pth"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
IMG_SIZE = 224

# ================= 2. 终极双头模型 =================
class DualHeadMobileNet(nn.Module):
    def __init__(self):
        super(DualHeadMobileNet, self).__init__()
        base_model = models.mobilenet_v3_small(weights=None)
        self.features = base_model.features
        
        self.reg_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(576, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )
        
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(576, 1, kernel_size=1),
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.features(x)
        return self.reg_head(features), self.heatmap_head(features)

# ================= 3. 极速预测并画图 (纯 Pillow 版) =================
def predict_and_draw(image_path):
    # 1. 加载模型
    model = DualHeadMobileNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    # 2. 加载并预处理图片
    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size
    
    # Letterbox 等比缩放 + 居中填充
    scale = IMG_SIZE / max(orig_w, orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    resized_img = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    padded_img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0, 0, 0))
    pad_x, pad_y = (IMG_SIZE - new_w) // 2, (IMG_SIZE - new_h) // 2
    padded_img.paste(resized_img, (pad_x, pad_y))
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(padded_img).unsqueeze(0).to(DEVICE)
    
    # 3. 双头预测
    with torch.no_grad():
        pred_reg, pred_heatmap = model(input_tensor)
        pred_reg = torch.sigmoid(pred_reg).cpu().numpy()[0]
        pred_heatmap = pred_heatmap.cpu().numpy()[0, 0]  # 提取 56x56 热力图
        
    # 4. 还原坐标并在原图上画红框
    x1 = max(0, (pred_reg[0] * IMG_SIZE - pad_x) / scale)
    y1 = max(0, (pred_reg[1] * IMG_SIZE - pad_y) / scale)
    x2 = min(orig_w, (pred_reg[2] * IMG_SIZE - pad_x) / scale)
    y2 = min(orig_h, (pred_reg[3] * IMG_SIZE - pad_y) / scale)
    
    draw = ImageDraw.Draw(image)
    line_width = max(3, int(min(orig_w, orig_h) / 200))
    draw.rectangle([x1, y1, x2, y2], outline="red", width=line_width)
    
    # 5. 生成纯粹的黑白热力图（最清晰，强推！）
    heatmap_img = Image.fromarray((pred_heatmap * 255).astype(np.uint8))
    heatmap_img = heatmap_img.resize((orig_w, orig_h), Image.Resampling.BILINEAR)
    pure_heatmap_name = f"pure_heatmap_{image_path.split('/')[-1]}"
    heatmap_img.save(pure_heatmap_name)
    
    # 6. 用纯 Pillow 生成“彩虹热力图”叠加效果
    # 将 0-1 的热力图转换为 0-255 的 RGB 彩虹色 (Jet colormap 的简化版)
    h = pred_heatmap
    r = np.clip(1.5 - abs(4.0 * h - 3.0), 0, 1)
    g = np.clip(1.5 - abs(4.0 * h - 2.0), 0, 1)
    b = np.clip(1.5 - abs(4.0 * h - 1.0), 0, 1)
    
    # 组合成 RGB 图像
    rgb_array = np.stack((r, g, b), axis=-1)
    rgb_array = (rgb_array * 255).astype(np.uint8)
    
    heatmap_color = Image.fromarray(rgb_array, 'RGB')
    heatmap_color = heatmap_color.resize((orig_w, orig_h), Image.Resampling.BILINEAR)
    
    # 将彩虹热力图叠加到原图上
    result_img = image.copy()
    result_img = Image.blend(result_img, heatmap_color, alpha=0.5)  # 50% 透明度混合
    
    # 7. 保存并用 Mac 预览打开
    save_name = f"dual_predicted_{image_path.split('/')[-1]}"
    result_img.save(save_name)
    
    print(f"✅ 预测完成！")
    print(f"📍 1. 红框+彩虹热力图已保存为: {save_name}")
    print(f"🔥 2. 纯粹黑白热力图已保存为: {pure_heatmap_name} (强烈建议看这张！)")
    
    # 自动用 Mac 预览打开两张图
    subprocess.run(["open", save_name])
    subprocess.run(["open", pure_heatmap_name])

# ================= 4. 命令行参数解析 =================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="双头摄影眼模型预测脚本")
    parser.add_argument("image", type=str, help="要预测的图片路径")
    args = parser.parse_args()
    predict_and_draw(args.image)