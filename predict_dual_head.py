import argparse
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import os
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

# ================= 3. 极速预测并画图 =================
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
    
    # 3. 双头预测 (MPS 加速，仅需几毫秒)
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
    
    # 5. 将 56x56 热力图放大到原图尺寸，并上色叠加
    heatmap_img = Image.fromarray((pred_heatmap * 255).astype(np.uint8))
    heatmap_img = heatmap_img.resize((orig_w, orig_h), Image.Resampling.BILINEAR)
    
    # 创建一个半透明的红色热力图
    heatmap_color = Image.new("RGB", (orig_w, orig_h), (255, 0, 0))
    heatmap_mask = heatmap_img.convert("L")  # 用热力图作为透明度遮罩
    heatmap_color.putalpha(heatmap_mask)     # 加上透明度
    
    # 将热力图叠加到原图上
    result_img = image.copy()
    result_img.paste(heatmap_color, (0, 0), heatmap_color)
    
    # 6. 保存并用 Mac 预览打开
    save_name = f"dual_predicted_{image_path.split('/')[-1]}"
    result_img.save(save_name)
    print(f"✅ 预测完成！结果已保存为 {save_name}")
    
    # 自动用 Mac 默认看图软件打开
    subprocess.run(["open", save_name])

# ================= 4. 命令行参数解析 =================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="双头摄影眼模型预测脚本")
    parser.add_argument("image", type=str, help="要预测的图片路径")
    args = parser.parse_args()
    predict_and_draw(args.image)