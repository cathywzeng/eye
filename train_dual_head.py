import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm

# ================= 1. 核心配置 =================
DATASET_DIR = "my_dataset"      # 你的图片和 LabelMe JSON 所在的同一个文件夹
BATCH_SIZE = 16                 # M1 16G 内存建议 16
EPOCHS = 50                     # 双头模型可能需要多跑几个 Epoch
LEARNING_RATE = 1e-4
IMG_SIZE = 224                  # MobileNet 标准输入尺寸
HEATMAP_SIZE = 56               # 热力图尺寸 (224 / 4 = 56)

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("🍎 成功启用 Apple Silicon MPS 加速！")
else:
    DEVICE = torch.device("cpu")
    print("⚠️ 未检测到 MPS，将使用 CPU 运行。")

# ================= 2. 高斯热力图生成器 =================
def generate_gaussian_heatmap(target_size=56, box=None):
    """根据框的坐标生成高斯热力图"""
    heatmap = np.zeros((target_size, target_size), dtype=np.float32)
    if box is None: return heatmap
    
    x1, y1, x2, y2 = box
    center_x, center_y = int((x1 + x2) / 2 * target_size), int((y1 + y2) / 2 * target_size)
    sigma = max(1, int(min(x2 - x1, y2 - y1) * target_size / 4))
    
    size = 6 * sigma + 1
    x_coord = np.arange(0, size, 1, float) - size // 2
    y_coord = np.arange(0, size, 1, float) - size // 2
    y, x = np.meshgrid(y_coord, x_coord)
    g = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    
    x_min, y_min = max(0, center_x - size // 2), max(0, center_y - size // 2)
    x_max, y_max = min(target_size, center_x + size // 2 + 1), min(target_size, center_y + size // 2 + 1)
    g_x_min, g_y_min = max(0, size // 2 - center_x), max(0, size // 2 - center_y)
    g_x_max, g_y_max = g_x_min + (x_max - x_min), g_y_min + (y_max - y_min)
    
    heatmap[y_min:y_max, x_min:x_max] = g[g_y_min:g_y_max, g_x_min:g_x_max]
    return heatmap

# ================= 3. 升级版：Letterbox 数据集类 =================
class LetterboxCompositionDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.samples = []
        
        for fname in os.listdir(img_dir):
            if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                json_path = os.path.join(img_dir, fname.rsplit('.', 1)[0] + '.json')
                if os.path.exists(json_path):
                    self.samples.append((fname, json_path))
        
        if len(self.samples) == 0:
            raise ValueError(f"⚠️ 在 {img_dir} 中未找到任何 图片+JSON 配对！")
        print(f"📊 成功加载 {len(self.samples)} 组标注数据。")

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_name, json_path = self.samples[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        box = None
        for shape in data['shapes']:
            if shape['shape_type'] in ['rectangle', 'oriented_rectangle']:
                points = shape['points']
                if len(points) == 2:
                    (x1, y1), (x2, y2) = points
                    x_min, x_max = min(x1, x2), max(x1, x2)
                    y_min, y_max = min(y1, y2), max(y1, y2)
                elif len(points) >= 4:
                    x_coords = [p[0] for p in points]
                    y_coords = [p[1] for p in points]
                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)
                else: continue
                    
                box = [x_min / orig_w, y_min / orig_h, x_max / orig_w, y_max / orig_h]
                break
        
        if box is None: box = [0.0, 0.0, 1.0, 1.0] 

        # Letterbox 等比缩放 + 居中填充
        scale = IMG_SIZE / max(orig_w, orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        resized_img = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        padded_img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0, 0, 0))
        pad_x, pad_y = (IMG_SIZE - new_w) // 2, (IMG_SIZE - new_h) // 2
        padded_img.paste(resized_img, (pad_x, pad_y))
        
        # 修正坐标
        x1_new = (box[0] * new_w + pad_x) / IMG_SIZE
        y1_new = (box[1] * new_h + pad_y) / IMG_SIZE
        x2_new = (box[2] * new_w + pad_x) / IMG_SIZE
        y2_new = (box[3] * new_h + pad_y) / IMG_SIZE
        final_box = [max(0.0, min(1.0, x1_new)), max(0.0, min(1.0, y1_new)), 
                     max(0.0, min(1.0, x2_new)), max(0.0, min(1.0, y2_new))]

        # 生成高斯热力图标签
        heatmap = generate_gaussian_heatmap(HEATMAP_SIZE, final_box)

        if self.transform:
            padded_img = self.transform(padded_img)
            
        return padded_img, torch.tensor(final_box, dtype=torch.float32), torch.tensor(heatmap, dtype=torch.float32).unsqueeze(0)

# ================= 4. 终极双头模型 =================
class DualHeadMobileNet(nn.Module):
    def __init__(self):
        super(DualHeadMobileNet, self).__init__()
        base_model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        self.features = base_model.features
        
        # 分支 A：坐标回归头
        self.reg_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(576, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )
        
        # 分支 B：热力图回归头 (保留空间结构)
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(576, 1, kernel_size=1),
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False),  # 【新增】将 7x7 放大 8 倍变成 56x56
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.features(x)
        reg_out = self.reg_head(features)
        heatmap_out = self.heatmap_head(features)
        return reg_out, heatmap_out

# ================= 5. 训练循环 =================
def train_model():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = LetterboxCompositionDataset(DATASET_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    model = DualHeadMobileNet().to(DEVICE)
    criterion_reg = nn.SmoothL1Loss()
    criterion_heatmap = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("-" * 40)
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for images, boxes, heatmaps in pbar:
            images = images.to(DEVICE)
            boxes = boxes.to(DEVICE)
            heatmaps = heatmaps.to(DEVICE)
            
            optimizer.zero_grad()
            pred_reg, pred_heatmap = model(images)
            pred_reg = torch.sigmoid(pred_reg)
            
            loss_reg = criterion_reg(pred_reg, boxes)
            loss_heatmap = criterion_heatmap(pred_heatmap, heatmaps)
            total_loss = loss_reg + 1.0 * loss_heatmap
            
            total_loss.backward()
            optimizer.step()
            
            running_loss += total_loss.item()
            pbar.set_postfix({"Loss": f"{total_loss.item():.4f}"})
            
        avg_loss = running_loss / len(dataloader)
        print(f"✅ Epoch {epoch+1} | 平均损失: {avg_loss:.4f}")

    torch.save(model.state_dict(), "dual_head_photography_eye.pth")
    print("\n🎉 双头模型训练完成！已保存为 dual_head_photography_eye.pth")

if __name__ == "__main__":
    train_model()