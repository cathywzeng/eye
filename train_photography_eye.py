import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm

# ================= 1. 核心配置 =================
DATASET_DIR = "my_dataset"      # 你的图片和 LabelMe JSON 所在的同一个文件夹
BATCH_SIZE = 16                 # M1 16G 内存建议 16，如果爆内存(OOM)改成 8
EPOCHS = 50                     # 110张图很少，可以多跑几个 Epoch
LEARNING_RATE = 1e-4
IMG_SIZE = 224                  # MobileNet 标准输入尺寸

# 自动检测 Apple Silicon MPS 加速
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("🍎 成功启用 Apple Silicon MPS 加速！")
else:
    DEVICE = torch.device("cpu")
    print("⚠️ 未检测到 MPS，将使用 CPU 运行。")

# ================= 2. 升级版：Letterbox 数据集类 =================
class LetterboxCompositionDataset(Dataset):
    def __init__(self, img_dir, target_size=224, transform=None):
        self.img_dir = img_dir
        self.target_size = target_size
        self.transform = transform
        self.samples = []
        
        # 扫描文件夹，找所有有对应 JSON 的图片
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
        
        # 1. 读取图片并获取原始尺寸
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size
        
        # 2. 读取 LabelMe JSON 并提取框
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
                else:
                    continue
                    
                # 归一化到 [0, 1]
                box = [x_min / orig_w, y_min / orig_h, x_max / orig_w, y_max / orig_h]
                break
        
        if box is None:
            box = [0.0, 0.0, 1.0, 1.0] 

        # 3. 【核心升级】Letterbox 等比缩放 + 居中填充
        # 计算缩放比例，取较小值以确保图片能完全放进 224x224 中
        scale = self.target_size / max(orig_w, orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        
        # 等比缩放图片
        resized_img = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # 创建一张纯黑色的 224x224 画布，并把缩放后的图片贴在正中间
        padded_img = Image.new("RGB", (self.target_size, self.target_size), (0, 0, 0))
        pad_x = (self.target_size - new_w) // 2
        pad_y = (self.target_size - new_h) // 2
        padded_img.paste(resized_img, (pad_x, pad_y))
        
        # 4. 【关键】修正坐标：因为加了黑边，框的相对位置变了！
        # 把原来 [0,1] 的坐标，映射到加了黑边后的 224x224 画布上，再归一化
        x1_new = (box[0] * new_w + pad_x) / self.target_size
        y1_new = (box[1] * new_h + pad_y) / self.target_size
        x2_new = (box[2] * new_w + pad_x) / self.target_size
        y2_new = (box[3] * new_h + pad_y) / self.target_size
        
        # 裁剪坐标，防止超出 [0, 1] 范围
        final_box = [
            max(0.0, min(1.0, x1_new)),
            max(0.0, min(1.0, y1_new)),
            max(0.0, min(1.0, x2_new)),
            max(0.0, min(1.0, y2_new))
        ]

        if self.transform:
            padded_img = self.transform(padded_img)
            
        return padded_img, torch.tensor(final_box, dtype=torch.float32)

# ================= 3. 构建轻量级回归模型 =================
def get_model():
    # 使用 MobileNetV3-Small，非常适合手机端部署
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    
    # 冻结特征提取层，只训练回归头（省内存、训练快）
    for param in model.features.parameters():
        param.requires_grad = False
        
    # 替换分类头为 4 坐标回归头
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, 4)
    return model

# ================= 4. 训练循环 =================
def train_model():
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = LetterboxCompositionDataset(DATASET_DIR, transform=transform)
    # num_workers=0 避免 Mac 上的多进程内存问题
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    model = get_model().to(DEVICE)
    criterion = nn.SmoothL1Loss()  # 回归任务常用损失函数
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    
    print("-" * 40)
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for images, boxes in pbar:
            images, boxes = images.to(DEVICE), boxes.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            # 使用 Sigmoid 确保输出坐标在 [0, 1] 之间
            outputs = torch.sigmoid(outputs) 
            
            loss = criterion(outputs, boxes)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
        avg_loss = running_loss / len(dataloader)
        print(f"✅ Epoch {epoch+1} | 平均损失: {avg_loss:.4f}")

    # 保存最终模型
    torch.save(model.state_dict(), "photography_eye_v1.pth")
    print("\n🎉 训练完成！模型已保存为 photography_eye_v1.pth")

if __name__ == "__main__":
    train_model()