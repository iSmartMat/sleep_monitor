import os
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# -------------------------------------------------------------
# 1. Dataset (優化 DataLoader 效率)
# -------------------------------------------------------------
class PressureDataset(Dataset):
    def __init__(self, root_dir, subject_ids, n_max=80, threshold=0.1, sparsity=1.0, noise_std=0.01):
        self.root_dir = root_dir
        self.files = [f for f in os.listdir(root_dir) if f.endswith('.txt') and f.split('-')[1] in subject_ids]
        self.n_max = n_max
        self.threshold = threshold
        self.sparsity = sparsity
        self.noise_std = noise_std

    def __len__(self):
        return len(self.files)

    def _grid_to_points(self, grid):
        grid = grid + np.random.normal(0, self.noise_std, grid.shape).astype(np.float32)
        grid = np.clip(grid, 0, None)
        ys, xs = np.where(grid > self.threshold)
        ps = grid[ys, xs]
        points = np.stack([xs, ys, ps], axis=1).astype(np.float32)
        if len(points) > 0:
            keep = max(1, int(len(points) * self.sparsity))
            idx = np.random.choice(len(points), keep, replace=False)
            points = points[idx]
        if len(points) > self.n_max:
            points = points[:self.n_max]
        else:
            pad = np.zeros((self.n_max - len(points), 3), dtype=np.float32)
            points = np.vstack([points, pad])
        return points

    def __getitem__(self, idx):
        fname = self.files[idx]
        grid = np.loadtxt(os.path.join(self.root_dir, fname), dtype=np.float32)
        label = int(fname.split('-')[0])
        grid_tensor = torch.from_numpy(grid).unsqueeze(0) 
        points_tensor = torch.from_numpy(self._grid_to_points(grid))
        return grid_tensor, points_tensor, torch.tensor(label, dtype=torch.long)

# -------------------------------------------------------------
# 2. Models (維持先前定義之所有變體)
# -------------------------------------------------------------
class GridBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 4))
        )
        self.output_dim = 32 * 3 * 4
    def forward(self, x): return self.net(x).flatten(1)

class SparsePointGridNet(nn.Module):
    def __init__(self, num_classes=6, film_dim=64):
        super().__init__()
        self.grid_branch = GridBranch()
        self.grid_to_film = nn.Linear(self.grid_branch.output_dim, film_dim * 2)
        self.point_branch = nn.Sequential(nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, film_dim))
        self.fuse_proj = nn.Linear(self.grid_branch.output_dim, film_dim) 
        self.classifier = nn.Linear(film_dim + film_dim, num_classes)
        self.relu = nn.ReLU()

    def forward(self, grid, points):
        g_feat = self.grid_branch(grid)
        gamma_beta = self.grid_to_film(g_feat)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        p = self.point_branch(points)
        p = p * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        p = torch.max(self.relu(p), dim=1)[0]
        g_proj = self.relu(self.fuse_proj(g_feat))
        return self.classifier(torch.cat([g_proj, p], dim=1))

# 其他簡化模型 (省略重複代碼，確保邏輯與 SPGN_v2.py 相同)
class PointNet(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, 64), nn.ReLU())
        self.classifier = nn.Linear(64, num_classes)
    def forward(self, grid, points):
        p = torch.max(self.mlp(points), dim=1)[0]
        return self.classifier(p)

class ConcatNet(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.grid_branch = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((3, 4)))
        self.point_branch = nn.Sequential(nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, 64), nn.ReLU())
        self.fc = nn.Linear(16 * 3 * 4 + 64, num_classes)
    def forward(self, grid, points):
        g = self.grid_branch(grid).flatten(1)
        p = torch.max(self.point_branch(points), dim=1)[0]
        return self.fc(torch.cat([g, p], dim=1))

def get_model(name):
    if "SPGN (Full)" in name: return SparsePointGridNet()
    if "PointNet" in name or "no-grid" in name: return PointNet()
    if "ConcatNet" in name: return ConcatNet()
    return SparsePointGridNet() # Default

# -------------------------------------------------------------
# 3. 訓練核心與恢復機制
# -------------------------------------------------------------
def run_accelerated_experiments(root_dir, sparsity_levels=[0.1, 0.4]):
    results_dir = 'results_v3'
    ckpt_dir = 'checkpoints'
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None # 修正 FutureWarning
    subjects = sorted({f.split('-')[1] for f in os.listdir(root_dir) if f.endswith('.txt')})

    exp_configs = {
        "Exp1_Ablation": ["SPGN (Full)", "SPGN (no-grid)"],
        "Exp3_SOTA": ["SPGN (Full)", "PointNet", "ConcatNet"]
    }

    for exp_name, models in exp_configs.items():
        csv_path = os.path.join(results_dir, f'{exp_name}_metrics.csv')
        # 如果 CSV 已存在，讀取已完成的數據實現斷點恢復
        processed_keys = set()
        if os.path.exists(csv_path):
            with open(csv_path, 'r') as f:
                reader = csv.reader(f)
                next(reader) # Skip header
                for row in reader: processed_keys.add(f"{row[0]}_{row[1]}")

        for sp in sparsity_levels:
            for model_name in models:
                if f"{model_name}_{sp}" in processed_keys:
                    print(f"Skipping {model_name} at sp={sp} (Already processed)")
                    continue

                fold_accs = []
                for test_sid in tqdm(subjects, desc=f"{model_name}_sp{sp}"):
                    # 檢查是否有模型存檔
                    model_path = os.path.join(ckpt_dir, f"{exp_name}_{model_name}_sp{sp}_{test_sid}.pth")
                    
                    train_ds = PressureDataset(root_dir, [s for s in subjects if s != test_sid], sparsity=sp)
                    test_ds = PressureDataset(root_dir, [test_sid], sparsity=sp)
                    # 加速：num_workers 與 pin_memory
                    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
                    test_loader = DataLoader(test_ds, batch_size=1, num_workers=2)

                    model = get_model(model_name).to(device)
                    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
                    criterion = nn.CrossEntropyLoss()

                    # 訓練
                    model.train()
                    for epoch in range(20):
                        for g, p, l in train_loader:
                            g, p, l = g.to(device), p.to(device), l.to(device)
                            optimizer.zero_grad()
                            
                            # 加速：混合精度訓練 (AMP)
                            if scaler:
                                with torch.amp.autocast(device_type='cuda'):
                                    loss = criterion(model(g, p), l)
                                scaler.scale(loss).backward()
                                scaler.step(optimizer)
                                scaler.update()
                            else:
                                loss = criterion(model(g, p), l)
                                loss.backward()
                                optimizer.step()

                    # 存檔模型
                    torch.save(model.state_dict(), model_path)

                    # 評估
                    model.eval()
                    preds, labels = [], []
                    with torch.no_grad():
                        for g, p, l in test_loader:
                            out = model(g.to(device), p.to(device))
                            preds.append(out.argmax(1).item())
                            labels.append(l.item())
                    fold_accs.append(accuracy_score(labels, preds))

                # 存入 CSV
                with open(csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    if os.path.getsize(csv_path) == 0:
                        writer.writerow(['Model', 'Sparsity', 'MeanAcc', 'StdAcc'])
                    writer.writerow([model_name, sp, np.mean(fold_accs), np.std(fold_accs)])

if __name__ == "__main__":
    # 設定數據路徑
    data_path = os.path.dirname(os.path.realpath(__file__))+ os.path.sep+'7-data(12x18_V1-5)'
    
    # 確保數據目錄存在且有資料
    if not os.path.exists(data_path) or not any(f.endswith('.txt') for f in os.listdir(data_path)):
        print(f"Error: Data directory '{data_path}' not found or empty.")
        exit()

    # 設定要觀察的稀疏度級別
    target_sparsity = [1.0, 0.8, 0.6, 0.4] # 可以增加更多點
    run_accelerated_experiments(data_path, target_sparsity)
