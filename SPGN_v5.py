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
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm
from collections import defaultdict

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
# === 對照組 1: 消融實驗用 ===
class GridCNN(nn.Module):
    """僅使用 Grid 分支 (SPGN no-point)"""
    def __init__(self, num_classes=6):
        super().__init__()
        self.grid_branch = GridBranch()
        self.classifier = nn.Linear(self.grid_branch.output_dim, num_classes)
    def forward(self, grid, points=None): # points 被忽略
        g = self.grid_branch(grid)
        return self.classifier(g)

# === 對照組 2: 機制比較用 ===
class SPGN_ConcatFusion(nn.Module):
    """結構與 SPGN 相同，但不使用 FiLM，改用最後拼接 (SPGN no-FiLM)"""
    def __init__(self, num_classes=6, film_dim=64):
        super().__init__()
        self.grid_branch = GridBranch()
        self.point_branch = PointBranch(film_dim)
        # 不生成 gamma/beta，直接投影
        self.grid_proj = nn.Linear(self.grid_branch.output_dim, film_dim)
        self.classifier = nn.Linear(film_dim + film_dim, num_classes)
        self.relu = nn.ReLU()

    def forward(self, grid, points):
        g = self.relu(self.grid_proj(self.grid_branch(grid)))
        p = self.point_branch(points)
        p = torch.max(self.relu(p), dim=1)[0] # 沒有 FiLM 調製
        combined = torch.cat([g, p], dim=1) # 簡單拼接
        return self.classifier(combined)

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
    if 'SPGN (no-point)' in name: return GridCNN()
    if 'SPGN (no-FiLM)' in name: return SPGN_ConcatFusion()
    if "SPGN (no-grid)" in name or "PointNet" in name: return PointNet()
    if "ConcatNet" in name: return ConcatNet()
    return SparsePointGridNet() # Default

# -------------------------------------------------------------
# 3. 訓練核心與恢復機制
# -------------------------------------------------------------
def run_accelerated_experiments(root_dir, sparsity_levels=[0.1, 0.4]):

    results_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))+ os.path.sep+'results_v5'
    ckpt_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))+ os.path.sep+'checkpoints_v5'
    cm_dir = os.path.join(results_dir, "confusion_matrices")
    plot_dir = os.path.join(results_dir, "plots")

    deploy_dir = results_dir

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(deploy_dir, exist_ok=True)
    os.makedirs(cm_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    subjects = sorted({f.split('-')[1] for f in os.listdir(root_dir) if f.endswith('.txt')})

    exp_configs = {
        "Exp1_Ablation": ["SPGN (Full)", "SPGN (no-grid)", "SPGN (no-point)"],
        "Exp2_FiLM_Mechanism": ["SPGN (Full)", "SPGN (no-FiLM)"],
        "Exp3_SOTA_Comparison": ["SPGN (Full)", "PointNet", "ConcatNet"]
    }

    for exp_name, models in exp_configs.items():

        csv_path = os.path.join(results_dir, f'{exp_name}_metrics.csv')
        processed_keys = set()

        if os.path.exists(csv_path):
            with open(csv_path, 'r') as f:
                reader = csv.reader(f)
                next(reader, None)
                for row in reader:
                    processed_keys.add(f"{row[0]}_{row[1]}")

        # ==============================
        # Cross-validation
        # ==============================
        for sp in sparsity_levels:
            for model_name in models:

                if f"{model_name}_{sp}" in processed_keys:
                    print(f"Skipping {model_name} at sp={sp}")
                    continue

                fold_accs = []
                all_preds = []
                all_labels = []

                for test_sid in tqdm(subjects, desc=f"{model_name}_sp{sp}"):

                    model_path = os.path.join(
                        ckpt_dir,
                        f"{exp_name}_{model_name}_sp{sp}_{test_sid}.pth"
                    )

                    train_ds = PressureDataset(
                        root_dir,
                        [s for s in subjects if s != test_sid],
                        sparsity=sp
                    )

                    test_ds = PressureDataset(
                        root_dir,
                        [test_sid],
                        sparsity=sp
                    )

                    train_loader = DataLoader(
                        train_ds,
                        batch_size=128,
                        shuffle=True,
                        num_workers=4,
                        pin_memory=True
                    )

                    test_loader = DataLoader(
                        test_ds,
                        batch_size=1,
                        num_workers=2
                    )

                    model = get_model(model_name).to(device)
                    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
                    criterion = nn.CrossEntropyLoss()

                    if os.path.exists(model_path):
                        model.load_state_dict(torch.load(model_path, map_location=device))
                    else:
                        model.train()
                        for epoch in range(20):
                            for g, p, l in train_loader:
                                g, p, l = g.to(device), p.to(device), l.to(device)
                                optimizer.zero_grad()

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

                        torch.save(model.state_dict(), model_path)

                    model.eval()
                    preds, labels = [], []

                    with torch.no_grad():
                        for g, p, l in test_loader:
                            out = model(g.to(device), p.to(device))
                            preds.append(out.argmax(1).item())
                            labels.append(l.item())

                    fold_accs.append(accuracy_score(labels, preds))
                    all_preds.extend(preds)
                    all_labels.extend(labels)

                # 混淆矩陣
                labels_sorted = sorted(set(all_labels))
                cm = confusion_matrix(all_labels, all_preds, labels=labels_sorted)

                base = f"{exp_name}_{model_name}_sp{sp}"
                np.savetxt(os.path.join(cm_dir, base + "_cm.csv"), cm, delimiter=',', fmt='%d')
                np.save(os.path.join(cm_dir, base + "_cm.npy"), cm)

                plt.figure(figsize=(6, 6))
                plt.imshow(cm)
                plt.title(f"{model_name} sp={sp}")
                plt.colorbar()
                plt.xticks(range(len(labels_sorted)), labels_sorted)
                plt.yticks(range(len(labels_sorted)), labels_sorted)
                plt.tight_layout()
                plt.savefig(os.path.join(cm_dir, base + "_cm.png"))
                plt.close()

                with open(csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    if os.path.getsize(csv_path) == 0:
                        writer.writerow(['Model', 'Sparsity', 'MeanAcc', 'StdAcc'])
                    writer.writerow([
                        model_name,
                        sp,
                        np.mean(fold_accs),
                        np.std(fold_accs)
                    ])

        # ==============================
        # 選最佳模型
        # ==============================
        df = pd.read_csv(csv_path)
        best_row = df.loc[df['MeanAcc'].idxmax()]
        best_model = best_row['Model']
        best_sp = best_row['Sparsity']

        print(f"Best for {exp_name}: {best_model} at sp={best_sp}")

        deploy_path = os.path.join(
            deploy_dir,
            f"{exp_name}_BEST_{best_model}_sp{best_sp}_deployment.pth"
        )

        if not os.path.exists(deploy_path):

            print("Training deployment model on ALL subjects...")

            full_ds = PressureDataset(root_dir, subjects, sparsity=best_sp)
            full_loader = DataLoader(
                full_ds,
                batch_size=128,
                shuffle=True,
                num_workers=4,
                pin_memory=True
            )

            model = get_model(best_model).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            criterion = nn.CrossEntropyLoss()

            model.train()
            for epoch in tqdm(range(20), desc="Deployment Training"):
                for g, p, l in full_loader:
                    g, p, l = g.to(device), p.to(device), l.to(device)
                    optimizer.zero_grad()

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

            torch.save(model.state_dict(), deploy_path)
            print(f"Deployment model saved to {deploy_path}")

        # ==============================
        # Accuracy Plot
        # ==============================
        plt.figure(figsize=(8, 5))
        for m in df['Model'].unique():
            sub = df[df['Model'] == m]
            plt.plot(sub['Sparsity'], sub['MeanAcc'], marker='o', label=m)

        plt.xlabel("Sparsity")
        plt.ylabel("Mean Accuracy")
        plt.title(exp_name + " Accuracy Comparison")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{exp_name}_accuracy_vs_sparsity.png"))
        plt.close()

    print("All experiments completed.")



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
