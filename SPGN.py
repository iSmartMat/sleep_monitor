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
from sklearn.manifold import TSNE
from tqdm import tqdm

# -------------------------------------------------------------
# 1. Dataset - 保持物理雜訊模擬
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
        # 增加雜訊以提升訓練難度與模型魯棒性
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
        points = self._grid_to_points(grid)
        return torch.from_numpy(grid[None, ...]), torch.from_numpy(points), torch.tensor(label)

# -------------------------------------------------------------
# 2. Models - 優化 Grid Branch (針對 12x18 設計)
# -------------------------------------------------------------
class SparsePointGridNet(nn.Module):
    def __init__(self, num_classes=6, film_dim=64):
        super().__init__()
        # 優化: 針對 12x18 調整 Kernel 與 Stride，避免資訊過早丟失
        self.grid_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 3), stride=1, padding=1), 
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=2, padding=1), # 變為 6x9
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 4)) # 最終壓縮至 12 個特徵點，適合嵌入式運算
        )
        self.grid_flatten_dim = 32 * 3 * 4
        self.grid_fc = nn.Linear(self.grid_flatten_dim, film_dim * 2)

        # Point branch
        self.point_mlp = nn.Sequential(
            nn.Linear(3, 32), nn.ReLU(),
            nn.Linear(32, film_dim)
        )
        
        self.fuse_proj = nn.Linear(self.grid_flatten_dim, film_dim) 
        self.classifier = nn.Linear(film_dim + film_dim, num_classes)
        self.relu = nn.ReLU()

    def forward(self, grid, points, return_features=False):
        g_feat = self.grid_conv(grid).flatten(1)
        gamma_beta = self.grid_fc(g_feat)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)

        p = self.point_mlp(points)
        # FiLM 融合
        p = p * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        p = torch.max(self.relu(p), dim=1)[0]

        g_proj = self.relu(self.fuse_proj(g_feat))
        combined = torch.cat([g_proj, p], dim=1)
        
        if return_features: return combined
        return self.classifier(combined)

class ConcatNet(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.grid_branch = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), 
            nn.AdaptiveAvgPool2d((3, 4))
        )
        self.point_branch = nn.Sequential(nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, 64), nn.ReLU())
        self.fc = nn.Linear(16 * 3 * 4 + 64, num_classes)

    def forward(self, grid, points):
        g = self.grid_branch(grid).flatten(1)
        p = torch.max(self.point_branch(points), dim=1)[0]
        return self.fc(torch.cat([g, p], dim=1))

# -------------------------------------------------------------
# 3. Visualization Helpers (修復 t-SNE 報錯)
# -------------------------------------------------------------
def plot_journal_confusion_matrix(y_true, y_pred, classes, save_path):
    cm = confusion_matrix(y_true, y_pred)
    cm_perc = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm_perc, annot=True, fmt='.2f', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Normalized Confusion Matrix (Sparsity=0.1)')
    plt.ylabel('Actual Posture')
    plt.xlabel('Predicted Posture')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_robust_tsne(features, labels, save_path):
    n_samples = len(features)
    # 修復: 確保 perplexity 小於樣本數
    perp = min(30, max(1, n_samples - 1))
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42, init='pca', learning_rate='auto')
    feat_2d = tsne.fit_transform(features)
    
    plt.figure(figsize=(9, 7))
    plt.style.use('seaborn-v0_8-whitegrid')
    scatter = plt.scatter(feat_2d[:, 0], feat_2d[:, 1], c=labels, cmap='Set2', alpha=0.8, edgecolors='w')
    plt.colorbar(scatter, label='Posture Label')
    plt.title(f't-SNE Feature Clustering (Perplexity={perp})')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# -------------------------------------------------------------
# 4. Main Experiment Logic (優化訓練策略)
# -------------------------------------------------------------
def run_optimized_experiment(root_dir, sparsity_levels=[0.1, 0.2, 0.3, 0.4]):
    os.makedirs('results', exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    subjects = sorted({f.split('-')[1] for f in os.listdir(root_dir) if f.endswith('.txt')})
    
    # 存檔結構
    summary_results = {m: {sp: [] for sp in sparsity_levels} for m in ['SPGN', 'ConcatNet']}
    tsne_data = {"features": [], "labels": []}
    cm_data = {"true": [], "pred": []}

    for sp in sparsity_levels:
        print(f"\n🚀 Evaluating Sparsity Level: {sp}")
        for model_name in ['SPGN', 'ConcatNet']:
            pbar = tqdm(subjects, desc=f"Mode: {model_name}")
            for test_sid in pbar:
                train_sids = [s for s in subjects if s != test_sid]
                train_ds = PressureDataset(root_dir, train_sids, sparsity=sp)
                test_ds = PressureDataset(root_dir, [test_sid], sparsity=sp)
                train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
                test_loader = DataLoader(test_ds, batch_size=1)

                model = SparsePointGridNet().to(device) if model_name == 'SPGN' else ConcatNet().to(device)
                
                # 優化訓練策略: Label Smoothing + AdamW
                optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
                # 優化訓練策略: 學習率調度
                scheduler = ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5)
                criterion = nn.CrossEntropyLoss(label_smoothing=0.1) 

                # Training Loop
                for epoch in range(20):
                    model.train()
                    for g, p, l in train_loader:
                        g, p, l = g.to(device), p.to(device), l.to(device)
                        optimizer.zero_grad()
                        loss = criterion(model(g, p), l)
                        loss.backward()
                        optimizer.step()

                # Evaluation
                model.eval()
                y_p, y_l = [], []
                with torch.no_grad():
                    for g, p, l in test_loader:
                        g, p = g.to(device), p.to(device)
                        out = model(g, p)
                        y_p.append(out.argmax(1).item())
                        y_l.append(l.item())
                        
                        # 收集 t-SNE 與 CM 數據 (僅針對 SPGN 在最低稀疏度)
                        if model_name == 'SPGN' and sp == min(sparsity_levels):
                            feat = model(g, p, return_features=True)
                            tsne_data["features"].append(feat.cpu().numpy().flatten())
                            tsne_data["labels"].append(l.item())
                            cm_data["true"].append(l.item())
                            cm_data["pred"].append(out.argmax(1).item())

                acc = accuracy_score(y_l, y_p)
                summary_results[model_name][sp].append(acc)
                scheduler.step(acc)
                pbar.set_postfix(acc=f"{acc:.2f}")

    # --- 數據導出 ---
    csv_path = 'results/loso_sparsity_metrics.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Sparsity', 'MeanAcc', 'StdAcc'])
        for name in ['SPGN', 'ConcatNet']:
            for sp in sparsity_levels:
                accs = summary_results[name][sp]
                writer.writerow([name, sp, np.mean(accs), np.std(accs)])

    # --- 繪製 Accuracy 圖 ---
    plt.figure(figsize=(8, 5))
    for name in ['SPGN', 'ConcatNet']:
        means = [np.mean(summary_results[name][sp]) for sp in sparsity_levels]
        stds = [np.std(summary_results[name][sp]) for sp in sparsity_levels]
        plt.errorbar(sparsity_levels, means, yerr=stds, marker='o', capsize=5, label=f'{name} (Mean±Std)')
    plt.xlabel('Remaining Point Ratio (Sparsity)')
    plt.ylabel('LOSO Mean Accuracy')
    plt.title('Performance Robustness Comparison')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('results/robustness_comparison.png', dpi=300)

    # --- 繪製 CM 與 t-SNE ---
    plot_journal_confusion_matrix(cm_data["true"], cm_data["pred"], range(6), 'results/journal_cm.png')
    if len(tsne_data["features"]) > 1:
        plot_robust_tsne(np.array(tsne_data["features"]), np.array(tsne_data["labels"]), 'results/journal_tsne.png')

    print(f"\n✅ 全部實驗完成！原始數據已存至: {csv_path}")

if __name__ == "__main__":
    # 使用者可自行定義 sparsity 陣列
    target_sparsity = [0.1, 0.2, 0.3, 0.4]
    data_path = os.path.join(os.getcwd(), 'data')
    run_optimized_experiment(data_path, target_sparsity)
