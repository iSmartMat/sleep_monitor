import os
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.manifold import TSNE
from tqdm import tqdm

# -------------------------------------------------------------
# 1. Dataset - 優化 A: 增加物理雜訊模擬
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
        # 加入微量雜訊模擬真實感測器底噪
        grid = grid + np.random.normal(0, self.noise_std, grid.shape).astype(np.float32)
        grid = np.clip(grid, 0, None) # 壓力值不為負
        
        ys, xs = np.where(grid > self.threshold)
        ps = grid[ys, xs]
        points = np.stack([xs, ys, ps], axis=1).astype(np.float32)
        
        if len(points) > 0:
            # 稀疏性控制
            keep = max(1, int(len(points) * self.sparsity))
            idx = np.random.choice(len(points), keep, replace=False)
            points = points[idx]
            
        # Padding / Truncation
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
# 2. Models - 優化 B/C: 參數量計算與 FiLM 邏輯修復
# -------------------------------------------------------------
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class SparsePointGridNet(nn.Module):
    def __init__(self, num_classes=6, film_dim=64):
        super().__init__()
        # Grid branch
        self.grid_conv = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)) # 降採樣以符合嵌入式記憶體
        )
        self.grid_flatten_dim = 8 * 4 * 4
        self.grid_fc = nn.Linear(self.grid_flatten_dim, film_dim * 2)

        # Point branch
        self.point_mlp = nn.Sequential(
            nn.Linear(3, 32), nn.ReLU(),
            nn.Linear(32, film_dim)
        )
        
        # 優化 C: 使用投影層而非截斷，保留完整全局特徵
        self.fuse_proj = nn.Linear(self.grid_flatten_dim, film_dim) 
        self.classifier = nn.Linear(film_dim + film_dim, num_classes)
        self.relu = nn.ReLU()

    def forward(self, grid, points, return_features=False):
        # Grid path
        g_feat = self.grid_conv(grid).flatten(1)
        gamma_beta = self.grid_fc(g_feat)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)

        # Point path with FiLM
        p = self.point_mlp(points)
        p = p * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        p = torch.max(self.relu(p), dim=1)[0]

        # Fusion
        g_proj = self.relu(self.fuse_proj(g_feat))
        combined = torch.cat([g_proj, p], dim=1)
        
        if return_features: return combined
        return self.classifier(combined)

# (其他模型 PointNet, ConcatNet 略，保持與原程式碼相似但建議加入 count_parameters)
class ConcatNet(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.grid_branch = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4)))
        self.point_branch = nn.Sequential(nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, 64), nn.ReLU())
        self.fc = nn.Linear(8 * 4 * 4 + 64, num_classes)

    def forward(self, grid, points):
        g = self.grid_branch(grid).flatten(1)
        p = torch.max(self.point_branch(points), dim=1)[0]
        return self.fc(torch.cat([g, p], dim=1))

# -------------------------------------------------------------
# 3. Visualization Helpers
# -------------------------------------------------------------
def plot_confusion_matrix(y_true, y_pred, classes, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix (Lowest Sparsity)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path)
    plt.close()

def plot_tsne(features, labels, save_path):
    tsne = TSNE(n_components=2, random_state=42)
    feat_2d = tsne.fit_transform(features)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(feat_2d[:, 0], feat_2d[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.legend(*scatter.legend_elements(), title="Postures")
    plt.title('t-SNE Visualization of SPGN Features (Sparsity=0.1)')
    plt.savefig(save_path)
    plt.close()

# -------------------------------------------------------------
# 4. Main Experiment Logic (LOSO + Evaluation)
# -------------------------------------------------------------
def run_experiment(root_dir, sparsity_levels=[0.1, 0.4]):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    subjects = sorted({f.split('-')[1] for f in os.listdir(root_dir) if f.endswith('.txt')})
    results = {'SPGN': {sp: [] for sp in sparsity_levels}, 'ConcatNet': {sp: [] for sp in sparsity_levels}}
    
    # 用於儲存最後一個 Fold 的視覺化數據
    final_y_true, final_y_pred = [], []
    final_features, final_labels = [], []

    for sp in sparsity_levels:
        print(f"\n--- Testing Sparsity: {sp} ---")
        for model_name in ['SPGN', 'ConcatNet']:
            pbar = tqdm(subjects, desc=f"{model_name}")
            for test_sid in pbar:
                train_sids = [s for s in subjects if s != test_sid]
                train_loader = DataLoader(PressureDataset(root_dir, train_sids, sparsity=sp), batch_size=16, shuffle=True)
                test_loader = DataLoader(PressureDataset(root_dir, [test_sid], sparsity=sp), batch_size=1)

                model = SparsePointGridNet().to(device) if model_name == 'SPGN' else ConcatNet().to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                criterion = nn.CrossEntropyLoss()

                # Train
                model.train()
                for _ in range(15): # Epochs
                    for g, p, l in train_loader:
                        g, p, l = g.to(device), p.to(device), l.to(device)
                        optimizer.zero_grad()
                        loss = criterion(model(g, p), l)
                        loss.backward()
                        optimizer.step()

                # Eval
                model.eval()
                all_p, all_l = [], []
                with torch.no_grad():
                    for g, p, l in test_loader:
                        g, p = g.to(device), p.to(device)
                        output = model(g, p)
                        all_p.append(output.argmax(1).item())
                        all_l.append(l.item())
                        
                        # 特徵收集 (僅針對 SPGN 在最低稀疏度時)
                        if model_name == 'SPGN' and sp == min(sparsity_levels):
                            feat = model(g, p, return_features=True)
                            final_features.append(feat.cpu().numpy().flatten())
                            final_labels.append(l.item())

                acc = accuracy_score(all_l, all_p)
                results[model_name][sp].append(acc)
                pbar.set_postfix(acc=f"{acc:.2f}")

                if sp == min(sparsity_levels) and model_name == 'SPGN':
                    final_y_true.extend(all_l); final_y_pred.extend(all_p)

    # ---------------------------------------------------------
    # Export CSV + plot
    # ---------------------------------------------------------
    save_dir='results'
    csv_path = os.path.join(save_dir, 'sparsity_results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Sparsity', 'MeanAcc', 'StdAcc'])
        for name in ['SPGN', 'ConcatNet']:
            for sp in sparsity_levels:
                accs = results[name][sp]
                writer.writerow([name, sp, np.mean(accs), np.std(accs)])

    # Plot
    plt.figure(figsize=(8, 5))
    for name in ['SPGN', 'ConcatNet']:
        means = [np.mean(results[name][sp]) for sp in sparsity_levels]
        plt.plot(sparsity_levels, means, marker='o', label=name)

    plt.xlabel('Remaining Point Ratio')
    plt.ylabel('LOSO Accuracy')
    plt.title('Sparsity Robustness Comparison')
    plt.legend()
    plt.grid(True)

    fig_path = os.path.join(save_dir, 'sparsity_comparison.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Saved figure: {fig_path}")
    print(f"Saved table: {csv_path}")
    
    # 額外視覺化
    plot_confusion_matrix(final_y_true, final_y_pred, range(6), 'results/cm_spgn.png')
    plot_tsne(np.array(final_features), np.array(final_labels), 'results/tsne_spgn.png')
    print("Optimization Complete. Results saved in /results")

if __name__ == "__main__":
    root = os.path.dirname(os.path.realpath(__file__))+ os.path.sep+'data'
    # sparsity = [0.1, 0.4]
    sparsity = [0.1, 0.2, 0.3, 0.4]
    run_experiment(root, sparsity)
    pass
