# 睡姿壓力感測模型程式架構圖

以下使用 Mermaid 繪製可直接貼入論文或技術文件的程式架構圖，涵蓋資料處理、模型雙分支、訓練流程與輸出產物。

```mermaid
flowchart TD
    A[原始壓力矩陣檔案\n(data/*.txt)] --> B[PressureDataset\n__getitem__]
    B --> C[載入 12x18 壓力矩陣\nnp.loadtxt]
    C --> D[物理雜訊模擬\n加性高斯雜訊 + Clip]
    D --> E[稀疏化\n依 threshold 擷取點]
    E --> F[點雲抽樣\n依 sparsity 隨機保留]
    F --> G[點雲補零/截斷\n固定 n_max]

    C --> H[Grid Tensor\nshape: 1x12x18]
    G --> I[Point Tensor\nshape: n_max x 3]
    B --> J[Label\n檔名解析]

    subgraph Model[模型：SparsePointGridNet]
        H --> K[Grid Branch\nConv2d + BN + ReLU]
        K --> L[AdaptiveAvgPool2d\n輸出 3x4]
        L --> M[Flatten]
        M --> N[Linear\n輸出 gamma/beta]

        I --> O[Point Branch\nMLP 3→32→film_dim]
        O --> P[FiLM 融合\n點特徵 * (1+gamma)+beta]
        P --> Q[MaxPool\n跨點聚合]

        M --> R[Grid Projection\nLinear → film_dim]
        Q --> S[Concat\nGrid+Point]
        R --> S
        S --> T[Classifier\nLinear → num_classes]
    end

    subgraph Training[訓練流程]
        J --> U[CrossEntropyLoss\nlabel smoothing]
        T --> U
        U --> V[AdamW Optimizer]
        V --> W[ReduceLROnPlateau]
        W --> X[Epoch Loop\n20 epochs]
    end

    subgraph Evaluation[評估與視覺化]
        T --> Y[LOSO 評估\naccuracy_score]
        Y --> Z[統計結果\nCSV 匯出]
        Y --> AA[混淆矩陣\nSeaborn Heatmap]
        T --> AB[t-SNE 特徵視覺化\nTSNE + Scatter]
    end

    Z --> AC[results/loso_sparsity_metrics.csv]
    AA --> AD[results/journal_cm.png]
    AB --> AE[results/journal_tsne.png]
```

## 圖例說明（對應程式段落）
- **資料處理**：壓力矩陣載入後加入雜訊、依閾值取點、依 sparsity 做隨機採樣並補零/截斷至固定長度。對應 `PressureDataset`。
- **雙分支特徵抽取**：Grid Branch 處理完整矩陣，Point Branch 處理稀疏點雲；透過 FiLM 融合後與 Grid 投影特徵拼接，進入分類器。對應 `SparsePointGridNet`。
- **訓練策略**：使用 AdamW、Label Smoothing、ReduceLROnPlateau 調度學習率，進行 20 epochs 訓練。對應 `run_optimized_experiment`。
- **輸出產物**：LOSO 平均準確率 CSV、混淆矩陣與 t-SNE 視覺化圖。對應 `plot_journal_confusion_matrix`、`plot_robust_tsne`。
