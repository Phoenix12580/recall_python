# recall-python

这是一个参考 [`lcrawlab/recall`](https://github.com/lcrawlab/recall) R 包核心逻辑实现的 Python 包。

当前实现覆盖了 R 包中最核心、最可迁移的部分：

- ZIP 参数估计（`estimate_zi_poisson`）
- ZIP 随机采样（`rzipoisson`）
- 负二项参数估计（`estimate_negative_binomial`，使用稳健矩估计）
- 人工 knockoff 特征构造（`augment_with_artificial_variables`）
- knockoff 阈值与筛选（`knockoff_threshold`、`compute_knockoff_filter`）

> 说明：R 版本依赖 Seurat/scDesign3。Python 版为保持轻量与可用性，采用通用 `pandas.DataFrame + labels` 接口，便于对接 `scanpy/anndata` 或任意矩阵工作流。

## 安装

```bash
pip install -e .
```

## 快速示例

```python
import numpy as np
import pandas as pd
from recall_python import augment_with_artificial_variables, compute_knockoff_filter

rng = np.random.default_rng(42)
X = pd.DataFrame({
    "geneA": rng.poisson(1.2, 100),
    "geneB": rng.poisson(2.0, 100),
    "geneC": rng.poisson(0.8, 100),
})
labels = np.array([0] * 50 + [1] * 50)

X_aug = augment_with_artificial_variables(X, null_method="ZIP", random_state=42)
result = compute_knockoff_filter(X_aug, labels, cluster1=0, cluster2=1, q=0.1)
print(result.threshold)
print(result.selected_features.head())
```
