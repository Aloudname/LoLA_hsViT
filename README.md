# LoLA-hsViT Model (local ver.)
## 概览

`HyperspectralTrainer` for highlighting the superiority of LoLA-hsViT over common ViT.

毕业设计用, LoLA-hsViT的本地部署部分

ablation.py作程序入口, 支持消融实验

---
## 工作结构
```
LoLA_hsViT/
├── config/     # 参数设置
│   ├── __init__.py
│   ├── config.yaml
│   └── config_yaml.py
├── model/    # 模型定义
│   ├── __init__.py
│   ├── u_net.py
│   ├── lola_vit.py
│   ├── common_vit.py
│   └── ...
├── pipeline/     # 工作流类
│   ├── __init__.py
│   ├── dataset.py
│   ├── trainer.py
│   └── monitor.py
├── ablation.py # 消融实验入口
├── LOG.md
├── README.md
├── requirements.txt
└── ...
```

---

## 输出目录结构

```
outputs/
│
├── model_A/
│   ├── models/
│   │   └── model_A_best.pth
│   ├── CAM/
│   │   ├── epoch_010.png
│   │   ├── epoch_015.png
│   │   ├── best_result.png
│   │   └── ...
│   ├── training_curves.png
│   └── confusion_matrix.png
├── model_B/
│   ├── models/
│   │   └── model_B_best.pth
│   ├── CAM/
│   │   ├── epoch_010.png
│   │   ├── epoch_015.png
│   │   ├── best_result.png
│   │   └── ...
│   ├── training_curves.png
│   └── confusion_matrix.png
├── log/
│   └── monitor.log
└── ablation/
    ├── model_A/
    ├── model_B/
    └── ...

```

---

## 消融实验 (Ablation Study)

通过 `ablation.py` 自动执行结构消融实验, 逐步缩参, 兼顾 **性能-参数量**。

### 工作流

```
搜索空间定义 -> 自动消融训练 -> 评估指标收集 -> 评估排名 -> 保留最优模型 -> 生成分析图表
```

1. **搜索空间**：预定义 12 组配置(CommonViT * 6 + LoLA_hsViT * 6), 覆盖：
   - `dim`：基础特征维度 (96 -> 32)
   - `depths`：Transformer 层深度 ([3,4,5] -> [1,1,1])
   - `mlp_ratio`：MLP 隐藏层比率 (4.0 -> 2.0)
   - `r` / `lora_alpha`：LoRA 秩与缩放 (16/32 -> 4/8, 仅 LoLA_hsViT)

2. **自动消融训练**：对每组配置调用 `hsTrainer` 完成完整训练, 并在每次训练结束后回收显存。

3. **最优选择与清理**：按综合评分排名, **保留**每类模型中 Balance Score 最高的 checkpoint (.pth)和**所有实验的可视化结果**。

### 评估标准

#### 核心指标
| 指标 | 定义 | 说明 |
|------|------|------|
| **Eval Accuracy** | 测试集最优准确率 | 目前是内部测试 |
| **Overfit Gap** | `best_train_acc - best_eval_acc` | 过拟合程度 |
| **Total Params** | 模型总参数量 | ⭐ |
| **Kappa** | Cohen's Kappa 系数 | 分类一致性 |

#### 评估指标

评估性能-参数量的复合指标定义为：

$$\text{score} = \text{Acc}_{\text{eval}} - 1.5 \cdot \max(0,\; \text{Gap} - \tau) - \mu \cdot \log_{10}(\text{Params})$$

其中：
- $\tau$：过拟合阈值(默认 8%), Gap ≤ τ 时无惩罚
- $\mu$：参数量惩罚权重(默认 2.0), 通过因子 $\log_{10}$ 平滑

**说明**：准确率为正向收益, 过拟合间隙和参数量为惩罚项, score 最高的配置即为"性能恰好不溢出且参数量最小"的平衡点

### 启动

```bash
# 干启动: 预览搜索空间和参数量(不训练)
python ablation.py --dry-run

# 完整消融实验
python ablation.py -e 10 -p 4

# 仅消融某模型
python ablation.py -e 10 -p 4 --model lola
python ablation.py -e 10 -p 4 --model common

# 中断后恢复, 如从第 5 个配置继续
python ablation.py -e 10 -p 4 --resume 5

# 调整过拟合阈值
python ablation.py -e 10 -p 4 --gap-threshold 10
```

### 输出目录结构

```
output/ablation/
├── LoLA_hsViT_full_stack/          # 各配置的训练可视化
│   ├── CAM/
│   ├── training_curves.png
│   └── confusion_matrix.png
├── LoLA_hsViT_shallow_dim/
│   ├── CAM/
│   ├── training_curves.png
│   └── confusion_matrix.png
├── CommonViT_reduced/
│   ├── CAM/
│   ├── training_curves.png
│   └── confusion_matrix.png
├── ...
└── summary/                      # 汇总分析
    ├── ablation_results.json     # 全部数值结果
    ├── best_models.txt           # 最优模型配置
    ├── param_vs_accuracy.png     # 参数量 vs 精度散点图
    ├── overfit_analysis.png      # 过拟合间隙柱状图
    ├── pareto_frontier.png       # Pareto 前沿
    ├── balance_scores.png        # 综合评分排名
    └── ablation_table.png        # 汇总表格
```

---

## bugs & to-do
### bugs
``patch_size``与位置嵌入``pos_embed``维度不匹配

因: pos_embed由固定参数传入
```
File "./model/lola_vit.py", line 773, in ``forward_features``:
    x = x + self.pos_embed
RuntimeError: The size of tensor a: x (15) must match the size of tensor b:  pos_embed (7) at non-singleton dimension 2.
```
- config作为参数冗余传入多个层(LoLA-hsViT, MAThsDataLoader)
### to-do
- Nvidia Clara API边缘部署支持
- 外源数据的泛化测试
- 多模态数据支持
- 训练日志WandB
