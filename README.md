# LoLA-hsViT Model (local ver.)
## 概览

`HyperspectralTrainer` for highlighting the ~~superiority~~ of LoLA-hsViT over common ViT.

毕业设计用, LoLA-hsViT的本地部署部分

run.py作程序入口, 支持消融实验



### 工作流

```
                 模型架构层
              ↗  model.py ↘  
    配置文件                  训练算法层 -> 封装总线 -> 用户层接口
  config.yaml ↘           ↗ trainer.py   core.py     run.py
                 数据处理层
                 dataset.py
```

## 工作结构
```
LoLA_hsViT/
├── config/                 # 参数设置
│   ├── __init__.py
│   ├── config.yaml         # 配置文件
│   └── config_yaml.py
├── model/                  # 模型架构层
│   ├── __init__.py
│   ├── u_net.py
│   ├── lola_vit.py
│   ├── common_vit.py
│   └── ...
├── pipeline/               # 工作流类
│   ├── __init__.py
│   ├── core.py             # 封装总线
│   ├── trainer.py          # 训练算法层
│   ├── dataset.py          # 数据处理层
│   ├── monitor.py          # 栈外服务
│   ├── analyzer.py
│   └── visualize.py
├── src/                    # .md插图资源
│   ├── xxx.png
│   └── ...
├── run.py                  # 用户层入口
├── monitor.py              # 资源监视器 (服务调用)
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
├── model_A_tag/
│   ├── models/
│   │   └── model_A_best.onnx
│   ├── CAM/
│   │   ├── epoch_010.png
│   │   ├── epoch_015.png
│   │   ├── best_result.png
│   │   └── ...
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   └── ...
├── model_B_tag/
│   ├── models/
│   │   └── model_B_best.onnx
│   ├── CAM/
│   │   ├── epoch_010.png
│   │   ├── epoch_015.png
│   │   ├── best_result.png
│   │   └── ...
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   └── ...
├── log/
│   └── 2_31_2026_11_45_14.log
└── ...

```

---

## 消融实验 (Ablation Study)

通过 `run.py` 执行结构消融实验, 逐步缩参, 兼顾 **性能-参数量**。

```bash
bash

python run.py \
    <--tag/-t> { /reduced/tiny/mini/2layer} \
    <--model/-m> { /common/lola/unet} \
    <--epoch/-e> {num_epochs}
```


1. **模型架构**：预定义 10 组配置(CommonViT * 5 + LoLA_hsViT * 5)：

| 模型名称 | dim | depths | num_heads |
| :---: | :---: | :---: | :---: |
| **fullstack** | 96 | [3, 4, 5] | [4, 8, 16] |
| **reduced** | 64 | [2, 3, 3] | [4, 8, 16] |
| **tiny** | 64 | [2, 2, 2] | [4, 8, 16] |
| **mini** | 48 | [1, 1, 2] | [2, 4, 8] |
| **2layer** | 32 | [1, 1] | [2, 4] |

   - `dim`：基础特征维度 (96 -> 32)
   - `depths`：Transformer 层深度 ([3,4,5] -> [1,1,1])
   - `num_heads`：分类头数量 ([4,8,16] -> [2,4,8])

1. **自动消融**：对选定的配置，顺序训练, 模型结束后立刻回收显存。

2. **最优选择与清理**：按综合评分排名, **保留**每类模型中 Balance Score 最高的 checkpoint (`.onnx`)和**所有实验的可视化结果**。

### 评估标准

#### 核心指标
| 指标 | 定义 | 说明 |
|:------:|:------:|:------:|
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

**说明**：准确率为正向收益, 过拟合间隙和参数量为惩罚项, score 最高的配置即为"性能恰好不溢出且参数量最小"的平衡点。

## bugs & to-do
### bugs
- ~~``patch_size``与位置嵌入``pos_embed``维度不匹配~~
    因: pos_embed由固定参数传入
```
File "./model/lola_vit.py", line 773, in `forward_features`:
    x = x + self.pos_embed
RuntimeError: The size of tensor a: x (15) must match the size of tensor b:  pos_embed (7) at non-singleton dimension 2.
```

- ~~config作为参数冗余传入多个层(LoLA-hsViT, MAThsDataLoader)~~
- 
### to-do
- Nvidia Clara API边缘部署支持(√：Nvidia Triton Server)
- 外源数据的泛化测试
- 多模态数据支持
- 训练日志WandB
