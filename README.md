# hsViT Model (local ver.)

## 概览

`HyperspectralTrainer` for highlighting the ~~superiority~~ of hsViT over RGB-ViT.

毕业设计用, hsViT的本地训练部分

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
│   ├── config.yaml         # 配置文件（包含采样、梯度累积、EMA 等参数）
│   └── loader.py
├── model/                  # 模型架构层
│   ├── __init__.py
│   ├── common_vit.py       # CommonViT
│   ├── lola_vit.py         # LoLAViT
│   ├── u_net.py
│   └── rgb_vit.py
├── pipeline/               # 工作流类
│   ├── __init__.py
│   ├── core.py             # 封装总线
│   ├── trainer.py          # 训练算法层
│   ├── dataset.py          # 数据处理层
│   ├── analyzer.py
│   ├── monitor.py          # 栈外服务
│   └── visualize.py
├── src/                    # 图像资源
│   └── *.png
├── run.py                  # 用户入口
├── monitor.py              # 资源监视器
├── requirements.txt
├── LOG.md                  # 项目日志
└── README.md
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

---

## 架构与数据流

### 数据处理层的采样策略

基本的全局均匀采样 + 前景、边界和难分类的重采样。

**策略**：

- **Random**：全局随机均匀采样；
- **FG**：${w^{-1}}$ 加权的前景采样，偏好难分类；
- **Boundary**：Sobel算子边缘采样；

### 训练层的早停指标

```python
composite = 0.60 * mIoU + 0.30 * FG_Dice + 0.10 * BG_IoU
```

| 权重 | 指标 | 说明 |
| :--: | :--: | :--: |
|**0.60**| mIoU | 综合 |
|**0.30**|FG Dice| 前景 |
|**0.10**|BG IoU| 背景 |

## 消融实验 (Ablation Study)

通过 `run.py` 执行结构消融实验, 逐步缩参, 兼顾 **性能-参数量**。

```bash
python run.py \
    <--tag/-t> { /reduced/tiny/mini/2layer} \
    <--model/-m> { /common/lola/unet} \
    <--epoch/-e> {num_epochs}
```

1. **模型架构**：预定义 10 组配置(CommonViT * 5 + LoLA_hsViT * 5)：

|      模型名称      | dim |  depths  | num_heads |
| :-----------------: | :-: | :-------: | :--------: |
| **fullstack** | 96 | [3, 4, 5] | [4, 8, 16] |
|  **reduced**  | 64 | [2, 3, 3] | [4, 8, 16] |
|   **tiny**   | 64 | [2, 2, 2] | [4, 8, 16] |
|   **mini**   | 48 | [1, 1, 2] | [2, 4, 8] |
|  **2layer**  | 32 |  [1, 1]  |   [2, 4]   |

- `dim`：基础特征维度 (96 -> 32)
- `depths`：Transformer 层深度 ([3,4,5] -> [1,1,1])
- `num_heads`：分类头数量 ([4,8,16] -> [2,4,8])

对选定的配置顺序训练，结束后回收显存。按综合评分排名, **保留**每类模型中 Balance Score 最高的 checkpoint (`.onnx`)和**所有实验的可视化结果**。

## 指标

|指标|定义|说明|
|:---------------------: | :---: | :------------: |
|**Eval Accuracy**|测试集最优准确率|目前是内部测试|
|**Overfit Gap**|`best_train_acc - best_eval_acc`|过拟合程度|
|**Total Params**|模型总参数量|⭐|
|**Kappa**|Cohen's Kappa 系数|分类一致性|

### 玩具

人工评估性能-参数量的复合指标定义为：

$$
\text{score} = \text{Acc}_{\text{eval}} - 1.5 \cdot \max(0,\; \text{Gap} - \tau) - \mu \cdot \log_{10}(\text{Params})
$$

其中：

- $\tau$：过拟合阈值(默认 8%), Gap ≤ τ 时无惩罚
- $\mu$：参数量惩罚权重(默认 2.0), 通过因子 $\log_{10}$ 平滑

**说明**：准确率为正向收益，过拟合间隙和参数量为惩罚项，认为 score 最高的配置是“**性能恰好不溢出且参数量最小**”的平衡点。

### Loss

采用密集分割任务的复合损失函数，由三个互补的损失分量组成：

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{focal}} + \lambda_{\text{dice}} \cdot \mathcal{L}_{\text{dice}} + \lambda_{\text{boundary}} \cdot \mathcal{L}_{\text{boundary}}
$$

#### 1. Focal Loss - 类别偏倚

$$
\mathcal{L}_{\text{focal}}(p_t) = -\alpha_t \cdot (1 - p_t)^\gamma \cdot \log(p_t)
$$

- 通过焦点项 $(1-p_t)^\gamma$ 对易分样本降权
- 优先训练难分样本，增强对前景类的关注
- 支持 label smoothing 防止过度自信
- 支持 ignore index 掩膜，处理边界和背景

#### 2. Soft Dice Loss - 区域匹配

$$
\mathcal{L}_{\text{dice}} = 1 - \frac{2 \sum_{c} w_c \cdot \text{Dice}_c}{\sum_{c} w_c}
$$

- 多类 Dice
- 利好难分类的权重 $w_c$
- 与 Focal Loss 互补，更关注区域整体性

#### 3. Boundary CE Loss - 边界约束

$$
\mathcal{L}_{\text{boundary}} = \frac{\sum_{(i,j) \in B} \text{CE}_{ij}}{\sum_{(i,j) \in B} 1}
$$

其中 $B$ 为膨胀后的边界区域（Sobel 检测 + 膨胀操作）

- 对类别边界区域应用额外的交叉熵约束
- 增强FG/BG边界的清晰度
- 形态学处理 (dilation) 扩大边界影响

#### 配置参数

| 配置参数 | 默认值 | 说明 |
| :--: | :--: | :--: |
|`gamma`|2.0|Focal Loss 的聚焦指数|
|`dice_weight`|0.35|Dice Loss 权重$\lambda_{\text{dice}}$|
|`boundary_weight`|0.20|Boundary Loss 权重$\lambda_{\text{boundary}}$|
|`boundary_dilation`|1|边界膨胀核大小|
|`dice_absent_prior`|0.05|缺失类的先验权重|
|`label_smoothing`|0.0|Label smoothing 系数（0-1）|

## To-do & Bugs

- [X] ~~背景类完全忽视~~
  实现四路采样 (FG+Boundary+BG+Random)
  配置：`sampler_mix_*` 参数可调
- [X] ~~验证集随机波动~~
  全集验证
  配置：`eval_use_batch_cap`
- [X] ~~结果文件的读写进程冲突~~
  解释器锁
- [X] ~~``patch_size``与位置嵌入 ``pos_embed``维度不匹配~~
  动态插值 `_resize_pos_embed(h, w)`
- [X] Nvidia Clara API 边缘部署支持 (√：Nvidia MONAI API)
- [ ] 完善RGB ViT
- [ ] StreamCat 测试
