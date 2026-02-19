# HyperspectralTrainer API 文档

## 概览

`HyperspectralTrainer` 是一个通用的超光谱图像分类训练器，提供清晰简洁的接口用于训练任意深度学习模型。

### 核心特性

- ✅ **通用接口**: 支持任意 `nn.Module` 模型
- ✅ **3个基本参数**: `config` (Munch), `dataLoader` (AbstractHyperspectralDataset), `epochs` (int)
- ✅ **完整训练管线**: 数据读取 → 模型创建 → 优化器初始化 → 训练 → 验证 → 可视化
- ✅ **生产级功能**: 混合精度训练、梯度累积、学习率预热、早停、模型保存
- ✅ **丰富可视化**: 训练曲线、混淆矩阵、CAM 激活图
- ✅ **自适应设备**: 自动支持 GPU/CPU，智能内存管理

---

## 快速开始

### 最简单的用法 (3 步)

```python
from pipeline.trainer import HyperspectralTrainer
from pipeline.dataset import MatHyperspectralDataset
from munch import Munch
import torch.nn as nn

# 第1步: 定义模型创建函数
def create_model():
    return MyHSIModel(num_classes=9, num_bands=15)

# 第2步: 准备配置和数据
config = Munch({'common': {'lr': 1e-4, 'use_amp': True}, 'num_workers': 4})
dataLoader = MatHyperspectralDataset(...)

# 第3步: 创建并训练
trainer = HyperspectralTrainer(
    config=config,
    dataLoader=dataLoader,
    epochs=50,
    model_fn=create_model,
    model_name='MyModel',
    output_dir='./output'
)
results = trainer.train()
```

---

## API 参考

### 初始化 (Constructor)

```python
HyperspectralTrainer(
    config: Munch,
    dataLoader: AbstractHyperspectralDataset,
    epochs: int,
    model_fn: Callable[..., nn.Module],
    model_name: str = "default_model",
    output_dir: str = "./output"
)
```

#### 参数说明

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `config` | `Munch` | ✅ | 配置对象，包含所有超参数 |
| `dataLoader` | `AbstractHyperspectralDataset` | ✅ | 数据加载器实例 |
| `epochs` | `int` | ✅ | 训练轮数 |
| `model_fn` | `Callable` | ✅ | 模型创建函数，无参数，返回 `nn.Module` |
| `model_name` | `str` | ❌ | 模型名字，用于文件保存，默认 `"default_model"` |
| `output_dir` | `str` | ❌ | 输出目录，默认 `"./output"` |

#### Config 中的关键参数

```python
config = Munch({
    # 必需
    'common': {
        'lr': 1e-4,                    # 学习率
        'weight_decay': 1e-5,          # L2正则化
        'use_amp': True,               # 混合精度训练 (仅GPU)
    },
    
    # 可选 (有默认值)
    'device_type': 'cuda',             # 'cuda' 或 'cpu'
    'num_workers': 4,                  # 数据加载器线程数
    
    # 调度器参数
    'common': {
        'scheduler': {
            'T_0': 10,                 # Cosine Annealing 初始周期
            'T_mult': 2,               # 周期倍增因子
            'eta_min': 1e-6            # 最小学习率
        },
        'warmup_epochs': 5,            # 预热轮数
        'label_smoothing': 0.1,        # 标签平滑
        'eval_batch_size': 64,         # 验证batch大小
        'eval_interval': 1,            # 验证间隔(epoch)
        'patience': 20,                # 早停耐心值
        'grad_clip': 1.0               # 梯度裁剪值
    }
})
```

---

### 主方法

#### train()

```python
results = trainer.train(debug_mode: bool = False) -> Dict[str, float]
```

开始训练.

**参数:**
- `debug_mode` (bool): 若为 True，每个 epoch 都生成 CAM 可视化

**返回值:**
```python
{
    'best_epoch': int,              # 最优模型的epoch (1-indexed)
    'best_accuracy': float,         # 最优准确率 (%)
    'final_accuracy': float,        # 最终测试准确率 (%)
    'final_kappa': float,           # 最终Kappa系数
    'training_time': float,         # 总耗时 (秒)
    'model_path': str              # 最优模型保存路径
}
```

**示例:**
```python
results = trainer.train(debug_mode=False)
print(f"Best Accuracy: {results['best_accuracy']:.2f}%")
print(f"Model saved at: {results['model_path']}")
```

---

#### evaluate()

```python
loss, acc, kappa, predictions, targets = trainer.evaluate() 
    -> Tuple[float, float, float, np.ndarray, np.ndarray]
```

在测试集上评估模型 (不更新权重)。

**返回值:**
- `loss` (float): 平均损失
- `acc` (float): 准确率 (%)
- `kappa` (float): Kappa系数 (%)
- `predictions` (np.ndarray): 预测标签 [N]
- `targets` (np.ndarray): 真实标签 [N]

**示例:**
```python
loss, acc, kappa, pred, target = trainer.evaluate()
print(f"Test Accuracy: {acc:.2f}%, Kappa: {kappa:.2f}%")
```

---

#### train_epoch()

```python
loss, acc = trainer.train_epoch(epoch: int) -> Tuple[float, float]
```

训练单个 epoch (内部使用，一般不需要手动调用)。

**返回值:**
- `loss` (float): epoch 平均损失
- `acc` (float): epoch 准确率 (%)

---

#### predict()

```python
predictions = trainer.predict(hsi: torch.Tensor) -> torch.Tensor
```

对新样本进行预测。

**参数:**
- `hsi` (torch.Tensor): 输入数据，形状 [B, C, H, W] 或 [B, H, W, C]

**返回值:**
- `predictions` (torch.Tensor): 预测标签 [B]

**示例:**
```python
# 从测试集读取样本
batch_hsi, batch_labels = next(iter(trainer.test_loader))
predictions = trainer.predict(batch_hsi)
accuracy = (predictions == batch_labels).float().mean()
print(f"Batch Accuracy: {accuracy:.4f}")
```

---

#### load_best_model()

```python
model = trainer.load_best_model() -> nn.Module
```

加载最优模型权重。

**返回值:**
- `model` (nn.Module): 加载权重后的模型

**示例:**
```python
best_model = trainer.load_best_model()
# 现在可以在推理中使用 best_model
```

---

## 内部方法 (高级)

### 数据处理

#### _unpack_batch()

```python
hsi, labels = trainer._unpack_batch(batch_data) -> Tuple[torch.Tensor, torch.Tensor]
```

解析 DataLoader 返回的 batch 数据，支持多种格式:
- `(hsi, labels)`
- `(hsi, pretrained_features, labels)`

---

#### _normalize()

```python
normalized_hsi = trainer._normalize(hsi) -> torch.Tensor
```

对超光谱数据进行标准化 (沿空间维度计算均值和标准差)。

---

### 模型管理

#### _save_model()

```python
trainer._save_model() -> None
```

保存最优模型到 `output_dir/models/{model_name}_best.pth`。

---

### 可视化

#### _generate_cam()

```python
trainer._generate_cam(epoch: int) -> None
```

生成指定 epoch 的样本可视化 (RGB 合成、灰度强度、光谱曲线)，保存到 `output_dir/CAM/epoch_XXX.png`。

---

#### _plot_training_curves()

```python
trainer._plot_training_curves() -> None
```

绘制训练/验证的损失和准确率曲线，保存到 `output_dir/training_curves.png`。

---

#### _plot_confusion_matrix()

```python
trainer._plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> None
```

绘制混淆矩阵热力图，保存到 `output_dir/confusion_matrix.png`。

---

## 配置示例

### 示例1: 轻量级配置 (CPU训练)

```python
config = Munch({
    'device_type': 'cpu',
    'num_workers': 2,
    'common': {
        'lr': 5e-5,
        'weight_decay': 1e-5,
        'use_amp': False,
        'warmup_epochs': 3,
        'patience': 15,
        'eval_batch_size': 16,
        'scheduler': {'T_0': 5, 'T_mult': 2, 'eta_min': 1e-7}
    }
})
```

### 示例2: 标准配置 (GPU训练)

```python
config = Munch({
    'device_type': 'cuda',
    'num_workers': 4,
    'common': {
        'lr': 1e-4,
        'weight_decay': 1e-5,
        'use_amp': True,
        'warmup_epochs': 5,
        'label_smoothing': 0.1,
        'patience': 20,
        'eval_batch_size': 64,
        'scheduler': {'T_0': 10, 'T_mult': 2, 'eta_min': 1e-6}
    }
})
```

### 示例3: 高性能配置 (大GPU)

```python
config = Munch({
    'device_type': 'cuda',
    'num_workers': 8,
    'common': {
        'lr': 2e-4,
        'weight_decay': 1e-4,
        'use_amp': True,
        'warmup_epochs': 10,
        'label_smoothing': 0.2,
        'patience': 25,
        'eval_batch_size': 128,
        'grad_clip': 1.5,
        'scheduler': {'T_0': 15, 'T_mult': 1.5, 'eta_min': 1e-7}
    }
})
```

---

## 模型定义示例

### 最小CNN模型

```python
import torch.nn as nn

class TinyHSICNN(nn.Module):
    def __init__(self, num_classes=9, num_bands=15):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(num_bands, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = self.features(x).flatten(1)
        return self.fc(x)

# 使用
trainer = HyperspectralTrainer(
    config=config,
    dataLoader=dataLoader,
    epochs=30,
    model_fn=lambda: TinyHSICNN(num_classes=9, num_bands=15),
    model_name='TinyHSICNN'
)
```

### 带LoRA的预训练模型

```python
import torch.nn as nn
from peft import get_peft_model, LoraConfig

def create_lora_model():
    # 加载预训练模型
    model = torch.hub.load(...)
    
    # 应用LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=['q_proj', 'v_proj'],
        lora_dropout=0.05
    )
    model = get_peft_model(model, lora_config)
    return model

trainer = HyperspectralTrainer(
    config=config,
    dataLoader=dataLoader,
    epochs=30,
    model_fn=create_lora_model,
    model_name='PretrainedWithLoRA'
)
```

---

## 完整工作流示例

```python
from pipeline.trainer import HyperspectralTrainer
from pipeline.dataset import MatHyperspectralDataset
from munch import Munch
import torch.nn as nn

# 1️⃣ 定义模型
class MyHSIModel(nn.Module):
    def __init__(self, num_classes, num_bands):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(num_bands, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)

# 2️⃣ 准备配置
config = Munch({
    'device_type': 'cuda',
    'num_workers': 4,
    'common': {
        'lr': 1e-4,
        'weight_decay': 1e-5,
        'use_amp': True,
        'warmup_epochs': 5,
        'patience': 20,
        'scheduler': {'T_0': 10, 'T_mult': 2, 'eta_min': 1e-6}
    }
})

# 3️⃣ 准备数据
dataLoader = MatHyperspectralDataset(
    data_path='/path/to/data.mat',
    label_path='/path/to/labels.mat',
    data_key='data',
    label_key='labels',
    target_names=['Class1', 'Class2', ..., 'Class9'],
    num_classes=9,
    patch_size=31,
    batch_size=32,
    test_rate=0.2,
    pca_components=15,
    pin_memory=True
)

# 4️⃣ 创建训练器
trainer = HyperspectralTrainer(
    config=config,
    dataLoader=dataLoader,
    epochs=50,
    model_fn=lambda: MyHSIModel(num_classes=9, num_bands=15),
    model_name='MyHSIModel_v1',
    output_dir='./outputs/exp1'
)

# 5️⃣ 开始训练
results = trainer.train(debug_mode=False)

# 6️⃣ 查看结果
print(f"\n{'='*60}")
print(f"Best Epoch: {results['best_epoch']}")
print(f"Best Accuracy: {results['best_accuracy']:.2f}%")
print(f"Final Kappa: {results['final_kappa']:.2f}%")
print(f"Training Time: {results['training_time']:.2f}s")
print(f"Model saved at: {results['model_path']}")
print(f"{'='*60}")

# 7️⃣ 使用最优模型进行推理
best_model = trainer.load_best_model()
test_hsi, test_labels = next(iter(trainer.test_loader))
predictions = trainer.predict(test_hsi)
```

---

## 故障排除

### 问题1: CUDA 内存不足

**解决方案:**
```python
config['common']['eval_batch_size'] = 32  # 减小验证batch大小
config['num_workers'] = 0                 # 禁用多进程数据加载
```

### 问题2: CPU 训练太慢

**解决方案:**
```python
config['device_type'] = 'cuda'     # 切换到GPU
config['num_workers'] = 8          # 增加数据加载线程
config['common']['warmup_epochs'] = 2  # 减少预热轮数
```

### 问题3: 模型过拟合

**解决方案:**
```python
config['common']['label_smoothing'] = 0.2  # 增加标签平滑
config['common']['weight_decay'] = 1e-4    # 增加L2正则化
config['common']['patience'] = 10          # 提早早停
```

---

## 输出目录结构

```
output_dir/
├── models/
│   └── {model_name}_best.pth          # 最优模型权重
├── CAM/
│   ├── epoch_001.png
│   ├── epoch_005.png
│   └── ...
├── training_curves.png                # 训练/验证曲线
└── confusion_matrix.png                # 混淆矩阵
```

---

## 性能优化建议

1. **启用混合精度训练**: `use_amp=True` (GPU自动速度提升 20-30%)
2. **增加数据加载线程**: `num_workers=4~8` (取决于CPU核数)
3. **使用学习率预热**: `warmup_epochs=5~10` (稳定训练初期)
4. **梯度裁剪**: `grad_clip=1.0` (防止梯度爆炸)
5. **早停机制**: `patience=20` (自动停止无改进的训练)

