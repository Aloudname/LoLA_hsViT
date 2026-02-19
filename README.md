# LoLA-hsViT Model (local ver.)
## 概览

`HyperspectralTrainer` for highlighting the superiority of LoLA-hsViT over common ViT.

---
## 工作结构
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
│   └── process_monitor.py
├── Utils/      # 杂项
│   ├── __init__.py
│   └── HDR2MAT.py      # .hdr -> .mat转换器
├── train.py    # 训练入口
├── README.md
├── requirements.txt
└── ...

---

## 输出目录结构

```
output/
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

## bugs & to-do
### bugs
- ``patch_size``与位置嵌入``pos_embed``维度不匹配:
    因：pos_embed由固定参数传入
            File "./model/lola_vit.py", line 773, in ``forward_features``:
                x = x + self.pos_embed
RuntimeError: The size of tensor a: x (15) must match the size of tensor b:  pos_embed (7) at non-singleton dimension 2.
- config作为参数传入多个层(LoLA-hsViT, )
### to-do
- Nvidia Clara API支持
- 外源数据的泛化测试
- 多模态数据支持
- 训练日志WandB
