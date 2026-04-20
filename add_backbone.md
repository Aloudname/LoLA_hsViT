# 稳健 & 高复用的预训练 ViT 接入设计指南（LoLA_hsViT）

本文档用于约束 LoLA_hsViT 中 backbone 解耦与预训练接入实现，确保：

- 可复现：配置字段统一、错误显式抛出
- 可扩展：backbone 统一 token 接口
- 可验证：代码行为与文档描述一致

## 1. 关键结论（必须遵守）

1. timm 的 `blocks` 本身不包含位置编码加法。
2. 仅调用 `blocks` 时，必须手动添加位置编码。
3. 推荐做位置编码插值适配，不建议关闭位置编码。
4. 配置字段统一使用 `use_pretrained`，不要混用 `pretrained_backbone` / `pretrained`。
5. 预训练加载失败必须显式报错，不允许静默 fallback。

## 2. 推荐目录结构

```text
model/
├── backbones/
│   ├── __init__.py
│   ├── base.py
│   ├── transformer_scratch.py
│   ├── vit_timm.py
│   └── builder.py
```

## 3. 统一接口与维度约束

```python
class BaseBackbone(nn.Module):
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        输入:
            tokens: [B, N, D]
        输出:
            tokens: [B, N, D]

        约束:
            D 必须与 backbone hidden_dim 对齐。
            若不一致，必须在 backbone 内部显式使用投影层（in/out projection）。
        """
        raise NotImplementedError
```

说明：

- 对 `vit_base_patch16_224`，hidden_dim 通常为 768。
- 当前项目 `embed_dim` 可为 128，因此需要 `in_proj/out_proj` 做对齐。

## 4. timm ViT 封装（正确位置编码路径）

关键点：

- `forward_features` 才是 timm 原生添加 `pos_embed` 的地方。
- 仅拿 `blocks` 使用时，必须自己在 `for blk in blocks` 前执行 `x = x + pos_embed`。

参考实现要点：

```python
x = self.in_proj(tokens)
pos = self._get_pos_embed(token_len=x.shape[1], device=x.device, dtype=x.dtype)
if pos is not None:
    x = x + pos

for blk in self.blocks:
    x = blk(x)

x = self.norm(x)
x = self.out_proj(x)
```

位置编码策略：

- 若 token 长度变化，使用 1D 插值对 `pos_embed` 做长度适配。
- 不建议通过“关闭位置编码”来规避 shape 问题，这会破坏预训练空间先验。

## 5. Backbone 工厂（强约束版）

配置字段统一：

- `model.use_pretrained`: 是否使用 timm ViT 包装器
- `model.pretrained_weights`: 是否加载预训练权重
- `model.backbone_name`: timm 模型名

参考实现：

```python
def build_backbone(config):
    use_pretrained = bool(config.get("use_pretrained", False))
    backbone_name = str(config.get("backbone_name", "vit_small_patch16_224"))
    pretrained_weights = bool(config.get("pretrained_weights", True))

    if use_pretrained:
        try:
            return TimmViTBackbone(
                input_dim=config["embed_dim"],
                name=backbone_name,
                pretrained=pretrained_weights,
                freeze=config["freeze_backbone"],
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load pretrained backbone '{backbone_name}': {e}") from e

    return TransformerBackbone(...)
```

注意：

- 不允许 silent fallback。否则论文复现实验不可控。

## 6. HSIAdapter / RGBViT 接入规范

模型侧仅负责前向计算，不承载训练策略：

```python
self.backbone = build_backbone(backbone_cfg)
tokens = self.backbone(tokens)
```

训练策略（如解冻最后 N 层）放到 Trainer 中控制。

## 7. 训练策略解耦（修正 N 未定义问题）

```python
def set_trainable_layers(model, config):
    N = int(config.model.unfreeze_last_n)
    if N <= 0:
        return

    blocks = list(model.backbone.blocks)
    N = min(N, len(blocks))
    for blk in blocks[-N:]:
        for p in blk.parameters():
            p.requires_grad = True
```

## 8. 配置示例（统一命名）

```yaml
model:
  backbone_name: vit_small_patch16_224
  use_pretrained: true
  pretrained_weights: true
  freeze_backbone: true
  unfreeze_last_n: 0
  spectral_dim: 32
  embed_dim: 128
```

兼容说明：

- 历史字段 `pretrained_backbone`、`backbone_pretrained` 可在代码中做只读兼容。
- 新实验统一使用 `use_pretrained`、`pretrained_weights`。

## 9. 鲁棒性检查清单

1. token 形状检查：`tokens.ndim == 3`
2. token 维度检查：`tokens.shape[-1] == input_dim`
3. 位置编码可用性检查：`pos_embed` shape 与 hidden_dim 一致
4. 失败显式抛错：预训练模型加载失败时 `raise RuntimeError`

## 10. 实验建议

- 小样本优先 `vit_small_patch16_224`，显存和过拟合风险更可控。
- 首次运行 timm 会下载权重，需保证网络或预缓存。
- 对比实验建议至少包含：
  - `use_pretrained: false`（scratch）
  - `use_pretrained: true, pretrained_weights: true`（预训练微调）

## 11. 总结

本项目中预训练 ViT 接入的最小正确集合是：

1. backbone 解耦
2. 统一配置命名
3. 手动添加并插值位置编码
4. 显式维度约束检查
5. 训练策略放在 Trainer
6. 预训练加载失败显式报错
