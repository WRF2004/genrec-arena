# 执行记录（2026-10-23，客户端日期）

## 已实际执行

- 安装项目并解析依赖：Python 3.12.14，torch 2.14.1，transformers 4.57.6，PEFT 0.21.2（精确版本见 smoke.json / requirements-tested.txt）。
- `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q`：**11 passed**，6.44 秒。一个无害的 checkpoint 目录 mtime 排序提示，Trainer 自动按 checkpoint 数字排序。
- `bash scripts/smoke.sh`：生成数据、双路 SID、LoRA 训练、最佳模型保存、重新加载、Popularity 基线、验证调权、独立测试、命令行推荐，全部退出码 0。
- 独立集成测试：两轮训练 vs 第一轮 checkpoint 恢复；最后步数相同、logits 逐元素完全一致。保存/加载后 logits 也完全一致。
- 冻结编码器接口以本地随机小型 Qwen2 快照执行，检查向量 shape、归一化、item ID 对齐并接入量化。

## 合成数据结果

24 用户，32 训练期商品，168 条训练样本（双流展开 336），验证/测试各 24 用户。模型为 **随机初始化的两层 Qwen2，hidden_size=48**，不是 Qwen2.5-1.5B 预训练权重。SID 为 2 层、每层 4 码 + 2 位消歧。

| 项目 | 数值 |
|---|---:|
| 单 epoch 优化步数 | 21 |
| 首次记录训练 loss | 3.6561 |
| 最后记录训练 loss | 2.4278 |
| 验证 token loss | 2.437861 |
| 验证集选择语义权重 alpha | 0.25 |
| 测试 Recall@5 | 0.125 |
| 测试 NDCG@5 | 0.101278 |
| 测试目录覆盖率 | 0.1875 |
| Popularity 测试 Recall@5 | 0.083333 |
| Popularity 测试 NDCG@5 | 0.067955 |

小样本随机模型的数值只用于链路验收，不推导真实数据效果或显著性，不据此宣称双流优于单流。原始逐用户预测、每步 loss、SID 诊断和延迟见 [smoke.json](smoke.json)。

## 尚未执行

Qwen2.5-1.5B 预训练权重的 GPU 训练、真实 MovieLens/电商数据实验、完整模型显存与吞吐测试、跨种子统计；CI 配置已提供，远端 Actions 是否运行取决于仓库设置。
