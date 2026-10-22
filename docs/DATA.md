# 数据获取与接入

## 1. Kaggle H&M（主线）

官方入口：https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data

在自己的 Kaggle 账户按页面规则获得数据后，将以下文件放在 `data/raw/`：
- `transactions_train.csv`：customer_id、article_id、t_dat 等。
- `articles.csv`：article_id 与商品内容元数据。
- `sample_submission.csv`：提交用户顺序。

项目不代替你接受比赛条款、不包含数据、不要求把 Kaggle token 上传 GitHub。
主 README 提供 1% 用户采样命令。抽样由 SHA256(user_id) 确定，不受 Python hash 随机化影响。

时间窗口：训练 `< 2020-09-08`，验证 `[2020-09-08, 2020-09-15)`，测试 `>=2020-09-15`（原始训练文件最后一周）。边界按 UTC 解释；官方按日期，避免时区混用。不要额外混入未来记录。

测试历史可使用验证窗口已发生的购买，但模型参数仍是训练期拟合的。复购允许，所以不一律过滤已见商品。

元数据必须是训练截止时已知的静态快照；严禁加入全时间热度、未来销售量等特征。官方静态目录的发布时间假设需要在实验报告中注明。

导出与官方 sample 格式一致的 CSV：

```bash
python -m genrec.cli hm-submit --data data/hm --run runs/hm-gen \
  --transactions data/raw/transactions_train.csv \
  --sample data/raw/sample_submission.csv --output runs/submission.csv --device cuda
```

这是离线模型的格式导出，不保证有竞争力。全用户逐条解码非常慢；先用小 sample 检查。若要正式参赛版本，应先固定超参、重新设计全训练期目录和训练/选模流程；本命令不会偷偷把验证/测试标签用于训练。不要用包含目标时间窗口的 transactions 做离线评测历史。

## 2. 阿里天池淘宝行为数据

官方入口：https://tianchi.aliyun.com/dataset/649

UserBehavior.csv 是无表头五列：user_id,item_id,category_id,behavior_type,timestamp。

```bash
python -m genrec.cli convert --kind taobao --input data/raw/UserBehavior.csv \
  --output data/taobao.csv --fraction 0.001 --event pv
python -m genrec.cli prepare --input data/taobao.csv --output data/taobao \
  --train-end 2017-12-01 --val-end 2017-12-02 --max-history 30
python -m genrec.cli train --data data/taobao --run runs/taobao \
  --config configs/hm.json --device cuda
python -m genrec.cli evaluate --data data/taobao --run runs/taobao --device cuda
```

可选择 `buy`、`cart`、`fav`，但这是更换任务目标，不是多任务学习。购买样本更稀疏，需要检查 `manifest.json`，不能直接比较点击任务指标。此数据是天池数据集，不把它误称为腾讯比赛数据或与某场天池比赛完全相同的官方协议。

## 3. 腾讯 2025：参考路线与扩展边界

官方仓库：https://github.com/TencentAdvertisingAlgorithmCompetition/baseline_2025
官方数据：https://huggingface.co/datasets/TAAC2025/TencentGR-1M

本项目借鉴其多模态生成推荐设定。当前没有腾讯原始 parquet adapter，没有复制官方曝光、点击、转化标签协议或加权评分。不要直接把所有曝光都变成正例，也不要把预切分的官方训练/测试拼接后重切分。

若将自己的腾讯研究子集接入，必须先按下载版本 schema 提取合法正反馈、原始用户/商品 ID、真实时间；输出下面的 canonical CSV，并明确这是自定义时间窗口实验。官方协议复现则应保留官方 split 与候选集合，扩展数据层，另行验收。多模态向量可通过下一节现成工具接入。

## 4. 通用数据契约

```csv
user_id,item_id,timestamp,category
u1,00001,1700000000,book
u1,00002,1700000100,book
```

ID 始终字符串，保留前导零；timestamp 为秒数或 ISO 日期时间。缺失/非有限时间拒绝。相同时间不假设先后。模型内部 item index 从 0 开始，不等于 raw ID。

`prepare` 输出：train/val/test.jsonl、items.json、features.npy、manifest.json。每个验证/测试用户只有一个时间窗口样本，target 是窗口内去重商品集合。冷商品以 -1 记录，每个冷商品都保留一个分母计数。

## 5. 文本/图像/其他预计算向量

每种模态一个安全 NPZ：`item_ids` 为字符串 `[N]`，`embeddings` 为浮点 `[N,D]`，不允许 object/pickle。仅提供训练时已知的商品属性。

```python
import numpy as np
np.savez('text.npz', item_ids=np.asarray(raw_item_ids, dtype=str), embeddings=text_vectors)
np.savez('image.npz', item_ids=np.asarray(raw_item_ids, dtype=str), embeddings=image_vectors)
```

```bash
python scripts/import_embeddings.py --data data/hm --npz text.npz image.npz
```

工具严格按 items.json 重排、检查缺失/重复/NaN、每模态 L2 归一化、拼接后再次归一化，并记录源文件哈希。**必须在训练前执行**；之后更换向量会改变数据指纹，旧 checkpoint 不能续训。编码器选择及图文特征提取不在当前实现中。
