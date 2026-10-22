# 算法与评估设计

## 语义量化

残差 K-Means：从 r₀=x 开始，cₗ=argmin_j ||rₗ−eₗⱼ||²，rₗ₊₁=rₗ−eₗ,cₗ。

RQ-VAE：MLP 编码 z=E(x)，对 z 逐层量化，解码器重建输入特征。损失是重建 MSE 加每层 codebook loss 和 0.25×commitment loss。重建梯度用 straight-through estimator，codebook 用显式量化损失更新；没有 EMA 或平衡分配正则。先以初始 encoder 的表示做 K-Means 初始化码本。

元数据相同会有相同 SID，这是合理现象但会阻止唯一检索。实现用训练目录中稳定排序的商品序号作为桶内 rank，转换为固定宽度 radix-K 后缀。每一层的 token 区间独立。后缀解决唯一性，但后缀不含语义，也不能恢复量化丢失的信息。

诊断：1−不同语义码数/商品数（重复占比）、最大碰撞桶、各层码本使用率、分布困惑度 exp(H)。唯一后缀之后碰撞为零不表示码本质量好。

## 生成目标

输入 `[BOS, SID(history_item_1), ..., SID(history_item_n), SEP]`。
预测目标 SID 的所有 token（含消歧后缀），固定长度无需 EOS。

L = −Σ_l log p(s_l | history, s_<l)。

Decoder-only Transformer 使用因果 mask、位置 embedding、padding mask，输入/输出 embedding tied。历史 token 对 loss 屏蔽，避免历史重建掩盖真正的下一商品目标。无 LLM 预训练依赖。

## 约束解码

Trie 按训练目录全部合法 SID 构建。每步对所有 beam 并行一次模型前向；累积原始 log-softmax 概率，只扩展合法子节点。最终固定长度叶子映射为 item index。

**先对完整词表归一化，后限制分支**。若只对有效分支重新归一化，不同前缀的分数含义会改变。
Beam search 是近似搜索，约束只保证合法性，不保证全局 Top-K 最优；足够大的 beam 才接近枚举。没有 KV cache，每层重复计算历史；词表小不意味着当前代码一定比检索快。

## 公平对比

所有模型使用相同训练窗口、训练目录、目标窗口、历史截断、允许复购、无 sampled negatives。Item-ID 序列基线使用相同 Transformer 实现，但一个商品对应一个 token；它不是原版 SASRec 的 BCE/负采样实现。参数量、token 长度不同，实验中应另外报告参数量与训练时间，不能声称严格等算力。

Popularity 按训练样本目标计频，ItemKNN 按训练样本中每用户观测商品集计算 co-occurrence cosine。极少交互用户可能没有训练样本，因此不进入这些统计，这一点应在数据清洗与报告中注明。

## 时间与泄漏

- `t < train_end` 的交互决定目录及训练样本。
- 验证以 train_end 为观察时刻，测试以 val_end 为观察时刻。
- 同一 timestamp 的目标不能看到彼此，尤其 H&M 仅提供日期。
- tokenizer 仅在训练目录的静态特征拟合。
- checkpoint 按验证 NDCG 选择；测试指标不参与训练。
- 冷商品不可召回，仍进入分母。冷用户输入 BOS/SEP，输出模型学到的无历史分布。
- 无负反馈/曝光 propensity 校正，因此不能把离线 MAP 等价为 CTR/转化率提升。

## 指标

R_u 为真实目标集合，P_u 为 Top-K 推荐列表。

- Recall@K = |R∩P| / |R|。
- Hit@K = 1[|R∩P|>0]。多目标窗口下 Hit 与 Recall 不相等。
- DCG@K = Σ hit_i/log₂(i+1)，NDCG 除以理想 DCG。
- AP@K = Σ precision@i × hit_i / min(|R|,K)，再对用户平均为 MAP。
- 重复预测只第一次命中有效；冷目标保留分母。
- coverage 是预测过的不同商品数/训练目录大小。

延迟包含 Python trie 搜索与模型前向，单用户顺序执行，GPU 前后同步；不是服务 QPS，没有批量 serving，未排除首条 warm-up，需报告硬件与这些条件。

## 工程限制

训练样本驻内存；特征量化计算完整 `[N,K,D]` 距离；ItemKNN 构建 O(Σ|I_u|²)，保存 top neighbors 仅限制推理量。真实大数据应先用户采样，再引入 Arrow 流式训练、分块距离/FAISS、稀疏乘法、KV cache、批量 beam、AMP 与 DDP。当前单 GPU 代码可运行，但尚无 GPU 实测或 H200 性能结论。
