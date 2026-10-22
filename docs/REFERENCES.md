# 一手参考资料

调研日期：2026-10-22。借鉴任务与算法思想，代码独立实现。

1. 腾讯官方赛事介绍（2025）：https://www.tencent.com/zh-cn/tencent-hosts-chinas-first-all-modal-generative-recommendation-competition-attracting-more-than-6000-students-worldwide/
2. 赛事官方论文（2026）：https://arxiv.org/abs/2604.04976
3. 腾讯官方 baseline：https://github.com/TencentAdvertisingAlgorithmCompetition/baseline_2025
4. TencentGR-1M：https://huggingface.co/datasets/TAAC2025/TencentGR-1M
5. Kaggle H&M：https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations
6. H&M 数据页：https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data
7. 天池淘宝行为数据：https://tianchi.aliyun.com/dataset/649
8. TIGER / Recommender Systems with Generative Retrieval：https://arxiv.org/abs/2305.05065
9. PyTorch Transformer API：https://docs.pytorch.org/docs/stable/generated/torch.nn.Transformer.html

腾讯官方 baseline 的 README 以 SASRec 序列建模为起点，并提供 RQ-VAE 扩展。比赛的生成式推荐主题不意味着其所有 baseline 都是 SID token 逐个生成，不能把名称当作实现证据。本项目选择明确的离散 SID 自回归路线。

H&M 是多目标购买列表任务；淘宝数据可构造点击或购买预测；腾讯有曝光/点击/转化与不同阶段加权规则。不同数据上的 MAP/Recall 不能直接横向排名。
