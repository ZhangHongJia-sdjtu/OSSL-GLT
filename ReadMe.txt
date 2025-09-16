1. 项目简介
------------
在线半监督图卷积网络模型（OSSL-GLT），结合 GCN、LSTM、Transformer，用于图结构时序数据分类，支持半监督学习和在线更新。


2. 数据格式
- Excel 文件
- 标签为整数（0 或 1）
- 特征需 reshape 为图节点特征格式（代码中 #图数据格式#）


3. 使用说明
1. 安装依赖:requirements.txt文件
2. 设置数据路径 `file_path`
3. 调整图数据 reshape 与模型维度参数
4. 运行脚本
