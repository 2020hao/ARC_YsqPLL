# SOGI参考＋TTFS脉冲主干

模型ID：`sogi_ttfs_snn_reference`。输入参考固定为`sogi`，使用8通道分解和TTFS编码；网络、神经元、加载器与训练组件均独立保存在本目录。

当前462条队列尚未验证，历史自采数据及指标不迁入当前评价。网络源类为PhaseEventSNN，配置与来源见`model.yaml`、`provenance.json`；历史配置仅作参数来源，当前数据由显式协议选择。

本目录支持`create_model()`、本地`data_loader`、`training.train_batch`和`inference.load_checkpoint`。
