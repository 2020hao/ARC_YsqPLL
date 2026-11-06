# 自采462条：1 MHz与冻结基准比较

协议：`configs/modules/self462_1m_baselines_20261105.yaml`；执行：`scripts/run_experiment_suite.py --suite configs/modules/self462_1m_baselines_20261105.yaml --execute`。

本轮固定462条人工复核的NET9770_114 CH0录波，保留电水壶、吹风机、电磁炉、电视四类目标负载；156个“目标＋背景”条件分四折，条件的全部重复采集同折。所有方法共享125,051个窗口（正常61,896、电弧63,155）。每组seed0、四折，共20次拟合；不将历史数据集性能抄入该队列。

1 MHz输入由真实2 MHz电流做因果抗混叠下采样得到，非另次硬件采集。103抽头Kaiser FIR通带400 kHz、阻带500 kHz，保持10–100/100–400 kHz高频活动。模型内部表征采样率另列，不把相位箱数当作ADC采样率。

| 方法 | 输入与上下文 | 训练预算 | 复现边界 |
| --- | --- | --- | --- |
| CNN–SNN 1 MHz | 9周波×5通道×64相位箱 | 50 epochs / batch512 | 当前自有直接分类模型，7097参数；无蒸馏、校正分支 |
| CNN–SNN 2 MHz | 同时刻、同结构及原2 MHz表征 | 同上 | 匹配重训采样率对照 |
| ArcFENet冻结重建 | 单20 ms，250 kHz→VMD/MBWVD RGB100×100 | 20 epochs / batch128 / FP32 | 本地论文数值重建；500步上限，不暗改为2000步修复版 |
| HYDRA冻结FD配置 | 单20 ms，整流→50 kHz、1000点 | 原随机FD＋λ=.001 L1-hinge | 原凸目标HiGHS解；不是作者MATLAB源码逐位复现 |
| Fusion-current冻结适配 | 单20 ms，12.5 kHz、250点及两阶差分＋RF选32 FFT项 | 70 epochs / batch64 / BF16训练 | 仅电流架构适配；不能称作者完整电压/差分方法 |

三个基准各自独立的模型文件、数据加载器、配置保持封存版本，研究增强入口均关闭。仅新建数据集处理与执行器，训练折内拟合标准化、RF和SparseScaler；保持冻结训练预算和模型参数。输入驻留GPU和通过原实现数值核对的CUDA图仅改变执行方式。神经网络使用物理GPU2、3，CPU执行RF和HYDRA凸优化；不使用GPU0，不触碰其他用户任务。

主指标为条件等权周波BAcc，其余报告汇总Acc、Precision、Recall、FPR、F1、AP、ROC-AUC、MCC和混淆计数、逐负载结果。概率阈值.5、HYDRA margin阈值0，均不依据测试集调参。四项主要比较按条件配对、bootstrap与符号检验、Holm校正。

我们的九周波与基准单周波上下文不同，此比较评估完整系统的实用表现，不能作为仅结构、仅SNN或仅PLL的贡献消融。人工边界缓冲和完整历史要求保留稳定区间，不报告起弧延迟。此数据已有前期开发历史，单种子分组交叉结果不替代独立外部确认。

本地结果目录`outputs/self462_1m_baselines_20261105`。完成后以`FINAL_CHECK.json`、`INPUT_AUDIT.json`、`REPLAY_AUDIT.json`、`TRAINING_COMPLETE.json`为准；完整报告`1MHz模型与原基准_完整验证报告.md`。数据、标签、权重、逐窗预测保持本地，不上传公共GitHub。

## HYDRA求解执行覆盖

f0/f1默认900秒HiGHS未获得最优解，原日志保留，不把未完成解计入指标。独立配置`configs/modules/self462_1m_hydra_solver_20261105.yaml`仅将求解执行切换为HiGHS-IPM、时限1800秒及更严格数值容差；全部特征/训练样本/标准化/λ/凸目标/模型源文件/分类阈值不变。每折必须检查最优状态、原始目标重放、原始—对偶间隙和约束残差；成功后才纳入主表，不宣称默认求解过程逐位重现。

原始形式IPM仍较慢，保留两次中止记录后，使用`configs/modules/self462_1m_hydra_dual_20261105.yaml`中的数学等价对偶求解。原目标

$$\min_{\boldsymbol w,b}\;\lambda\lVert\boldsymbol w\rVert_1+\frac{1}{n}\sum_{i=1}^{n}\max\{0,1-t_i(\boldsymbol w^\mathsf T\boldsymbol z_i+b)\}$$

的对偶为

$$\max_{\boldsymbol\alpha}\;\sum_i\alpha_i,\qquad 0\le\alpha_i\le\frac1n,\quad\sum_i\alpha_it_i=0,\quad\left|\sum_i\alpha_it_i z_{ij}\right|\le\lambda\quad\forall j.$$

其中，$n$为训练窗口数，$\boldsymbol z_i$为原SparseScaler后的1024维特征，$t_i\in\{-1,1\}$为原类别，$\boldsymbol w,b$为同一线性分类器，$\boldsymbol\alpha$为优化对偶变量。约束对偶乘子恢复$\boldsymbol w,b$后，检查全部原始训练约束、原始目标、对偶目标及间隙。两组小规模数值测试原始/对偶目标差小于$10^{-15}$；全数据必须逐折重新取得最优性证书。此处改变求解的数学表示和执行效率，不改变分类器、损失或正则化，不宣称求解路径逐位相同。基准封存源文件和模型YAML保持不变。

全规模对偶内点法仍返回求解错误，保留两折日志；最终有效结果采用`self462_1m_hydra_dual_ds_20261105.yaml`的HiGHS双单纯形法。首个有效折的原始—对偶差约$1.05\times10^{-14}$，原始约束残差约$1.31\times10^{-14}$。为缩短时间，`self462_1m_hydra_parallel_20261105.yaml`在内存充足时让四折CPU并行，正在进行的前两折不重启；旧双折队列仅在首折收据完整后终止，避免重复拟合后两折。

另以`self462_1m_hydra_scaled_20261105.yaml`对优化变量做$\beta_i=n\alpha_i$的等价数值缩放诊断，分类器、特征、样本、损失及正则化均不变；该内点求解诊断返回错误，未提供可用结果，日志保留。方法选择以最优性证书为准，不能依据测试指标选择求解器。最终复现报告需列明各折实际求解算法和全部数值执行覆盖；“冻结模型不变”不等于求解执行过程逐位相同。

最后一折全量对偶DS在1800秒内仍未最优，保留日志后由`self462_1m_hydra_active_20261105.yaml`执行等价工作集LP。训练内同目标SGD只提供数值warm-start，不是最终分类器；自由变量以外的样本并未删除，固定对偶值继续进入约束右端和总目标。每轮释放违反KKT的固定变量，直到**全部训练数据**原始/对偶目标差小于$10^{-6}$且所有原始、对偶约束通过。3000×16数值测试目标与完整原始LP差约$1.55\times10^{-15}$。仅按数值最优性选择求解流程，不看测试分数；1024原特征、全部训练窗口、Scaler、正则化和阈值不变。

相邻工作集之间另以`self462_1m_hydra_native_20261105.yaml`使用原生HiGHS、数学等价双边约束和单纯形基检查点，避免仅新增少量变量后从头求解。训练目标、全样本证书和“先获得合格最优解、不得按测试分数选择求解器”规则保持不变。接口依赖实际SciPy内置HiGHS版本，环境版本与附加源快照在验证后记录。

完成后的`SELF462_1M_BASELINE_REGISTRY.json`及`baseline_metrics_frozen.csv`固定本队列三基准：包含数据/划分/OOF/原代码封存哈希、每折权重和预处理收据、指标及复现资格。未来只有数据、窗口队列、分折、输入表示、模型和训练协议逐项匹配时才可复用；队列或前端改变需要另建版本，不覆盖本次参照，也不覆盖旧数据集指标。

## 已完成结果摘要

1 MHz CNN–SNN BAcc 96.94% / Acc 97.12%；2 MHz匹配对照 BAcc 96.60% / Acc 96.77%；ArcFENet冻结重建 BAcc 85.93% / Acc 86.31%；HYDRA冻结FD BAcc 95.37% / Acc 95.41%；Fusion-current冻结适配 BAcc 94.06% / Acc 94.07%。

全部20个最终模型及核验完成。最大原始—对偶目标差1.05e-14，最大原始约束残差2.86e-14。完整指标、混淆计数、逐负载、配对区间与复现边界见本地完整报告及最终收据。
