# ArcFENet与Fusion current基线资格验证

本轮补齐两项本地实现的能力核验与版本化数值敏感性实验。原冻结基线保持不变；
同源窗口识别、分组泛化和训练内校准分开报告。结果文件保留在服务器`outputs/`，
公开仓库提供可检查的代码、协议及测试，不附带录波、标签、权重或逐窗预测。

## 实验范围

| 内容 | 训练数量与预算 | 用途 |
|---|---|---|
| ArcFENet旧前端同源能力 | 双数据集×3种子，共6次，20轮 | 核验冻结结构的基本识别能力 |
| ArcFENet数值修订 | 同源能力6次＋分组五折三种子30次，均20轮 | 检查数值前端修订的影响 |
| Fusion-current同源能力 | 双数据集×3种子，共6次，70轮 | 核验当前架构的基本识别能力 |
| Fusion-current内层OOF | 双数据集×5外折×2内折，共20次，70轮 | 为seed0的普通校准生成诚实训练分数 |
| Fusion-current校准 | 10份正斜率两参数仿射模型，训练内拟合 | 区分决策偏置与表征问题 |

共68次新神经网络拟合。模型结构、原有损失和训练预算未加入RCR/C0/额外动态特征。
所有模型只用物理GPU1/2/3；GPU0保留。新拟合的完整预测已经由保存权重独立重放，
概率误差及判决变化均为0。CPU合同测试覆盖滤波规格、校准及冻结保护。

同源能力试验可支持当前网络具备已见录波识别能力，不代替未见条件结论。
数值修订提高了VMD收敛比例，却没有产生双数据集一致的分类提升；普通校准也没有
稳定解决公共数据的跨电器误报。因此这两个候选不自动覆盖原冻结比较标准。

## ArcFENet修订范围

从151份原始连续录波重建，逐份复核完整原文件哈希，使用NET9770_114 CH0原生
2 MHz和公共C3原生1 MHz电流。没有重新猜标签或加入旧数据。

数值修订保持250 kHz、5,000点、三模态及50/22k/101kHz初始中心、MBWVD/HFDC/DDC
和20轮Adam训练。因果FIR仍为401/801 taps、Kaiser beta7.5、0.2ms延迟，将数字截止
改为118.5kHz，使<=112kHz通带误差低于0.1dB、>=125kHz阻带抑制超过60dB；VMD上限
固定2,000次，alpha2000、tau0、tol1e-7不变。设置由原文频带和既有训练内数值证据确定。

FP64 CUDA Graph执行与CPU相同公式在500/2,000步训练样本上对照，保留完成标记与
未收敛状态。程序没有通过增大tol、锁死中心或删除困难窗口来提高收敛比例。
归一化/RGB等作者未披露实现选择仍需说明，不将本地重建称为作者完整代码复现。

第一次能力执行的全张量有限性检查触及人为显存上限，改为分块CPU检查，未改变模型
或数据。另修正了新执行器的计时变量覆盖；受影响的旧能力耗时不用于成本结论。
原日志、修订记录和准确的已执行源码快照均保存，分类预测不受这两项执行修正影响。

## Fusion current范围

保留12.5kHz/250点电流与一二阶导数、训练内RF32和原时频CNN/Transformer/交叉注意力。
正斜率校准只使用每个外训练集合中两内折的组隔离预测，固定损失、正则和概率阈值，
不读取外测试标签。校准结果为seed0控制，不冒充三种子模型整体提升。

当前仍缺原方法的同步供电电压、健康差分及合成目标故障适配，名称保持Fusion-current。
使用普通电流PLL做健康差分是另一项可研究的适配，需明确正常初始化/目标正常访问，
本轮没有将其混入零目标适配分组主表。

## 入口与必要数据

本轮使用`configs/models.yaml`中的冻结网络参数，两个研究YAML只声明实验范围。
需要已有审核元数据、索引与两个数据集的当前输入准备产物；不能仅提供任意原始文件
而省略标签与划分。参见[数据划分协议](DATA_SPLIT_PROTOCOL_20261030.md)。

```bash
python scripts/run_experiment_suite.py --suite configs/modules/fenet_qualification_20261030.yaml
python scripts/run_experiment_suite.py --suite configs/modules/fusion_qualification_20261030.yaml
```

新建研究输出目录后，FENet按`--prepare`、准备脚本`--freeze`、CPU的`--native 0/1`、
GPU的`--smoke`、流式准备的`--worker`、CPU的`--assemble`、分类器的
`--worker old_capability/repaired`和汇总脚本顺序执行。CPU命令设置`CUDA_VISIBLE_DEVICES=''`；
GPU命令只暴露1或2。多个worker通过文件锁分配任务，不在同一GPU重复启动。
已冻结输出目录禁止重建；新实验应版本化输出路径。

Fusion脚本提供`--prepare`、`--worker`、`--calibrate`、`--summarize`；worker和校准只暴露
1/2/3中的一个物理GPU。三GPU共享claim队列，数据集独立，全部内折完成后再校准。

## 本地结果位置

- `outputs/baseline_qualification_summary_20261030/`：总表、划分计数和汇总报告。
- `outputs/fenet_qualification_20261030/`：42份模型、数值参数、原始文件身份及预测重放。
- `outputs/fusion_qualification_20261030/`：26份模型、10份校准、诚实OOF与预测重放。
- `outputs/grouped_baseline_standard_20261029/`：原冻结比较标准，保持不变。
