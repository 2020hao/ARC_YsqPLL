# ARC_YsqPLL

**基于电流自同步与跨周波动态信息的交流串联电弧检测。**

> 当前自采数据来自最新人工标注导出库：**468次采集**。按此前排除微波炉、台灯目标的要求，现有实验纳入**462次**。旧138份、38,374窗口队列已停用。

完整重构代码与最新文档在[当前重构分支](https://github.com/2020hao/ARC_YsqPLL/tree/refactor/model-layout-v2-20261106)，见[PR #1](https://github.com/2020hao/ARC_YsqPLL/pull/1)。main当前保留旧代码作为兼容来源，使用当前模型请进入该分支；本首页已按最新数据更新。

## 最新数据范围

| 目标负载 | 标注并导出 | 当前实验纳入 |
|---|---:|---:|
| 吹风机 | 186 | 186 |
| 热水壶 | 174 | 174 |
| 电磁炉 | 90 | 90 |
| 电视 | 12 | 12 |
| 微波炉 | 3 | 0 |
| 台灯 | 3 | 0 |
| **合计** | **468** | **462** |

468次对应936个双卡BIN文件；它们不是936次独立采集。人工标注候选总数为504，尚未导出的记录不进入实验。

| 项目 | 当前自采实验队列 | 当前公共数据协议 |
|---|---|---|
| 录波采集 | 462次，从468次导出库按规则纳入 | 13份电器录波 |
| 工况/录波组 | 156个目标与背景工况 | 13组 |
| 原始采样率 | 2 MHz | 1 MHz |
| 检测输入 | 114设备CH0总路电流 | C3电流；C2仅作标注辅助 |
| 标签 | 人工复核起弧、熄弧区间 | 已审核正常/电弧区间 |
| 共同评价窗口 | 125,051：正常61,896、电弧63,155 | 1,357：正常794、电弧563 |
| 外层划分 | 四折工况隔离，同采集整体归组 | 五折电器/录波组隔离 |

自采电流位置：`/data/hzy/test-manager/1020重新处理/人工对齐_BIN`。标签及导出记录：`/data/hzy/test-manager/1020重新处理/人工学习复核`。

**468是最新原始导出库规模，462是现有实验纳入规模。** 已完成的125,051窗口成绩对应462次，不能直接改名为使用全部468次的结果。

## 当前代码与说明

- [完整README](https://github.com/2020hao/ARC_YsqPLL/blob/refactor/model-layout-v2-20261106/README.md)
- [当前数据与划分](https://github.com/2020hao/ARC_YsqPLL/blob/refactor/model-layout-v2-20261106/docs/DATASETS.md)
- [468/462核对说明](https://github.com/2020hao/ARC_YsqPLL/blob/refactor/model-layout-v2-20261106/docs/DATASET_SCOPE_20261106.md)
- [模型目录](https://github.com/2020hao/ARC_YsqPLL/blob/refactor/model-layout-v2-20261106/docs/MODEL_CATALOG.md)
- [快速开始](https://github.com/2020hao/ARC_YsqPLL/blob/refactor/model-layout-v2-20261106/docs/QUICKSTART.md)

三个冻结baseline保留原实现；1 MHz动态旁路与自有CNN–SNN分别登记，实验状态按各自协议报告。数据、标签明细、权重和本机配置保持本地，仅同步代码与文档。
