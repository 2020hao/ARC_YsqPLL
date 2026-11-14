from pathlib import Path
import sys,subprocess,json
import pandas as pd,numpy as np
O=Path(__file__).resolve().parents[2]/'outputs/current_correction_20261109';import heads
heads.summary();subprocess.run([sys.executable,str(Path(__file__).with_name('audit.py'))],check=True)
df=pd.read_csv(O/'完整性能指标.csv');delta=pd.read_csv(O/'配对增益与错误转移.csv');groups=pd.read_csv(O/'逐工况指标.csv')
baseparams={'ArcFENet':98917,'HYDRA':1025,'Fusion-current':918258}
name={'baseline':'原基准模型','calibration_only':'仅分数校准','RCR_dynamic_no_gate':'RCR动态校正（无门控）','RCR_dynamic_soft_gate':'RCR动态校正（软门控）'}
view=df.copy();view['模型']=view.model;view['接入方式']=view.arm.map(name);view['数据集']=view.dataset.map({'self462':'完整自采462条','zenodo5501527':'公共1357窗口'});view['总可训练参数']=view.apply(lambda r:baseparams[r.model]+(0 if r.arm=='baseline' else 2 if r.arm=='calibration_only' else 38),axis=1)
for k in ['BAcc','Acc','Precision','Recall','FPR','FNR','F1','AP','ROC_AUC']:view[k]=view[k]*100
with pd.ExcelWriter(O/'两个数据集_三基准校正完整结果.xlsx') as w:
 view.to_excel(w,sheet_name='完整指标',index=False);delta.to_excel(w,sheet_name='配对增益和错误转移',index=False);groups.to_excel(w,sheet_name='逐工况',index=False)
with (O/'结果与分析.md').open('a') as out:
 out.write('\n\n## 结果解释\n\n')
 for ds in df.dataset.unique():
  sub=df[df.dataset==ds];out.write(f'### {ds}\n\n')
  for model in baseparams:
   x=sub[sub.model==model].set_index('arm')
   if len(x)==0:continue
   b=x.loc['baseline'];g=x.loc['RCR_dynamic_soft_gate'];n=x.loc['RCR_dynamic_no_gate'];c=x.loc['calibration_only'];out.write(f'- **{model}**：软门控相对基准Acc变化{100*(g.Acc-b.Acc):+.2f} pp，Recall变化{100*(g.Recall-b.Recall):+.2f} pp，FPR变化{100*(g.FPR-b.FPR):+.2f} pp。仅校准BAcc变化{100*(c.BAcc-b.BAcc):+.2f} pp；软门控相对无门控BAcc变化{100*(g.BAcc-n.BAcc):+.2f} pp。\n')
 out.write('\n本次固定融合强度为1，未搜索更优超参数；表格保留无门控与软门控全部结果，不按测试集逐模型挑选融合形式。动态头36个系数、正仿射校准2个参数，共新增38个可训练参数；PCA、均值和标准差另占固定存储，不计入可训练参数。端到端计算量还需加上1MHz滤波、RCR及特征计算，本次不以这38个参数代表全流程开销。\n\n自采复用的是1MHz数字适配基准，校正特征也来自同一批462条录波的1MHz分支；不能将其当作额外采集的新硬件数据。原始2MHz数据和已有2MHz自有模型结果未改动。\n')
print('FINALIZED',len(df))
