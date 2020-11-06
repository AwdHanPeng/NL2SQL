Nov05_16-47-36_iZm5e7nybdevafi694evqwZ column特征重新提取的，效果正常，跑的很慢
runs/Nov05_19-31-51_iZm5e7nybdevafi694evqwZ 改进db_unit, 不加signal，小样本，正常
runs/Nov05_19-41-43_iZm5e7nybdevafi694evqwZ 加了signal，小样本也很正常，拟合速度相差无几，爽到
runs/Nov05_20-14-17_iZm5e7nybdevafi694evqwZ 全部样本 IndexError: index 422 is out of bounds for dimension 1 with size 422
改 model中230和240中的sum为mean，attention可能有问题，看一眼

runs/Nov05_21-05-20_iZm5e7nybdevafi694evqwZ: 200样本，学得很慢，改进特征sum为mean
runs/Nov05_21-24-49_iZm5e7nybdevafi694evqwZ：改进特征sum为mean，fix bert, 学习率1e-3，不动
runs/Nov05_21-32-34_iZm5e7nybdevafi694evqwZ：1e-3不行，还是1e-4，db也无法学习
bert不fix，data1000个

改acc为所用样本平均而不是所有token平均