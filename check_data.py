import numpy as np
import os

data_dir = "condition+monitoring+of+hydraulic+systems"  
profile = np.loadtxt(os.path.join(data_dir, "profile.txt"))

print("profile 形状:", profile.shape)
print("前 5 行:\n", profile[:5])
# 确定txt形状 profile中应该有5列 分别为四部件状态和稳定性标志 行数为2205个周期

for f in ["PS1.txt", "FS1.txt", "TS1.txt", "EPS1.txt"]:
    x = np.loadtxt(os.path.join(data_dir, f))
    print(f, "形状:", x.shape, "第一行前5个值:", x[0, :5])
# 对于压力部件PSx和功率部件EPS1列数为100*60=6000
# 对于流量部件FSx列数为60*10=600
# 对于温度部件TSx和振动部件VS1列数为60*1=60