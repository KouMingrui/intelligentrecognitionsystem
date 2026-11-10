import numpy as np
import os

data_dir = "condition+monitoring+of+hydraulic+systems"  
profile = np.loadtxt(os.path.join(data_dir, "profile.txt"))

print("profile 形状:", profile.shape)
print("前 5 行:\n", profile[:5])

for f in ["PS1.txt", "FS1.txt", "TS1.txt", "EPS1.txt"]:
    x = np.loadtxt(os.path.join(data_dir, f))
    print(f, "形状:", x.shape, "第一行前5个值:", x[0, :5])