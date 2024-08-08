import sys
sys.path.insert(0, '/root/workspace/Long-Seq-Model')
print(sys.path)
import models
print(models.__file__)
import numpy as np 
# 导入 models.mamba_ssm.Mamba1 中的 Mamba 类
from models.mamba_ssm.Mamba2 import Mamba2

# 测试导入是否成功
mamba_instance = Mamba2(d_model=128)
print(mamba_instance)
