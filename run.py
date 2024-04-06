import time
import pandas as pd
from utils import extarct_data
from DFT_data_processing import Data_Processing
from REAXFF_para_gen import Create
from RF_importance import rf
from NN_train import train
import glob
import os

init_time = time.time()
print('Are you ready? (¯▿¯) I’m about to start optimizing reaxff parameters. ε=ε=┌( >_<)┘')
#
#
amount = Data_Processing()
print(r'here we go ヽ(｡ゝω・｡)ﾉ, Let’s explore the importance of each force field parameter together, like adventurers!')
st = time.time()
# 调整n_dataset_for_rf可以改变随机森林训练数据集大小，单位是10000
n_dataset_for_rf = 4
Create('RF',n_dataset_for_rf)
#调用windows的终端来通过gitbash来运行shell脚本，需要提取配置好gitbash。
os.chdir('./lammps_running')
os.system(r'"C:\Program Files\Git\bin\bash.exe" ./parallel_computing_rf.sh')
print('end')

# #linux的终端的运行脚本方法
# os.chdir('./lammps_running')
# print('start')
# os.system('./parallel_computing_rf.sh')
# print('end')

print('computing MD data cost time: %s h' % ((time.time() - st) / 3600))


开始收集lammps的输出结果
for i in range(1,n_dataset_for_rf+1):
    path = './lammps_running/run%s/info%s.data' %(i,i)
    if not os.path.exists('./data/random_forest_data/output'):
        os.makedirs('./data/random_forest_data/output')
    raw_path = './data/random_forest_data/output/Energy%s.csv' %(i)
    extarct_data(amount,path,raw_path)

# 调用随机森林特征重要性模块进行力场参数重要性排序
X = pd.concat([pd.read_csv(i) for i in sorted(glob.glob('./data/random_forest_data/input/*.csv'),key= lambda x:int(x.split('\\')[1].split('.')[0]))])
y = pd.concat([pd.read_csv(i,header=None) for i in sorted(glob.glob('./data/random_forest_data/output/*.csv'),key= lambda x:int(x.split('\\')[1].split('.')[0]))])
print(X.shape)
print(y.shape)#
rf(X,y)
print('It took us %sh to finally find the importance of force field parameters.' % ((time.time() - st) / 3600))

#通过随机森林结果取前50进行力场参数组合建立
st2 = time.time()
#
data = pd.read_csv('./data/random_forest_data/result/all_rank.csv')
data = data['feature'][:50]
param = []
for i in data:
    type,name = i.split('_')
    param.append([str(name),int(type)])

batch_size = 2
for batch in range(batch_size):
    Create('Custom',20,parameter=param,batch_size=batch)
    #调用windows的终端来通过gitbash来运行shell脚本，需要提取配置好gitbash。
    os.chdir('./lammps_running')
    os.system(r'"C:\Program Files\Git\bin\bash.exe" ./parallel_computing.sh')
    # # #linux的终端的运行脚本方法
    # os.chdir('./lammps_running')
    # print('start')
    # os.system('./parallel_computing_rf.sh')
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
    os.chdir(parent_dir)
    for i in range(1,21):
        path = './lammps_running/run%s/info%s.data'%(i,i)
        if not os.path.exists('./data/ml_dataset/output'):
            os.makedirs('./data/ml_dataset/output')
        raw_path = './data/ml_dataset/output/Energy%s.csv'%(batch*20+i)
        extarct_data(amount,path,raw_path)
print('computing cost time: %s h' % ((time.time() - st2) / 3600))
# #
#深度学习模型的训练，并保存最好的模型
print('start modeling !')
st3 = time.time()
X = pd.concat([pd.read_csv(i) for i in sorted(glob.glob('./data/ml_dataset/input/*.csv'),key= lambda x:int(x.split('\\')[1].split('.')[0]))])
y = pd.concat([pd.read_csv(i,header=None) for i in sorted(glob.glob('./data/ml_dataset/output/*.csv'),key= lambda x:int(x.split('\\')[1].split('.')[0].split('y')[1]))])
print(X.shape)
print(y.shape)
train(X,y)
print('modeling cost time: %s h' % ((time.time() - st3) / 3600))

#使用遗传算法进行参数空间探索

#
#
#
#
#
#
#
#
#
#
#
#
#
