from utils import get_coordinate
import glob
import numpy as np
import pandas as pd
import os
from shutil import copyfile

#搜索目录下所有的高斯scan结果log文件，得到lammps输入坐标文件和对应的能量(Hartree)
#Convert all Gaussian scan result log files under directory to LAMMPS input coordinate files and get corresponding structural energy(Hartree)
def Data_Processing():
    loges = glob.glob('./data/dataset_dft/*/*.LOG')
    for log in loges:
        log = log.replace('\\','/')
        print('Collecting data in %s' %log)
        dir = log.split('/')[-2]
        file_name = log.split('/')[-1].split('.')[0]
        output_path = './data/data_for_lammps/%s/%s/' % (dir,file_name)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        get_coordinate(log,output_path)

    # 将lammps坐标文件放入运行lammps的目录里，并将能量转化为相对能量(Kcal/mol)
    # Put the LAMMPS coordinate file into the directory where LAMMPS is running and convert energy to relative energy (Kcal/mol)
    corr = glob.glob('./data/data_for_lammps/*/*/')

    total_struture = []
    struture_txet = './data/data_for_lammps/struture.txt'
    if os.path.exists(struture_txet):
        os.remove(struture_txet)

    for dir in corr:

        with open(struture_txet,'a',newline='') as f:
            f.writelines([str(dir.split('\\')[-2])+'\n'])
        struture_file = glob.glob(dir+'\data_*')
        struture_file = sorted(struture_file, key=lambda x: int(x.split("\\")[3].split('_')[1]))
        for struture in struture_file:
            total_struture.append(struture)
        # print(total_struture)
        energy_file = dir + 'Energy.txt'
        df = pd.read_csv(energy_file, sep=' ', header=None)
        min_vals = np.min(df.iloc[:, 1])
        df.iloc[:, 1] = (df.iloc[:, 1] - min_vals) * 627.5094  # 1 Hartree = 627.5094 kcal/mol
        if not os.path.exists('./data/data_for_GAopt/'):
            os.makedirs('./data/data_for_GAopt/')
        df.to_csv('./data/data_for_GAopt/' + str(dir.split('\\')[-2]) + '_Energy.csv',header=['Scan_Step','DFT'], index=False)
        # print(dir.split('\\')[-2])
    flag = 0
    for i in range(1,21):
        run_path = './lammps_running/run%s/'%(i)
        if not os.path.exists(run_path):
            os.makedirs(run_path)
        flag = 0
        for st in total_struture:
            flag += 1
            data_path=run_path+'data_'+str(flag)
            copyfile(st,data_path)


    #   打开shell脚本模板修改参数并分配到每个run目录
        with open('./data/template/auto_lammps.sh', 'r') as file:
            # 读取文件内容
            content = file.read()

        content = content.replace('PLACEHOLDER', str(i))

        # 将替换后的内容写回到文件中
        with open('./lammps_running/run%s/auto_lammps.sh'%(i),'w') as file:
            file.write(content)

    #   打开input模板修改参数并分配到每个run目录
        with open('./data/template/input.in', 'r') as file:
            # 读取文件内容
            content = file.read()

        content = content.replace('PLACEHOLDER', str(i))
        content = content.replace('LOOPEND', str(len(total_struture)))

        # 将替换后的内容写回到文件中
        with open('./lammps_running/run%s/input.in'%(i),'w') as file:
            file.write(content)

        copyfile('./data/template/ffield.reax.cho','./lammps_running/run%s/ffield.reax.cho'%(i))

    return len(total_struture)

if __name__ == '__main__':
    Data_Processing()