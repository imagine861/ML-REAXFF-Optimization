import os
from utils import ffield
import time

ff = ffield()
def Create(mode: object, N_dataset: object, parameter: object = None, batch_size: object = 1) -> object:
    para = []
    if mode  == 'RF':

        for m in range(1,4):
            for p in ['ro(sigma)','val','rvdw','Dij','gamma','ro(pi)','val(e)','alfa','gamma(w)','Val(angle)','p(ovun5)',
                      'chiEEM','etaEEM','ro(pipi)','p(lp2)','Heat increment','p(boc4)','p(boc3)','p(boc5)','p(ovun2)','p(val3)','Val(boc)','p(val5)']:
                para.append([p,m])
        for s in range(1,7):
            for p in ['De(sigma)','De(pi)','De(pipi)','p(be1)','p(bo5)','13corr','p(bo6)','p(ovun1)','p(be2)','p(bo3)','p(bo4)','p(bo1)','p(bo2)']:
                para.append([p,s])
        for j in range(1,4):
            for p in ['dDij','dRvdW','alpha','Ro(dsigma)','Ro(dpi)','Ro(dpipi)']:
                para.append([p,j])
        for k in range(1,19):
            for p in ['Thetao,o','p(val1)','p(val2)','p(coa1)','p(val7)','p(pen1)','p(val4)']:
                para.append([p,k])
        for l in range(1,27):
            for p in ['V1','V2','V3','p(tor1)','p(cot1)']:
                para.append([p,l])

        save_path = './data/random_forest_data/input/'
    #自定义方法来进行力场参数改写，需要提供一个列表其中包括一个含有参数名和原子种类的列表
    if mode == 'Custom':
        para = parameter
        save_path = './data/ml_dataset/input/1'

    for i in range(1,N_dataset+1):
        csv_path = save_path + str(20*(batch_size)+i)+'.csv'
        ffiled_path = './lammps_running/Parameter/'+str(i)+'/'
        if not os.path.exists(ffiled_path):
            os.makedirs(ffiled_path)
        data0 = []

        for i in para:
            data0.append(ff.Change(i[0], i[1]))
            with open('./data/data_for_GAopt/raw.txt', 'a', newline='') as f:
                f.writelines([str(ff.Change(i[0], i[1])[2]) + '\n'])
                f.close()
        ts = ff.Create_trainset(data0, csv_path)
        ff.Create_reaxff(ts, data0, ffiled_path)
    return N_dataset




