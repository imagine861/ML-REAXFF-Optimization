import os.path
import re
import pandas as pd
import csv
#定义Ffiled类对象，完成对力场文件的修改和参数值保存的功能。
class ffield:
    def __init__(self):
        self.generalpara = 'pboc1,pboc2,pcoa2,ptrip4,ptrip3,kc2,povun6,ptrip2,povun7,povun8,ptrip1,swa,swb,n.u.,pval7,plp1,pval9,pval10,n.u.,ppen2,ppen3,ppen4,n.u.,ptor2,ptor3,ptor4,n.u.,pcot2,pvdw1,cutoff,pcoa4,povun4,povun3,pval8,n.u.,n.u.,n.u.,n.u.,pcoa3'.split(
            ',')
        self.atom1 = 'ro(sigma);val;mass;rvdw;Dij;gamma;ro(pi);val(e)'.split(';')
        self.atom2 = 'alfa;gamma(w);Val(angle);p(ovun5);n.u.;chiEEM;etaEEM;n.u.'.split(';')
        self.atom3 = 'ro(pipi);p(lp2);Heat increment;p(boc4);p(boc3);p(boc5);n.u.;n.u.'.split(';')
        self.atom4 = 'p(ovun2);p(val3);n.u.;Val(boc);p(val5);n.u.;n.u.;n.u.'.split(';')
        self.bond1 = 'De(sigma);De(pi);De(pipi);p(be1);p(bo5);13corr;n.u.;p(bo6)'.split(';')
        self.bond2 = 'p(ovun1);p(be2);p(bo3);p(bo4);n.u.;p(bo1);p(bo2)'.split(';')
        self.diagonal = 'dDij;dRvdW;alpha;Ro(dsigma);Ro(dpi);Ro(dpipi)'.split(';')
        self.angle = 'Thetao,o;p(val1);p(val2);p(coa1);p(val7);p(pen1);p(val4)'.split(';')
        self.torsion = 'V1;V2;V3;p(tor1);p(cot1);n.u;n.u.'.split(';')
        self.hydro = 'r(hb);p(hb1);p(hb2);p(hb3)'.split(';')

    def Change(self, para, type):
        S = 10000
        percent = 50
        general = self.generalpara
        atom1 = self.atom1
        atom2 = self.atom2
        atom3 = self.atom3
        atom4 = self.atom4
        bond1 = self.bond1
        bond2 = self.bond2
        diagonal = self.diagonal
        angle = self.angle
        torsion = self.torsion
        hydro = self.hydro

        data = []
        with open("./data/template/ffield.reax.cho") as f:
            for line in f.readlines():
                data.append(line)
        import numpy as np
        def LHSample(D, bounds, N):

            result = np.empty([N, D])
            temp = np.empty([N])
            d = 1.0 / N

            for i in range(D):

                for j in range(N):
                    temp[j] = np.random.uniform(
                        low=j * d, high=(j + 1) * d, size=1)[0]

                np.random.shuffle(temp)

                for j in range(N):
                    result[j, i] = temp[j]

            b = np.array(bounds)
            lower_bounds = b[:, 0]
            upper_bounds = b[:, 1]
            if np.any(lower_bounds > upper_bounds):
                print('范围出错')
                return None

            np.add(np.multiply(result,
                               (upper_bounds - lower_bounds),
                               out=result),
                   lower_bounds,
                   out=result)

            return np.round(result, 5)

        i = type

        if para in general:
            index = general.index(para)
            num = data[2+index].split()[0]
            lx = round(float(num), 4) - round(float(num), 4) * (int(percent) / 100)
            hx = round(float(num), 4) + round(float(num), 4) * (int(percent) / 100)
            D = [lx, hx]
            D.sort()
            X = [0, 100]
            new = LHSample(2, [D, X], int(S))[:, 0]

        if para in atom1:
            index = atom1.index(para)
            row1 = data[45 + 4 * (i - 1)].split()
            num = row1[index + 1]
            lx = round(float(num), 4) - round(float(num), 4) * (int(percent) / 100)
            hx = round(float(num), 4) + round(float(num), 4) * (int(percent) / 100)
            D = [lx, hx]
            D.sort()
            X = [0, 100]
            new = LHSample(2, [D, X], int(S))[:, 0]

        if para in atom2:
            index = atom2.index(para)
            row1 = data[46 + 4 * (i - 1)].split()
            num = row1[index]
            lx = round(float(num), 4) - round(float(num), 4) * (int(percent) / 100)
            hx = round(float(num), 4) + round(float(num), 4) * (int(percent) / 100)
            D = [lx, hx]
            D.sort()
            X = [0, 100]
            new = LHSample(2, [D, X], int(S))[:, 0]

        if para in atom3:
            index = atom3.index(para)
            row1 = data[47 + 4 * (i - 1)].split()
            num = row1[index]
            lx = round(float(num), 4) - round(float(num), 4) * (int(percent) / 100)
            hx = round(float(num), 4) + round(float(num), 4) * (int(percent) / 100)
            D = [lx, hx]
            D.sort()
            X = [0, 100]
            new = LHSample(2, [D, X], int(S))[:, 0]

        if para in atom4:
            index = atom4.index(para)
            row1 = data[48 + 4 * (i - 1)].split()
            num = row1[index]
            lx = round(float(num), 4) - round(float(num), 4) * (int(percent) / 100)
            hx = round(float(num), 4) + round(float(num), 4) * (int(percent) / 100)
            D = [lx, hx]
            D.sort()
            X = [0, 100]
            new = LHSample(2, [D, X], int(S))[:, 0]

        if para in bond1:
            index = bond1.index(para)
            row1 = data[59 + 2 * (i - 1)].split()
            num = row1[index + 2]
            lx = round(float(num), 4) - round(float(num), 4) * (int(percent) / 100)
            hx = round(float(num), 4) + round(float(num), 4) * (int(percent) / 100)
            D = [lx, hx]
            D.sort()
            X = [0, 100]
            new = LHSample(2, [D, X], int(S))[:, 0]

        if para in bond2:
            index = bond2.index(para)
            row1 = data[60 + 2 * (i - 1)].split()
            num = row1[index]
            lx = round(float(num), 4) - round(float(num), 4) * (int(percent) / 100)
            hx = round(float(num), 4) + round(float(num), 4) * (int(percent) / 100)
            D = [lx, hx]
            D.sort()
            X = [0, 100]
            new = LHSample(2, [D, X], int(S))[:, 0]

        if para in diagonal:
            index = diagonal.index(para)
            row1 = data[72 + (i - 1)].split()
            num = row1[index + 2]
            lx = round(float(num), 4) - round(float(num), 4) * (int(percent) / 100)
            hx = round(float(num), 4) + round(float(num), 4) * (int(percent) / 100)
            D = [lx, hx]
            D.sort()
            X = [0, 100]
            new = LHSample(2, [D, X], int(S))[:, 0]

        if para in angle:
            index = angle.index(para)
            row1 = data[76 + (i - 1)].split()
            num = row1[index + 3]
            lx = round(float(num), 4) - round(float(num), 4) * (int(percent) / 100)
            hx = round(float(num), 4) + round(float(num), 4) * (int(percent) / 100)
            D = [lx, hx]
            D.sort()
            X = [0, 100]
            new = LHSample(2, [D, X], int(S))[:, 0]

        if para in torsion:
            index = torsion.index(para)
            row1 = data[95 + (i - 1)].split()
            num = row1[index + 4]
            lx = round(float(num), 4) - round(float(num), 4) * (int(percent) / 100)
            hx = round(float(num), 4) + round(float(num), 4) * (int(percent) / 100)
            D = [lx, hx]
            D.sort()
            X = [0, 100]
            new = LHSample(2, [D, X], int(S))[:, 0]

        if para in hydro:
            index = hydro.index(para)
            row1 = data[122 + (i - 1)].split()
            num = row1[index + 3]
            lx = round(float(num), 4) - round(float(num), 4) * (int(percent) / 100)
            hx = round(float(num), 4) + round(float(num), 4) * (int(percent) / 100)
            D = [lx, hx]
            D.sort()
            X = [0, 100]
            new = LHSample(2, [D, X], int(S))[:, 0]

        return str(type)+'_'+para, new, num

    def Create_trainset(self, data,path):
        data1 = {}
        key = []
        for i in range(len(data)):
            data1.update({data[i][0]: data[i][1]})
            key.append(data[i][0])
            trainset = pd.DataFrame(data1, columns=key)
            trainset.to_csv(path,
                            float_format='%.5f',
                            na_rep='我是空值',
                            index=False)

        return trainset

    def Create_reaxff(self, ts,data0, path):
        import numpy as np
        x = np.array(ts)
        for i in range(x.shape[0]):
            para = []
            for j in x[i]:
                para.append(str(j))

            origin = []
            for m in range(len(data0)):
                origin.append(data0[m][2])

            file = open("./data/template/ffield.reax.cho", 'r')
            file_new = open(path + 'ffield' + str(i) + '.reax.cho', 'w')
            for line in file:
                for k in range(x.shape[1]):
                    if origin[k] in line:
                        line = line.replace(origin[k], para[k])
                file_new.write(line)
            file.close()
            file_new.close()

#定义了方法来通过读取guassian软件的scan的log日志文件来转化为lammps可读取坐标文件
def get_coordinate(input_path,output_path):
    input_file1 = open(input_path,'r')
    line_init = input_file1.readline()
    while line_init:
        match = re.search('NAtoms=\s*(\d+)', line_init)
        if match:
            num_atom = int(match.group(1))
            break
        line_init = input_file1.readline()

    input_file = open(input_path , 'r')
    line = input_file.readline()

    cycle = 0
    Energy_path = output_path + 'Energy.txt'
    if os.path.exists(Energy_path):
        os.remove(Energy_path)
    no = 0
    while line:
        match = re.search(r'E\(RB3LYP\) =\s+(-?\d+\.\d+)\s+A\.U\.', line)
        if match:
            cycle += 1
            energy = match.group(1)
            with open(Energy_path, 'a', newline='') as f:
                f.writelines([str(cycle)+' '+str(energy)+'\n'])
                f.close()

        tmp1 = line.split()
        if len(tmp1) == 2:
            if tmp1[0] == "Z-Matrix" and tmp1[1] == "orientation:":
                no += 1
                line = input_file.readline()
                line = input_file.readline()
                line = input_file.readline()
                line = input_file.readline()
                output_file = open(output_path +'data_'+str(no),'w')

                output_file.write("LAMMPS data file Created\n")
                output_file.write("\n")
                output_file.write(str(num_atom) + " atoms\n")
                output_file.write("\n")
                output_file.write("3 atom types\n")
                output_file.write("\n")
                output_file.write("-20 20.0 xlo xhi\n")
                output_file.write("-20 20.0 ylo yhi\n")
                output_file.write("-20 20.0 zlo zhi\n")
                output_file.write("\n")
                output_file.write("Masses\n")
                output_file.write("\n")
                output_file.write("1 12.011\n")
                output_file.write("2 1.008\n")
                output_file.write("3 15.999\n")
                output_file.write("\n")
                output_file.write("Atoms\n")
                output_file.write("\n")
                for i in range(num_atom):
                    line = input_file.readline().split()
                    atom_id = line[0]
                    mol_id = "1"

                    if line[1] == "6":
                        atom_type = "1"
                    elif line[1] == "1":
                        atom_type = "2"
                    else:
                        atom_type = "3"
                    atom_q = "0"
                    x = line[3]
                    y = line[4]
                    z = line[5]
                    tmp = atom_id + " " + mol_id + " " + atom_type + " " + atom_q + " " + x + " " + y + " " + z + "\n"
                    output_file.write(tmp)
                output_file.close()
        line = input_file.readline()


#
def extarct_data(N,path,raw_path):
    with open(path, 'r') as f:
        lines = f.readlines()
    # 按照126列为一组，将数据分组并转换成CSV格式
    groups = [lines[i:i+N] for i in range(0, len(lines), N)]
    for group in groups:
        with open(raw_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([float(j)
                             for j in group])
        file.close()


if __name__ == '__main__':
    # get_coordinate('./data/dataset_dft/CH4/CH4.LOG','./data')
    extarct_data(399,'./lammps_running/run2/info2.data','./data/random_forest_data/output/')


