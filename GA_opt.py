import pandas as pd
import numpy as np
import tensorflow.keras as keras
import csv
import glob
import pickle
import math
#
with open('./weight/scaler_x.pkl', 'rb') as f:  
    scaler = pickle.load(f)
with open('./weight/scaler_y.pkl', 'rb') as f:
    scaler2 = pickle.load(f)
# 导入深度学习模型
model = keras.models.load_model('./weight/best_model.h5')

gold_dft = glob.glob('./data/data_for_GAopt/*.csv')
print(gold_dft)
split_data,s = [],[]
count = 0
for i in gold_dft:
    data = pd.read_csv(i)
    count += len(data)
    s.append(count)
    split_data.append(data)
dft_data = pd.concat(split_data)['DFT']
dft_data.to_csv('./result/verify.csv')
# #1.基因编码
with open('./data/data_for_GAopt/raw.txt','r') as f:
    data = f.readlines()
# print(data)
parameter = []

for i in data:
    parameter.append([float(i)*(1-0.5),float(i)*(1+0.5)])
# print(parameter)
DNA_SIZE = 24
POP_SIZE = 100
DNA_Length = 50

pop = np.random.randint(2, size=(POP_SIZE, DNA_SIZE*DNA_Length)) #matrix (POP_SIZE, DNA_SIZE*2)

def translateDNA(pop):#pop表示种群矩阵，一行表示一个二进制编码表示的DNA，矩阵的行数为种群数目
    X = np.zeros((pop.shape[0],DNA_Length))
    for i in range(pop.shape[0]):
        for j in range(1,DNA_Length+1):
            x_pop = pop[i][DNA_SIZE*(j-1):DNA_SIZE*j]#奇数列表示X
            X_BOUND = parameter[j-1]
            #pop:(POP_SIZE,DNA_SIZE)*(DNA_SIZE,1) --> (POP_SIZE,1)完成解码
            X[i][j-1]=x_pop.dot(2**np.arange(DNA_SIZE)[::-1])/float(2**DNA_SIZE-1)*(X_BOUND[1]-X_BOUND[0])+X_BOUND[0]

    return X
#
def Loss(y,res):

    return np.sum(np.array(np.power(res - y, 1)))/len(res)


# x = translateDNA(pop)
# # x = scaler.transform(x)
# num = scaler2.inverse_transform(model.predict(x))
#2、构造适应度函数
def get_fitness(pop,model):

    x = translateDNA(pop)
    x = scaler.transform(x)
    num = scaler2.inverse_transform(model.predict(x))
    for n in range(len(num)):
        current_index = 0
        for indx in s:
            num[n][current_index:indx] = num[n][current_index:indx] - min(num[n][current_index:indx])
            current_index = indx

    result = np.sum(np.abs(np.matrix(num)-np.matrix(dft_data)),axis=1)#绝对平均误差
    result_n = np.log(result)

    L =max(result_n)  - result_n #适应函数F=Cmax-f

    # L = 1/result + 1e-3#适应函数F=Cmax-f
    Cmult = 1 #C属于[1,2]
    alpha = ((Cmult-1)*np.mean(L))/(np.max(L)-np.mean(L))
    beta = (np.abs((np.max(L)-Cmult*np.mean(L)))*np.mean(L))/(np.max(L)-np.mean(L))
    L2 = alpha*L+beta#线性变换

    beta = (np.abs((np.max(L)-Cmult*np.mean(L)))*np.mean(L))/(np.max(L)-np.mean(L))

    return np.array(L).ravel(),min(result)
# get_fitness(pop,model)
# print('res',res)
# print('l',L)


def select(pop, fitness):    # nature selection wrt pop's fitness
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,
                           p=(fitness)/(fitness.sum()) )
    return pop[idx]

# print(select(pop,fitness=get_fitness(pop,model)))

def crossover_and_mutation(pop, CROSSOVER_RATE=0.8):
    new_pop = []
    for father in pop:		#遍历种群中的每一个个体，将该个体作为父亲
        child = father		#孩子先得到父亲的全部基因（这里我把一串二进制串的那些0，1称为基因）
        if np.random.rand() < CROSSOVER_RATE:			#产生子代时不是必然发生交叉，而是以一定的概率发生交叉
            mother = pop[np.random.randint(POP_SIZE)]	#再种群中选择另一个个体，并将该个体作为母亲
            cross_points = np.random.randint(low=0, high=DNA_SIZE*DNA_Length)	#随机产生交叉的点
            child[cross_points:] = mother[cross_points:]		#孩子得到位于交叉点后的母亲的基因
        mutation(child)	#每个后代有一定的机率发生变异
        new_pop.append(child)

    return new_pop

def mutation(child, MUTATION_RATE=0.003):
    if np.random.rand() < MUTATION_RATE: 				#以MUTATION_RATE的概率进行变异
        mutate_point = np.random.randint(0, DNA_SIZE)	#随机产生一个实数，代表要变异基因的位置
        child[mutate_point] = child[mutate_point]^1 	#将变异点的二进制为反转

def print_info(pop):
    fitness,loss = get_fitness(pop,model)
    max_fitness_index = np.argmax(fitness)
    print("最小误差为:",float(loss))
    x = translateDNA(pop)

    return x[max_fitness_index],float(fitness)


N_GENERATIONS = 1
for _ in range(N_GENERATIONS):# 种群迭代进化N_GENERATIONS代
    print(_)
    crossover_and_mutation(pop, CROSSOVER_RATE=0.9)  # 种群通过交叉变异产生后代
    fitness = get_fitness(pop,model)[0]  # 对种群中的每个个体进行评估
    pop = select(pop, fitness)  # 选择生成新的种群
    best_x,score = print_info(pop)
    with open('./log/GA_log.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([_,score])

best_x = best_x.reshape(1,len(data))
result = scaler2.inverse_transform(model.predict(best_x))[0]
replace_value = [round(i,4) for i in best_x.reshape(len(data))]
current_index = 0
for indx in s:
    result[current_index:indx] = result[current_index:indx] - min(result[current_index:indx])
    current_index = indx
file = open("./data/template/ffield.reax.cho", 'r')
file_new = open('./result/ffield_new' + '.reax.cho', 'w')
for line in file:
    for k in range(len(data)):
        if str(float(data[k])) in line:
            line = line.replace(str(float(data[k])), str(replace_value[k]))
    file_new.write(line)
file.close()
file_new.close()
pd.DataFrame(best_x.T).to_csv('./result/best_parameter.csv',columns=None,index_label=None)
pd.DataFrame(result).to_csv('./result/verify.csv',columns=None,index_label=None)

