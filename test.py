# import pandas as pd
# import glob
# data = pd.read_csv('./result/verify.csv')
# print(data)
# split_data,s = [],[]
# count = 0
# gold_dft = glob.glob('./data/data_for_GAopt/*.csv')
# for i in gold_dft:
#     data_x = pd.read_csv(i)
#     count += len(data_x)
#     s.append(count)
#     split_data.append(data_x)
# print(s)
# Reax_org = data['Reax_origin']
# Reax_new = data['Reax_new']
# current_index = 0
# for indx in s:
#     Reax_new[current_index:indx] = Reax_new[current_index:indx] - min(Reax_new[current_index:indx])
#     current_index = indx
# Reax_new.to_csv('./result/Reax_new.csv')


# import pandas as pd
#
# filename = './result/verify.csv'
# chunksize = 21
#
# reader = pd.read_csv(filename, chunksize=chunksize)
#
# for i, chunk in enumerate(reader):
#     chunk.to_csv(f'./result/verify_{i}.csv', index=False)

import os
import originpro as op

# 设置要遍历的文件夹路径
folder_path = r"D:/MLReaxopt/result/"

# 获取文件夹内所有CSV文件的路径
csv_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".csv")]

# 打开Origin软件
app = op.ApplicationSI()

# 遍历CSV文件并绘制散点图
for csv_file in csv_files:
    # 打开CSV文件
    wks = app.Worksheet()
    wks.FromFile(csv_file)

    # 获取数据范围
    x_range = wks.Columns(1)
    y_range = wks.Columns([2, 3, 4])

    # 绘制散点图
    graph = app.NewGraph(template="Scatter")
    plot = graph.AddPlot(x_range, y_range)
    plot.Lines.Type = 1
    plot.Symbols.Type = 0

    # 保存图形
    graph.Save(os.path.splitext(csv_file)[0] + ".ogg")

# 关闭Origin软件
app.Quit()