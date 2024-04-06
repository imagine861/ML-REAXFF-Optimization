from network import model_v_5
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import tensorflow as tf
from tensorflow.keras.callbacks import History,EarlyStopping,LearningRateScheduler,ModelCheckpoint

with open('./weight/scaler_x.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('./weight/scaler_y.pkl', 'rb') as f:
    scaler2 = pickle.load(f)
# 导入深度学习模型
model = tf.keras.models.load_model('./weight/best_model.h5')
x = pd.read_csv('./data/ml_dataset/input/verify/1.csv')
x = scaler.transform(x)
num = scaler2.inverse_transform(model.predict(x))
pd.DataFrame(num).to_csv('./data/ml_dataset/output/verify/verify.csv',columns=None,index_label=None)
