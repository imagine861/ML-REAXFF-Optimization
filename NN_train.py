from network import model_v_5
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import tensorflow as tf
from tensorflow.keras.callbacks import History,EarlyStopping,LearningRateScheduler,ModelCheckpoint


def train(X,y):
    scaler = StandardScaler()
    scaler2 = StandardScaler()
    X = scaler.fit_transform(X)
    y = scaler2.fit_transform(y)

    with open('./weight/scaler_x.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('./weight/scaler_y.pkl', 'wb') as f:
        pickle.dump(scaler2, f)
    X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.05,random_state=42)



    history = History()
    model = model_v_5(input_dim=X.shape[1],output_dim=y.shape[1])

    # optimizer = tf.keras.optimizers.SGD(learning_rate=10e-4, momentum=0.9)
    optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_absolute_error'])

    es = EarlyStopping(monitor='val_mean_absolute_error', mode='min', verbose=1, patience=10)

    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=128, callbacks=[history,es])

    model.save('./weight/best_model.h5')
    pd.DataFrame(history.history).to_csv('./log/log.csv')
    
if __name__ == "__main__":
    import glob
    X = pd.concat([pd.read_csv(i) for i in sorted(glob.glob('./data/ml_dataset/input/*.csv'),
                                                  key=lambda x: int(x.split('\\')[1].split('.')[0]))])
    y = pd.concat([pd.read_csv(i, header=None) for i in sorted(glob.glob('./data/ml_dataset/output/*.csv'),
                                                               key=lambda x: int(
                                                                   x.split('\\')[1].split('.')[0].split('y')[1]))])
    print(X.shape)
    print(y.shape)
    train(X, y)

