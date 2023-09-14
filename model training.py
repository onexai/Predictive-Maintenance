import pandas as pd
import joblib
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def data_wrangling():
    updated_data=pd.read_csv(input("enter file_path file_name for training "))
    model_training_data=updated_data
    return model_training_data
def data_scalling(new):
    scalar=StandardScaler()
    x=new[["temperature"]]
    y=new[["alert_status_temp"]]

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    X_tr_scaled = scalar.fit_transform(X_train)
    X_te_scaled = scalar.transform(X_test)

    joblib.dump(scalar, "scaler_temp.pkl")

    return X_tr_scaled,X_te_scaled,y_train,y_test

def model_temp(x_trs,x_tes,y_tr,y_te):
    model_temp=Sequential()
    model_temp.add(layers.Input(shape=(1,)))
    model_temp.add(layers.Dense(64,activation="relu"))
    model_temp.add(layers.Dense(3,activation="softmax"))
    model_temp.compile(loss="sparse_categorical_crossentropy",
                       metrics=["accuracy"],
                       optimizer=keras.optimizers.Adam(learning_rate=0.001))
    model_temp.fit(x_trs,y_tr,epochs=100,batch_size=32,verbose=2)
    model_evaluation=model_temp.evaluate(x_tes,y_te,batch_size=32,verbose=2)
    model_temp.save("model_temp.keras")
    print("loss=",model_evaluation[0])
    print("accuracy=",model_evaluation[1])

def data_scalling_current(new):
    scalar=StandardScaler()
    x=new[["current"]]
    y=new[["alert_status_cur"]]

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    X_tr_scaled = scalar.fit_transform(X_train)
    X_te_scaled = scalar.transform(X_test)

    joblib.dump(scalar, "scaler_current.pkl")
    return X_tr_scaled,X_te_scaled,y_train,y_test

def model_current(x_trs,x_tes,y_tr,y_te):
    model_temp=Sequential()
    model_temp.add(layers.Input(shape=(1,)))
    model_temp.add(layers.Dense(64,activation="relu"))
    model_temp.add(layers.Dense(3,activation="softmax"))
    model_temp.compile(loss="sparse_categorical_crossentropy",
                       metrics=["accuracy"],
                       optimizer=keras.optimizers.Adam(learning_rate=0.001))
    model_temp.fit(x_trs,y_tr,epochs=100,batch_size=32,verbose=2)
    model_evaluation=model_temp.evaluate(x_tes,y_te,batch_size=32,verbose=2)
    model_temp.save("model_current.keras")
    print("loss=",model_evaluation[0])
    print("accuracy=",model_evaluation[1])

def data_scalling_vibration(new):
    scalar=StandardScaler()
    x=new[["vibration"]]
    y=new[["alert_status_vibration"]]

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    X_tr_scaled = scalar.fit_transform(X_train)
    X_te_scaled = scalar.transform(X_test)

    joblib.dump(scalar, "scaler_vibration.pkl")
    return X_tr_scaled,X_te_scaled,y_train,y_test

def model_vibration(x_trs,x_tes,y_tr,y_te):
    model_temp=Sequential()
    model_temp.add(layers.Input(shape=(1,)))
    model_temp.add(layers.Dense(64,activation="relu"))
    model_temp.add(layers.Dense(3,activation="softmax"))
    model_temp.compile(loss="sparse_categorical_crossentropy",
                       metrics=["accuracy"],
                       optimizer=keras.optimizers.Adam(learning_rate=0.001))
    model_temp.fit(x_trs,y_tr,epochs=100,batch_size=32,verbose=2)
    model_evaluation=model_temp.evaluate(x_tes,y_te,batch_size=32,verbose=2)
    model_temp.save("model_vibration.keras")
    print("loss=",model_evaluation[0])
    print("accuracy=",model_evaluation[1])

def data_scalling_pressure(new):
    scalar=StandardScaler()
    x=new[["pressure"]]
    y=new[["alert_status_pre"]]

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    X_tr_scaled = scalar.fit_transform(X_train)
    X_te_scaled = scalar.transform(X_test)

    joblib.dump(scalar, "scaler_pressure.pkl")
    return X_tr_scaled,X_te_scaled,y_train,y_test

def model_pressure(x_trs,x_tes,y_tr,y_te):
    model_temp=Sequential()
    model_temp.add(layers.Input(shape=(1,)))
    model_temp.add(layers.Dense(64,activation="relu"))
    model_temp.add(layers.Dense(3,activation="softmax"))
    model_temp.compile(loss="sparse_categorical_crossentropy",
                       metrics=["accuracy"],
                       optimizer=keras.optimizers.Adam(learning_rate=0.001))
    model_temp.fit(x_trs,y_tr,epochs=100,batch_size=32,verbose=2)
    model_evaluation=model_temp.evaluate(x_tes,y_te,batch_size=32,verbose=2)
    model_temp.save("model_pressure.keras")
    print("loss=",model_evaluation[0])
    print("accuracy=",model_evaluation[1])

def data_scalling_humidity(new):
    scalar=StandardScaler()
    x=new[["humidity"]]
    y=new[["alert_status_humi"]]

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    X_tr_scaled = scalar.fit_transform(X_train)
    X_te_scaled = scalar.transform(X_test)

    joblib.dump(scalar, "scaler_humidity.pkl")
    return X_tr_scaled,X_te_scaled,y_train,y_test

def model_humidity(x_trs,x_tes,y_tr,y_te):
    model_temp=Sequential()
    model_temp.add(layers.Input(shape=(1,)))
    model_temp.add(layers.Dense(64,activation="relu"))
    model_temp.add(layers.Dense(3,activation="softmax"))
    model_temp.compile(loss="sparse_categorical_crossentropy",
                       metrics=["accuracy"],
                       optimizer=keras.optimizers.Adam(learning_rate=0.001))
    model_temp.fit(x_trs,y_tr,epochs=100,batch_size=32,verbose=2)
    model_evaluation=model_temp.evaluate(x_tes,y_te,batch_size=32,verbose=2)
    model_temp.save("model_humidity.keras")
    print("loss=",model_evaluation[0])
    print("accuracy=",model_evaluation[1])

def data_scalling_RPM(new):
    scalar=StandardScaler()
    x=new[["RPM"]]
    y=new[["alert_status_RPM"]]

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    X_tr_scaled = scalar.fit_transform(X_train)
    X_te_scaled = scalar.transform(X_test)

    joblib.dump(scalar, "scaler_RPM.pkl")
    return X_tr_scaled,X_te_scaled,y_train,y_test

def model_RPM(x_trs,x_tes,y_tr,y_te):
    model_temp=Sequential()
    model_temp.add(layers.Input(shape=(1,)))
    model_temp.add(layers.Dense(64,activation="relu"))
    model_temp.add(layers.Dense(3,activation="softmax"))
    model_temp.compile(loss="sparse_categorical_crossentropy",
                       metrics=["accuracy"],
                       optimizer=keras.optimizers.Adam(learning_rate=0.001))
    model_temp.fit(x_trs,y_tr,epochs=100,batch_size=32,verbose=2)
    model_evaluation=model_temp.evaluate(x_tes,y_te,batch_size=32,verbose=2)
    model_temp.save("model_RPM.keras")
    print("loss=",model_evaluation[0])
    print("accuracy=",model_evaluation[1])

def data_scalling_voltage(new):
    scalar=StandardScaler()
    x=new[["voltage"]]
    y=new[["alert_status_vol"]]

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    X_tr_scaled = scalar.fit_transform(X_train)
    X_te_scaled = scalar.transform(X_test)

    joblib.dump(scalar, "scaler_voltage.pkl")
    return X_tr_scaled,X_te_scaled,y_train,y_test

def model_voltage(x_trs,x_tes,y_tr,y_te):
    model_temp=Sequential()
    model_temp.add(layers.Input(shape=(1,)))
    model_temp.add(layers.Dense(64,activation="relu"))
    model_temp.add(layers.Dense(3,activation="softmax"))
    model_temp.compile(loss="sparse_categorical_crossentropy",
                       metrics=["accuracy"],
                       optimizer=keras.optimizers.Adam(learning_rate=0.001))
    model_temp.fit(x_trs,y_tr,epochs=100,batch_size=32,verbose=2)
    model_evaluation=model_temp.evaluate(x_tes,y_te,batch_size=32,verbose=2)
    model_temp.save("model_voltage.keras")
    print("loss=",model_evaluation[0])
    print("accuracy=",model_evaluation[1])

#data wrangling run for each model
result=data_wrangling()
#temperature model training
a1,b2,c3,d4=data_scalling(result)
model_temp(a1,b2,c3,d4)
#current model training
a1,b2,c3,d4=data_scalling_current(result)
model_current(a1,b2,c3,d4)
#vibration model training
a1,b2,c3,d4=data_scalling_vibration(result)
model_vibration(a1,b2,c3,d4)
#pressure model training
a1,b2,c3,d4=data_scalling_pressure(result)
model_pressure(a1,b2,c3,d4)
#humidity model training
a1,b2,c3,d4=data_scalling_humidity(result)
model_humidity(a1,b2,c3,d4)
#RPM model training
a1,b2,c3,d4=data_scalling_RPM(result)
model_RPM(a1,b2,c3,d4)
#voltage model training
a1,b2,c3,d4=data_scalling_voltage(result)
model_voltage(a1,b2,c3,d4)