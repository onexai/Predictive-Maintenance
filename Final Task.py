import time
import datetime
import numpy as np
import socket
import json
import joblib
import tensorflow as tf
from tensorflow import keras
import numpy as np
import multiprocessing
from multiprocessing import Queue
import plotly.graph_objs as go
import dash
from dash import dcc,html
from dash.dependencies import Input, Output

app = dash.Dash(__name__)
def data_receiver(data_queue, client):
    while True:
        data = client.recv(1024)
        try:
            sensor_data = json.loads(data.decode("utf-8"))
            # Process the received sensor data as needed
            print("Received sensor data:", sensor_data)
            for key,value in sensor_data.items():
                obtained_data=key,value
                modified_data=dict([obtained_data])
                data_queue.put(modified_data)
                print("modifies data=",modified_data)
        except json.JSONDecodeError as e:
            pass
#load model and scalar file
def load_saved_model(sensor):
    model_filename = f"model_{sensor}.keras"
    loaded_model = keras.models.load_model(model_filename)
    scalar_name=f"scaler_{sensor}.pkl"
    loaded_scalar=joblib.load(scalar_name)
    return loaded_model,loaded_scalar

def prediction_temperature(data_queue, temp_timestamps,temp_data,temp_predict):
    trained_model,loaded_scaler = load_saved_model("temp")

    while True:
        data_get = data_queue.get()
        if "temperature" in data_get:
            user_data = data_get.get("temperature")
            input_values = np.array(user_data).reshape(1, -1)
            input_data_scaled = loaded_scaler.transform(input_values)
            predictions = trained_model.predict(input_data_scaled)
            predicted_class = np.argmax(predictions, axis=-1)
            print("predicted class for temp=", predicted_class,"value=",user_data)

            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f") 

        #put values of current,data,prediction in their corresponding variable
            temp_timestamps.append(current_time)
            temp_data.append(user_data)
            temp_predict.append(predicted_class)    
        # Sleep for a short interval 
            time.sleep(1)

def prediction_vibration(data_queue,vib_timestamps,vib_data,vib_predict):
    trained_model, loaded_scaler= load_saved_model("vibration")

    while True:
        data_get = data_queue.get()
        if "vibration" in data_get:
            user_data = data_get.get("vibration")
            input_values = np.array(user_data).reshape(1, -1)
            input_data_scaled = loaded_scaler.transform(input_values)
            predictions = trained_model.predict(input_data_scaled)
            predicted_class = np.argmax(predictions, axis=-1)
            print("predicted class for vib=", predicted_class,"value=",user_data)
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

        #put values of current,data,prediction in their corresponding variable
            vib_timestamps.append(current_time)
            vib_data.append(user_data)
            vib_predict.append(predicted_class)    
        # Sleep for a short interval 
            time.sleep(1)

def prediction_pressure(data_queue,pre_timestamps,pre_data,pre_predict):
    trained_model, loaded_scaler = load_saved_model("pressure")

    while True:
        data_get = data_queue.get()
        if "pressure" in data_get:
            user_data = data_get.get("pressure")
            input_values = np.array(user_data).reshape(1, -1)
            input_data_scaled = loaded_scaler.transform(input_values)
            predictions = trained_model.predict(input_data_scaled)
            predicted_class = np.argmax(predictions, axis=-1)
            print("predicted class for pressure=", predicted_class,"value=",user_data)
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

        # put values of current,data,prediction in their corresponding variable
            pre_timestamps.append(current_time)
            pre_data.append(user_data)
            pre_predict.append(predicted_class)    
        # Sleep for a short interval 
            time.sleep(1)

def prediction_humidity(data_queue,humi_timestamps,humi_data,humi_predict):
    trained_model, loaded_scaler= load_saved_model("humidity")
    while True:
        data_get = data_queue.get()
        if "humidity" in data_get:
            user_data = data_get.get("humidity")
            input_values = np.array(user_data).reshape(1, -1)
            input_data_scaled = loaded_scaler.transform(input_values)
            predictions = trained_model.predict(input_data_scaled)
            predicted_class = np.argmax(predictions, axis=-1)
            print("predicted class for humidity=", predicted_class,"value=",user_data)
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

        # put values of current,data,prediction in their corresponding variable
            humi_timestamps.append(current_time)
            humi_data.append(user_data)
            humi_predict.append(predicted_class)    
        # Sleep for a short interval 
            time.sleep(1)
        
def prediction_RPM(data_queue,rpm_timestamps,rpm_data,rpm_predict):
    trained_model, loaded_scaler = load_saved_model("RPM")
    while True:
        data_get = data_queue.get()
        if "RPM" in data_get:
            user_data = data_get.get("RPM")
            input_values = np.array(user_data).reshape(1, -1)
            input_data_scaled = loaded_scaler.transform(input_values)
            predictions = trained_model.predict(input_data_scaled)
            predicted_class = np.argmax(predictions, axis=-1)
            print("predicted class for RPM=", predicted_class,"value=",user_data)
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

        # put values of current, data, prediction in their corresponding queues
            rpm_timestamps.append(current_time)
            rpm_data.append(user_data)
            rpm_predict.append(predicted_class)    
        # Sleep for a short interval 
            time.sleep(1)


def prediction_current(data_queue,cur_timestamps,cur_data,cur_predict):
    trained_model, loaded_scaler = load_saved_model("current")

    while True:
        data_get = data_queue.get()
        if "current" in data_get:
            user_data = data_get.get("current")
            input_values = np.array(user_data).reshape(1, -1)
            input_data_scaled = loaded_scaler.transform(input_values)
            predictions = trained_model.predict(input_data_scaled)
            predicted_class = np.argmax(predictions, axis=-1)
            print("predicted class for current=", predicted_class,"value=",user_data)
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

        # put values of current, data, prediction in their corresponding queues
            cur_timestamps.append(current_time)
            cur_data.append(user_data)
            cur_predict.append(predicted_class)    
        # Sleep for a short interval (e.g., 1 second)
            time.sleep(1)


def prediction_voltage(data_queue,volt_timestamps,volt_data,volt_predict):
    trained_model, loaded_scaler = load_saved_model("voltage")

    while True:
        data_get = data_queue.get()
        if "voltage" in data_get:
            user_data = data_get.get("voltage")
            input_values = np.array(user_data).reshape(1, -1)
            input_data_scaled = loaded_scaler.transform(input_values)
            predictions = trained_model.predict(input_data_scaled)
            predicted_class = np.argmax(predictions, axis=-1)
            print("predicted class for voltage=", predicted_class,"value=",user_data)
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

        # put values of current, data, prediction in their corresponding queues
            volt_timestamps.append(current_time)
            volt_data.append(user_data)
            volt_predict.append(predicted_class)    
        # Sleep for a short interval
            time.sleep(1)

if __name__ == '__main__':

    data_queue = multiprocessing.Queue(7)
    manager = multiprocessing.Manager()

    # Create multiple lists for storing data for all sensors
    temp_timestamps = manager.list()
    temp_data = manager.list()
    temp_predict = manager.list()
    
    vib_timestamps = manager.list()
    vib_data = manager.list()
    vib_predict = manager.list()

    pre_timestamps = manager.list()
    pre_data = manager.list()
    pre_predict = manager.list()

    humi_timestamps = manager.list()
    humi_data = manager.list()
    humi_predict = manager.list()

    rpm_timestamps = manager.list()
    rpm_data = manager.list()
    rpm_predict = manager.list()

    cur_timestamps = manager.list()
    cur_data = manager.list()
    cur_predict = manager.list()

    volt_timestamps = manager.list()
    volt_data = manager.list()
    volt_predict = manager.list()

    host = socket.gethostname()
    port = 1234
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((host, port))

    process_receiver = multiprocessing.Process(target=data_receiver, args=(data_queue, client))

    # Create separate processes for each sensor
    process_temperature = multiprocessing.Process(target=prediction_temperature, args=(data_queue, temp_timestamps,temp_data,temp_predict))
    process_vibration = multiprocessing.Process(target=prediction_vibration, args=(data_queue, vib_timestamps,vib_data,vib_predict))
    process_pressure = multiprocessing.Process(target=prediction_pressure, args=(data_queue,pre_timestamps,pre_data,pre_predict))
    process_humidity = multiprocessing.Process(target=prediction_humidity, args=(data_queue,humi_timestamps,humi_data,humi_predict))
    process_RPM = multiprocessing.Process(target=prediction_RPM, args=(data_queue,rpm_timestamps,rpm_data,rpm_predict))
    process_current = multiprocessing.Process(target=prediction_current, args=(data_queue,cur_timestamps,cur_data,cur_predict))
    process_voltage = multiprocessing.Process(target=prediction_voltage, args=(data_queue,volt_timestamps,volt_data,volt_predict))


    process_receiver.start()
    process_temperature.start()
    process_vibration.start()
    process_pressure.start()
    process_humidity.start()
    process_RPM.start()
    process_current.start()
    process_voltage.start()

    #create layout for app 
    app.layout = html.Div([
    # create thre sub div section which show no of charts i:e one div section have 3 charts
    html.Div([
        dcc.Graph(id='live-update-chart-temp', style={'width': '33%', 'display': 'inline-block'}),
        dcc.Graph(id='live-update-chart-vib', style={'width': '33%', 'display': 'inline-block'}),
        dcc.Graph(id='live-update-chart-pre', style={'width': '33%', 'display': 'inline-block'}),
        ]),
    
    html.Div([
        dcc.Graph(id='live-update-chart-humi', style={'width': '33%', 'display': 'inline-block'}),
        dcc.Graph(id='live-update-chart-rpm', style={'width': '33%', 'display': 'inline-block'}),
        dcc.Graph(id='live-update-chart-cur', style={'width': '33%', 'display': 'inline-block'}),
        ]),
    
    html.Div([
        dcc.Graph(id='live-update-chart-volt', style={'width': '45%', 'display': 'inline-block'}),
        ]),
    
    dcc.Interval(
        id='interval-component',
        interval=1000,  # Update every 1 second
        n_intervals=0
        )
    ])


    # Define separate callback functions for each chart
    @app.callback(Output('live-update-chart-temp', 'figure'), [Input('interval-component', 'n_intervals')])
    def update_graph_temp(n):
        return update_chart(temp_timestamps, temp_data, temp_predict, "Temperature Data", "temperature")

    @app.callback(Output('live-update-chart-vib', 'figure'), [Input('interval-component', 'n_intervals')])
    def update_graph_vib(n):

        return update_chart(vib_timestamps, vib_data, vib_predict, "vibration data", "vibration")
    
    @app.callback(Output('live-update-chart-pre', 'figure'), [Input('interval-component', 'n_intervals')])
    def update_graph_pre(n):
        return update_chart(pre_timestamps, pre_data, pre_predict, "pressure Data", "pressure")
    
    @app.callback(Output('live-update-chart-humi', 'figure'), [Input('interval-component', 'n_intervals')])
    def update_graph_humi(n):
        return update_chart(humi_timestamps, humi_data, humi_predict, "humidity Data", "humidity")
    
    @app.callback(Output('live-update-chart-rpm', 'figure'), [Input('interval-component', 'n_intervals')])
    def update_graph_rpm(n):
        return update_chart(rpm_timestamps, rpm_data, rpm_predict, "RPM Data", "RPM")
    
    @app.callback(Output('live-update-chart-cur', 'figure'), [Input('interval-component', 'n_intervals')])
    def update_graph_cur(n):
        return update_chart(cur_timestamps, cur_data, cur_predict, "current Data", "Current")
    
    @app.callback(Output('live-update-chart-volt', 'figure'), [Input('interval-component', 'n_intervals')])
    def update_graph_volt(n):
        return update_chart(volt_timestamps, volt_data, volt_predict, "voltage Data", "Voltage")
   
   #create update function which update all charts data
    def update_chart(timestamps, data, predict, title, sensor_name):
        predict_value = predict[-1] if predict else None

        if predict_value == 2:
            bg = "red"
        elif predict_value == 1:
            bg = "orange"
        else:
            bg = "white"

        trace = go.Scatter(x=list(timestamps), y=list(data), mode='lines+markers', name=sensor_name)
        layout = go.Layout(title=f'Live {title}', xaxis=dict(title='Time'), yaxis=dict(title=sensor_name), plot_bgcolor=bg)
        return {'data': [trace], 'layout': layout}

    app.run_server(debug=True, use_reloader=False)
