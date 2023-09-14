
                    #data send by server

import json
import random
import time
import socket

def generate_smooth_sensor_data():
    sensor_ranges = {
        "temperature": (20, 30),
        "vibration": (0, 5),
        "pressure": (100, 200),
        "humidity": (30, 70),
        "RPM": (1000, 3000),
        "current": (0, 10),
        "voltage": (220, 240),
        
    }

    smooth_data = {sensor: random.uniform(min_val, max_val) for sensor, (min_val, max_val) in sensor_ranges.items()}
    smoothing_factor = 0.2
    anomaly_sensor = None
    anomaly_counter = 0
    while True:
        for sensor in sensor_ranges:
            if anomaly_sensor is None or random.random() < 0.05:
                target_value = random.uniform(*sensor_ranges[sensor])
            else:
                target_value = smooth_data[anomaly_sensor]

                current_value = smooth_data[sensor]
                smooth_data[sensor] = current_value + smoothing_factor * (target_value - current_value)

        if random.random() < 0.01:
            anomaly_sensor = random.choice(list(sensor_ranges.keys()))
            anomaly_counter = random.randint(3, 10)
        elif anomaly_counter > 0:
            anomaly_counter -= 1
        else:
            anomaly_sensor = None
        time.sleep(0.1)
        return smooth_data  
        

if __name__ == "__main__":
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host = socket.gethostname()    
    port=1234
    sock.bind((host,port))
    sock.listen(1)
    communication_socket, address = sock.accept()
    print(f"Connection from {address} has been established.")
    while True:
        sensor_data = generate_smooth_sensor_data()
        sensor_data_json = json.dumps(sensor_data)
        communication_socket.send(sensor_data_json.encode("utf-8"))
        time.sleep(0.1)
    
    