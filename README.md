# Predictive-Maintenance

This Python code is designed to create a real-time dashboard for visualizing sensor data predictions using machine learning models.
The dashboard is built using Dash and Plotly libraries.

**Features include**

   Real-time prediction and visualization of sensor data.
   Multiple charts for different sensor types: temperature, vibration, pressure, humidity, RPM, current, and voltage.
   Machine learning models for each sensor type.
   Predicted classes are highlighted with different background colors (e.g., red, orange, white).

**Requirements**   

    python(for coding) 
    VS Code(IDE)
    pandas 
    joblib(for load scalling file) 
    tensorflow (for creating feed forward neural network model for training) 
    multiprocessing(for runing all sensors prediction functions simultaneously)
    dash(for creating web application)
    json(for accepting data that come from server)
    random(for server side:generating random values for sensors)
    time(for timstamps
    socket(built connection between client and server)

**Workflow**

    The workflow for the provided code can be summarized as follows:

    Data Receiver:
              A server socket is set up to receive sensor data.
    Sensor data is received in JSON format, parsed, and modified.
    The modified data is put into a queue for processing.
              
    Prediction Process:
              Separate Processes are created for each sensor type (temperature, vibration, pressure, humidity, RPM, current, voltage).
    Each process loads a pre-trained machine learning model and a scaler.
    Sensor data is retrieved from the queue.
    Data is preprocessed and scaled using the loaded scaler.
    Predictions are made using the loaded machine learning model.
    The predicted class and sensor data are stored in respective lists.
    The process repeats at regular intervals (1 second).
    
    Dash Web Application:
          A Dash web application is created to visualize real-time sensor data and predictions.
    Multiple graphs are displayed in the application, each representing a sensor type.
    The application uses callback functions to update the graphs with the latest data at regular intervals (1 second).
    The background color of the graphs changes based on prediction values.

    Main Execution:
          The main script sets up a multiprocessing environment with separate processes for data reception and prediction threads.
    The data receiver process listens for incoming data.
    Prediction processes continuously process data and make predictions.
    The Dash web application is started and runs to display the real-time data.

   **steps to run code**

       1- Clone the Repository:
              Clone the repository containing the code from your version control system or download it as a ZIP file and extract it.
       2- Install Dependencies:
              Install python and python libraries.
       3- Run the Code
       4- Access the Dashboard:
              Once the script is running, open a web browser.
              Access the dashboard by navigating to the URL provided in the terminal, typically something like http://127.0.0.1:8050/.
       5- View Real-Time Sensor Data:
              The dashboard should display real-time sensor data and predictions in various graphs.
      6- Stop the Code:
              To stop the code execution, press Ctrl+C in the terminal where the script is running.

  **Good luck!**
