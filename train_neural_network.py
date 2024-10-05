# -*- coding: utf-8 -*-
"""
This project uses a Feedforward Neural Network (Multi-Layer Perceptron, MLP) to
 predict equipment calibration needs in a manufacturing environment based on 
 synthetic sensor data.
Created on Fri Sep 27 19:19:05 2024
@author:  mcall 
"""            # 2. Phase 2: Model Training
               # Filename: 02_train_neural_network.py Purpose: Train a neural network
               # model on the synthetic data and save the model and scaler for future predictions.

import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


                         # Load the synthetic dataset
data = pd.read_csv('C:/Users/mcall/synthetic_sensor_data_90_days.csv')

                   # One-hot encode categorical columns (ensure consistency with training)
categorical_columns = ['sensor_type', 'equipment_state', 'usage_pattern']
for col in categorical_columns:
    if col in data.columns:
        data = pd.get_dummies(data, columns=[col], drop_first=True)

                       # Save the column names after one-hot encoding
training_columns = data.columns.tolist()
with open('C:/Users/mcall/training_columns.txt', 'w') as file:
    for column in training_columns:
        file.write(f"{column}\n")

         # Features for model training (excluding non-feature columns)
features = data.drop(columns=['timestamp', 'calibration_needed', 'sensor_id'])
labels = data['calibration_needed']

                               # Train/test split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

                   # Normalize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

                                # Save the scaler for future use
joblib.dump(scaler, 'C:/Users/mcall/scaler.pkl')

                            # Define the neural network model / relu = (Rectified Linear Unit)
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))  # Binary classification

                              # Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

                            # Train the model
model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2)

                        # Save the trained model
model.save('C:/Users/mcall/trained_neural_network.h5')

print("Model training complete and saved successfully.")
"""
 Neural Network Summary: Feedforward Neural Network for Predictive Maintenance


The neural network processes various sensor readings to identify patterns 
indicative of potential faults or calibration requirements.
 The goal is to enable proactive maintenance and optimize equipment uptime. 
 
                            Neural Network Architecture

    Input Layer:
        The input layer receives multiple features from the dataset, including processed
        numerical data (e.g., pressure deviation, temperature, humidity) 
        and one-hot encoded categorical data (e.g., sensor types, equipment states).
        The number of input neurons corresponds to the number of features in the
        training data (after preprocessing).

    Hidden Layers:
        The network includes two hidden layers:
            First Hidden Layer: Contains 64 neurons with a ReLU (Rectified Linear Unit) activation function. 
            This layer learns complex patterns from the input data.
            Dropout Layer: Applies dropout regularization to prevent overfitting by
            randomly deactivating 50% of the neurons during each training iteration.
            Second Hidden Layer: Contains 32 neurons, 
            also using a ReLU activation function, allowing the network to model 
            intricate relationships in the data.
            Another Dropout Layer: Similar to the first,
            it helps prevent overfitting by randomly deactivating neurons.

    Output Layer:
        A single neuron with a sigmoid activation function. This layer produces a
        probability between 0 and 1, representing whether
        equipment calibration is needed.
        The output is then thresholded (e.g., > 0.5) to produce a binary classification
        (1 = calibration needed, 0 = no calibration needed).

    Loss Function and Optimizer:
        Loss Function: Binary cross-entropy, which is appropriate for binary 
        classification tasks.
        Optimizer: Adam (Adaptive Moment Estimation) is used for efficient and
        adaptive learning during training.

   Model Training

    The neural network is trained using the synthetic dataset that includes 
    various sensor readings and equipment states over a 90-day period.
    A StandardScaler is used to normalize input features, which helps 
    stabilize the training 
    process and improve model performance.
    The model is trained for a specified number of epochs with a mini-batch size to 
    iteratively adjust the network weights using backpropagation.

   Prediction Process

    During prediction, new sensor data undergoes the same preprocessing steps as the training data, 
    including one-hot encoding for
    categorical features and scaling for numerical features.
    The neural network outputs a probability indicating the likelihood that calibration is needed. 
    The final output is a binary decision (0 or 1), highlighting which equipment requires attention.

    Highlights and Capabilities

    Fault Detection: The model can identify potential faults in the equipment by analyzing sensor 
    readings such as pressure deviation and vibration levels.
    Proactive Maintenance: By predicting calibration needs, the model enables proactive maintenance, 
    reducing unplanned downtime and increasing operational efficiency.
    Scalability: The network can be retrained with additional sensors and data as new 
    information becomes available, making it adaptable to various manufacturing environments.

Conclusion

This feedforward neural network provides a sophisticated 
yet flexible solution for predictive maintenance in manufacturing facilities. By analyzing 
real-time sensor data, it supports the timely calibration of equipment, helping 
maintain high operational standards and reducing unexpected failures.
"""