# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 00:41:44 2024

@author: mcall
""" #1. Phase 1: Data Generation
# Filename: 01_generate_synthetic_sensor_data.py Purpose: Generate
# synthetic sensor data for 90 days with various parameters, including pressure, temperature, vibration, and more.
import pandas as pd
import numpy as np
import os

 # Set the output file path synthetic_sensor_data_90_days.csv 
output_dir = 'C:/Users/mcall/'
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, 'synthetic_sensor_data_90_days.csv')

# Set random seed for reproducibility
np.random.seed(42)
"""
  This seed ensures the same random data is generated every time, remove it and you will
  get different data every run.The number 42 is just an arbitrary value used to initialize 
  the random number generator in a deterministic way. You can use any integer as the seed to
  get reproducible results in your data generation. Number of data points (e.g., hourly readings for 90 days)"""

num_data_points = 90 * 24

# Define possible sensor types
sensor_types = ['Pressure', 'Temperature', 'Humidity', 'Vibration', 'Flow Rate', 'Power Consumption', 'Particle Count', 'Pressure Differential']

      #  Generate core sensor data with noise and patterns
data = pd.DataFrame({
    'timestamp': pd.date_range(start='2023-01-01', periods=num_data_points, freq='h'),
    'sensor_id': np.random.randint(1000, 1100, size=num_data_points),  # Unique sensor IDs between 1000 and 1100
    'sensor_type': np.random.choice(sensor_types, size=num_data_points),  # Randomly select a sensor type for each row
    'pressure_deviation': np.random.normal(25, 5, size=num_data_points),
    'temperature': np.random.normal(25, 2, size=num_data_points),
    'humidity': np.random.normal(50, 10, size=num_data_points),
    'vibration': np.random.normal(0.02, 0.01, size=num_data_points),
    'flow_rate': np.random.normal(100, 10, size=num_data_points),
    'power_consumption': np.random.normal(1500, 200, size=num_data_points),
    'particle_count': np.random.normal(50, 10, size=num_data_points),
    'pressure_differential': np.random.normal(10, 2, size=num_data_points),
    'equipment_age': np.random.randint(1, 10, size=num_data_points),
    'usage_pattern': np.random.choice(['Low', 'Medium', 'High'], size=num_data_points, p=[0.2, 0.5, 0.3]),
    'equipment_state': np.random.choice(['Startup', 'Running', 'Shutdown', 'Maintenance'], size=num_data_points, p=[0.1, 0.7, 0.1, 0.1])
})

 # One-hot encode categorical columns
data = pd.get_dummies(data, columns=['equipment_state', 'usage_pattern'], drop_first=True)

# Add columns for fault condition and calibration
data['fault_condition'] = np.where((data['pressure_deviation'] > 30) | (data['vibration'] > 0.05), 1, 0)
data['time_since_last_calibration'] = np.random.randint(1, 180, size=num_data_points)
data['calibration_needed'] = np.where((data['fault_condition'] == 1) & (data['time_since_last_calibration'] > 90), 1, 0)

   # Save to .CSV 
data.to_csv(output_file, index=False)
print(f"Synthetic sensor data saved to: {output_file}")

'''
 Explanation of Fault Condition Logic

    np.where: This function is used to evaluate a condition and return 1 (indicating a fault) if the condition is true or 0 if the condition is false.

    Condition:
        (data['pressure_deviation'] > 30): Checks if the pressure_deviation is greater than 30. If this condition is true, it indicates an abnormal pressure state, which might suggest a fault.
        (data['vibration'] > 0.05): Checks if the vibration value exceeds 0.05. High vibration levels can indicate issues with the equipment, potentially signaling a fault.
        Logical OR (|): Combines the two conditions. If either the pressure deviation is above 30 or the vibration exceeds 0.05, the row is marked as a fault.

    Setting the fault_condition:
        If either of the conditions is true, the value of fault_condition is set to 1.
        If neither condition is met, the value of fault_condition is set to 0.

            Summary:

     The fault_condition is determined in the synthetic data using this logic:

        A fault is flagged (1) if:
        Pressure deviation is greater than 30, OR
        Vibration is greater than 0.05.
        If neither condition is met, the fault_condition is set to 0.

                 Modify these thresholds (30 for pressure deviation, 0.05 for vibration) to simulate different fault conditions in your dataset as per the needs of your project.
'''