# Multi-Layer-Perceptron-Model
Feedforward Neural Network (Multi-Layer Perceptron, MLP) 
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
            
            First Hidden Layer:
                  Contains 64 neurons with a ReLU (Rectified Linear Unit) activation function. 
            This layer learns complex patterns from the input data.
            Dropout Layer: Applies dropout regularization to prevent overfitting by
            randomly deactivating 50% of the neurons during each training iteration.  
            
            Second Hidden Layer: 
                 Contains 32 neurons,  also using a ReLU activation function,
            allowing the network to model 
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
