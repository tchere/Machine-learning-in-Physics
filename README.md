# 🤖 Machine Learning in Physics


## 🎯 Objective
We apply different supervised machine learning techinques to solve the physics problem, including the 2-D classical Ising model, classification problem and the second order polynomial equation.

## 🧠 What I Learned
- Linear Regression (including gradient descent, steepest descent and conjugate descent)
- Regularization (principle of Lagrange multiplier)
- Logistic Regression (physics example: Fermi-Dirac distribution from Statistical mechanics)
- Support Vector Machine
- Principal Component Analysis
- K-mean clustering
- Neural Network (forward and backward propagation)

## 🚀 Assignment description 

Assignment 1 
- Basic series calculation (fibonacci sequence)
- Basic Matrix mutiplication through numpy
- Gradient descent to solve the A quadratic polynomial, here is the sample Pseudocode:
  Function GradientDescent(A, b, x, alpha = 0.01, imax = 1000, eps = 0.001):

    Initialize steps = [(x₀, x₁)]          # Store initial guess
    Initialize residuals = []              # Store residual norms
    Set i = 0                              # Iteration counter

    Compute r = b - A * x                  # Initial residual
    Compute delta = rᵀ * r                 # Squared residual norm
    Set delta0 = delta                     # Store initial residual norm

    While i < imax AND delta > (eps² * delta0):

        Update x = x + alpha * r           # Gradient descent step
        Append (x₀, x₁) to steps           # Save current x to steps

        Compute r = b - A * x              # New residual
        Compute delta = rᵀ * r             # New squared residual norm

        Append sqrt(delta) to residuals    # Store residual norm

        Increment i = i + 1

    End While

    Return x, steps, i, residuals


Assignement 2 
- Sigmoid function application
- Iris dataset training 
    - load the data
    - split the data to training and testing dataset
    - Train the model (cost: Corss entropy; activation: hyperpolitic tangent)
    - model Prediction

- Logestic Regression (example with Fermi Surface)
  
    Function logistic_regression_model(N, mu, n_itr = 50000):

    #Generate random values for kx and ky
  
    For i from 1 to N:
  
        kx[i] ← random number in range [0, π)
        ky[i] ← random number in range [0, π)

    #Calculate energy E for each (kx, ky)
  
    For i from 1 to N:
        E[i] ← -2 × (cos(kx[i]) + cos(ky[i]))

    #Determine occupancy based on chemical potential mu
    For i from 1 to N:
        If E[i] ≤ mu:
            occupancy[i] ← 1
        Else:
            occupancy[i] ← 0

    #Construct feature matrix X
    For i from 1 to N:
        X[i] ← [
            kx[i],
            ky[i],
            kx[i]^2,
            ky[i]^2,
            kx[i] × ky[i]
        ]

    #Construct label vector y
  
    y ← occupancy as a column vector of shape (N, 1)

    #Add bias term to X
  
    For i from 1 to N:
        X_b[i] ← [1, X[i][0], X[i][1], X[i][2], X[i][3], X[i][4]]

    #Initialize parameter vector theta
  
    theta ← zero vector of shape (number of features + 1, 1)

    #Train logistic regression model using gradient descent
  
    Create model instance: model1 ← LogisticRegressionUsingGD()
    model1.fit(X_b, y, theta, n_itr)

    #Evaluate model accuracy
  
    accuracy ← model1.accuracy(X_b, y)

    #Get trained parameters

    parameters ← model1.w_

    Return (accuracy, parameters)
  
- Support Vector Machine
- Comparsion between two training method with system size (N = 30 and N =200)


Assignment 3
- Test 3 system size 2_D Ising model and plot the magnetization
- principal component analysis
  - creation of covarance matrix
  - Do the Singular value decomposition of the matrix
  - extract the the first 2 eigenvector as the principal component (factor of 2 is dimensional dependent)
  - Do the K-mean classification on the graph of pca, there should be 3 centroid of 3 pca
  - Analysis on the feature of eigenvector

Assignment 4
- Parpare X and Y, define the signmoid function as the activation function
- Do the class of Neural network to to train the model, the network's structure is shown as follow:
  
    Class Network:

  Constructor(n_inputs, n_layers, n_neurons_per_layer, n_outputs):
    - Set input/output/layer configurations
    - If n_neurons_per_layer is a single int, repeat it for all layers
    - Construct full neuron architecture: [input] + [hidden layers] + [output]
    - Initialize weights randomly for each layer (including bias)

  Method forward_propagation(input_row):
    - Initialize activations list
    - For each layer:
        - Compute activations using weights and activation function
        - Store activations for backprop
    - Return final output (prediction)

  Method backward_propagation(actual, predicted):
    - Initialize list of deltas (error signals)
    - For each layer from output to input:
        - Compute error (difference or propagated error)
        - Multiply by derivative of activation function
        - Store delta
    - Reverse deltas to match layer order

  Method update_weights(input_row, learning_rate):
    - For each layer:
        - Use activations from previous layer (or input for first layer)
        - Update each weight using delta and learning rate

  Method training(training_data, actual_data, train=1, learning_rate=0.05, n_epoch=1):
    - For each epoch:
        - For each training sample:
            - Forward propagate to get prediction
            - Track accuracy and cost
            - If training enabled:
                - Backpropagate error
                - Update weights
    - Store final accuracy and cost
- build the Network A and B
  Network A ( 100 input data point, 1 hidden layer, 3 neurons each layer, 1 output layer)
  Network B ( 100 input data point, 2 hidden layer, 3 neurons each layer, 1 output layer)

- Accuracy and Network Comparsion


  




