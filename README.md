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

    Generate random values for kx and ky
    For i from 1 to N:
        kx[i] ← random number in range [0, π)
        ky[i] ← random number in range [0, π)

    Calculate energy E for each (kx, ky)
    For i from 1 to N:
        E[i] ← -2 × (cos(kx[i]) + cos(ky[i]))

    Determine occupancy based on chemical potential mu
    For i from 1 to N:
        If E[i] ≤ mu:
            occupancy[i] ← 1
        Else:
            occupancy[i] ← 0

    Construct feature matrix X
    For i from 1 to N:
        X[i] ← [
            kx[i],
            ky[i],
            kx[i]^2,
            ky[i]^2,
            kx[i] × ky[i]
        ]

    Construct label vector y
    y ← occupancy as a column vector of shape (N, 1)

    Add bias term to X
    For i from 1 to N:
        X_b[i] ← [1, X[i][0], X[i][1], X[i][2], X[i][3], X[i][4]]

    Initialize parameter vector theta
    theta ← zero vector of shape (number of features + 1, 1)

    Train logistic regression model using gradient descent
    Create model instance: model1 ← LogisticRegressionUsingGD()
    model1.fit(X_b, y, theta, n_itr)

    Evaluate model accuracy
    accuracy ← model1.accuracy(X_b, y)

    Get trained parameters
    parameters ← model1.w_

    Return (accuracy, parameters)
- Support Vector Machine
- Comparsion between two training method with system size (N = 30 and N =200)


Assignment 3
- Test 3 system size 2_D Ising model and plot the magnetization
- principal component analysis


  




