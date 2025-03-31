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

  




