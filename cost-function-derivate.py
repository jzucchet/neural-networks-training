import numpy as np

# Basic functions
# a(1) = sigma (z(1))
# z(1) = w(1)*a(0)+b(n)
# Ck = (a(1)-y)^2

# Apply multivariable chain rule to solve the functions and gain a
# basic understanding about the structure of a neural network
# By chain rule we can apply:
# dCdw = dCda*dadz*dzdw and
# dCdb = dCda*dadz*dzdb

# Furthermore, the activation function (sigma) is the
# Hyperbolic Tangent tanh(z) and its derivate respect to z is:
# dTanh(z)dz = 1/cosh(z)^2

#Sigma function.
sigma = np.tanh

# Feed-forward equation.
def a1 (w1, b1, a0) :
  return sigma(w1 * a0 + b1)

# Cost function is the square of the difference between
# the network output and the training data output.
def C (w1, b1, x, y) :
  return (a1(w1, b1, x) - y)**2

# Returns -via multivariate chain rule- the derivative of the cost function with
# respect to the weight.
def dCdw (w1, b1, x, y) :
  z = w1 * x + b1
  dCda = 2 * (a1(w1, b1, x) - y) # Derivative of cost with activation
  dadz = 1/np.cosh(z)**2 # derivative of activation with weighted sum z
  dzdw = x # derivative of weighted sum z with weight
  return dCda * dadz * dzdw # Return the chain rule product.

# Returns -via multivariate chain rule- the derivative of the cost function with
# respect to the bias.
def dCdb (w1, b1, x, y) :
  z = w1 * x + b1
  dCda = 2 * (a1(w1, b1, x) - y)
  dadz = 1/np.cosh(z)**2
  dzdb = 1 # derivative of weighted sum z with bias
  return dCda * dadz * dzdb

# Let's start with an unfit weight and bias.
w1 = 2.3
b1 = -1.2
# We can test on a single data point pair of x and y.
x = 0
y = 1
# Output how the cost would change
# in proportion to a small change in the bias
print( dCdb(w1, b1, x, y) )
