import numpy as np

# Activation equations are:
# a(1) = sigma (z(1))
# z(1) = w(1)*a(0)+b(1)
# where z(1) is the weighted sum of activation and bias.

# For a particular input, x, and desired output y, we can define the cost
# of that specific training example as the square of the difference
# between the network's output and the desired output, that is,
# Ck = (a(1)-y)^2

# Where k labels the training example and a(1) a is assumed to be
# the activation of the output neuron when the input neuron a(0) is set to x

# Now, a NOT function would be that whom for the input x = 1 we would like that
# the network outputs y = 0

# For instance, if the starting weight and bias are w(1) = 1.3 and b(1) = -0.1,
# the network actually outputs a(1) = 0.834. If we work out the cost function
# for this example, we get
# Ck = (0.834-0)^2 = 0.696

#First we set the state of the network
sigma = np.tanh
w1 = 1.3
b1 = -0.1

# Then we define the neuron activation.
def a1(a0) :
  return sigma(w1 * a0 + b1)

# Experiment with different values of x below.
# Then do the same calculation for an input x = 0 and desired output y = 1.
x = 0
print(pow(a1(x)-1,2))
