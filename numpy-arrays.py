import numpy as np

a = np.array([1,2,3])
b = np.array([(1,2,3), (4,5,6)])
c = np.array([[(1,2,3), (4,5,6)], [(3,2,1), (4,5,6)]])
d = np.eye(3)
e = np.arange(10,25,5)
f = np.linspace (2,5,3)
g = np.random.random((2,2))


print 'A one dimensional array is ', a
print 'A two dimensional array is ', b
print 'Same two dimensional array transposed ', b.T
print 'A three dimensional array is ', c
print 'An Identity Matrix is ', d
print 'An array of evenly distributed values is (step value) ', e
print 'An array of evenly distributed values is (number of steps) ', f
print 'An array with random values ', g


