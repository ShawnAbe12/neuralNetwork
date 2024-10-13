import numpy as np
#sigmoid function
def nonlin(x,deriv=False):
    if(deriv == True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

#Input Dataset
X = np.array([[0,0,1],
              [0,1,1],
              [1,0,1],
              [1,1,1]   ])
#Output Dataset
y = np.array([[0,0,1,1]]).T

#Seed random numbers to make calculations
# deterministic (just good practice)
np.random.seed(1)

#initialize weights randomly with mean 0
syn0 = 2*np.random.random((3,1)) - 1

for iter in range(10000):

    #forward propagation
    l0 = X
    l1 = nonlin(np.dot(l0,syn0))


    #how much did we miss
    l1_error = y - l1
    # multiply how much we missed by the 
    # slope of the sigmoid at the value in l1
    l1_delta = l1_error * nonlin(l1, True)

    syn0 += np.dot(l0.T,l1_delta)

    print("Output After Training")
    print(l1)  
