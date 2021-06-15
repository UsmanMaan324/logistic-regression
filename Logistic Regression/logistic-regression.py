
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

alpha = 0.001

# Function to make list of our desire size


def helper_function( size):
    h_list = []
    for counter in range(size):
        h_list.append(1)
    return h_list


# Function to find the cost

def cost_function(hOfX, lable, normalF):
    lable = np.array([lable])
    normalF = np.array([normalF])
    hOfx1 = np.log(hOfX)
    hOfX2 = np.subtract(normalF, hOfX)
    hOfX2 = np.log(hOfX2)
    lable1 = np.subtract(normalF, lable)
    lable1 = np.transpose(lable1)
    hOfX2 = np.dot(hOfX2, lable1)
    lable2 = - lable
    lable2 = np.transpose(lable2)
    hOfx1 = np.dot(hOfx1, lable2)
    hOfX = np.subtract(hOfx1, hOfX2)
    cost = np.sum(hOfX)
    cost = cost/lable.size
    return cost

def gradient_descent(hOfX, lable, xData):
    lable = np.array(lable)
    hOfX = np.subtract(hOfX, lable)
    xData = np.transpose(xData)
    hOfX = np.dot(hOfX,xData)
    updatedTheetas = hOfX/lable.size
    return updatedTheetas


def hOfX_Finder(gOfx):
    return_hOfX = []
    for z in range(gOfx.size):
        return_hOfX.append(1 / (1 + np.exp(-gOfx[0][z])))
    return return_hOfX


# Get data form file
data = pd.read_csv('ex2data1.txt', sep=",", header=None)
print(data)
# Making list of data size which contain 1's
normalF = []
for i in range(data[0].size):
    normalF.append(1)
# Add list of 1's in feature Matrix
featureMatrix = np.array([normalF, data[0], data[1]])
# Making list of theetas
theetasVector = helper_function(data.columns.size)
# Vectorize the list of theetas
theetasVector = np.array([theetasVector])

cost = 100
while cost > 20:
    # Find the HOf(x) with the help of dot product
    gOfZ = np.dot(theetasVector, featureMatrix)
    hOfX = hOfX_Finder(gOfZ)

    cost = cost_function(hOfX, data[2], normalF)

    updatedTheetas = []
    updatedTheetas.append(gradient_descent(hOfX,data[2],normalF))
    updatedTheetas.append(gradient_descent(hOfX,data[2],data[0]))
    updatedTheetas.append(gradient_descent(hOfX,data[2],data[1]))
    updatedTheetas = np.array(updatedTheetas)
    updatedTheetas = np.dot(alpha, updatedTheetas)
    updatedTheetas = np.transpose(updatedTheetas)
    theetasVector = np.subtract(theetasVector, updatedTheetas)

print(theetasVector)
fig, ax = plt.subplots()
colors = {0:'red', 1:'green'}
ax.scatter(data[0], data[1], c=data[2].map(colors))
#plt.show()