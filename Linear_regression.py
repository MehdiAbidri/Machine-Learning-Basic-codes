# =============================================================================
# Linear regression code 
# 1 feature and 100 rows
# =============================================================================
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.datasets import make_regression

# Dataset
x,y = make_regression(n_samples=100,n_features=1,noise=10)
y=y.reshape(y.shape[0],1)

m=len(y)
n=x.shape[1]
bias = np.ones((m,1))
X= np.hstack((x,bias))
theta=np.random.randn(n+1,1)

# Model :  
def model(X,theta):
    return X.dot(theta)

# Cost function:
def cost(X,y,theta):
    z=len(y)
    return (1/(2*z))*np.sum((model(X,theta)-y)**2)

# Gradient :
def grad(X,y,theta):
    z=len(y)
    return (1/z)*X.T.dot(model(X, theta)-y)

# Gradient descent:
def grad_descent(X,y,theta,lrate,n_iteration):
    cost_history = np.zeros((n_iteration,1))
    for i in range(0,n_iteration):
        theta = theta - lrate*grad(X, y, theta)
        cost_history[i]=cost(X, y, theta) 
    return theta,cost_history

# Training : 
lrate = 0.01
iteration = 400
theta_final,cost_h = grad_descent(X,y,theta,lrate,iteration)
prediction = model(X,theta_final)


# Plot model and cost 
fig, ((ax1, ax2)) = plt.subplots(2, 1)
fig.suptitle('Result of linear regression example')
ax1.scatter(x,y)
ax1.plot(x,prediction,c='r')
ax2.plot(range(iteration),cost_h)
ax1.set_title('Model trained vs dataset')
ax2.set_title('Cost ')




