# =============================================================================
#  Non linear regression code 
#  1 feature and 100 rows
# =============================================================================
import numpy as np
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

# =============================================================================
# Dataset 
# =============================================================================


x,y = make_regression(n_samples=100,n_features=1,noise=10)
y=y + abs(y/2)+y**2 # in order to make the relation non linear
y=y.reshape(y.shape[0],1)
plt.scatter(x,y)
  

# =============================================================================
# Matrix X (feature)    
# =============================================================================
X=np.hstack((x,np.ones(x.shape)))
X = np.hstack((x**2,X))

theta = np.random.randn(3,1)

# =============================================================================
# Model
# =============================================================================

def model (X, theta): 
    return X.dot(theta)

plt.scatter(x,y)
plt.plot(x,model(X,theta))

# =============================================================================
# FoCost function
# =============================================================================
def cost_function(x,y,theta):
    m = len(y)
    return 1/(2*m) * np.sum((model(X,theta)-y)**2)

# =============================================================================
# Gradient descent 
# =============================================================================

def grad(X,y,theta):
    m=len(y)
    return 1/m * X.T.dot(model(X,theta)-y)

def gradient_descent (x,y,theta,alpha,n_iteration):
    cost_history = np.zeros(n_iteration)
    for i in range(0,n_iteration):
        theta = theta - alpha * grad(X,y,theta)
        cost_history[i]= cost_function(x, y, theta)
    return theta , cost_history


# =============================================================================
# Training
# =============================================================================
iteration = 1000
lrate = 0.01
theta_final,cost_history =gradient_descent(x,y,theta,lrate,iteration)

prediction = model(X,theta_final)
# =============================================================================
# Coefficient of determination 
# =============================================================================

def coef_det(y,pred):
    u=((y-pred)**2).sum() 
    v=((y-y.mean())**2).sum()
    return 1 - u/v 

# =============================================================================
# Learning curve
# =============================================================================

fig, ((ax1, ax2)) = plt.subplots(2, 1)
fig.suptitle('Result of non linear regression example')
ax1.scatter(x,y)
ax1.scatter(x,prediction,c='r')
ax2.plot(range(iteration),cost_history)
ax1.set_title('Model trained vs dataset')
ax2.set_title('Cost ')

print( 'the coefficient of determination of this model is : ', coef_det(y,prediction))






