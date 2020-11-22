import numpy as np
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

# =============================================================================
# Dataset 
# =============================================================================

np.random.seed()
x,y = make_regression(n_samples=100,n_features=1,noise=10)
y=y + abs(y/2)
plt.scatter(x,y)

# vérification des dimension du datset : 
print(x.shape)
y=y.reshape(y.shape[0],1)
print(y.shape)    

#Matrice X : 
    
X=np.hstack((x,np.ones(x.shape)))
X = np.hstack((x**2,X))

theta = np.random.randn(3,1)

# =============================================================================
# Modèle
# =============================================================================

def model (X, theta): 
    return X.dot(theta)

plt.scatter(x,y)
plt.plot(x,model(X,theta))

# =============================================================================
# Fonction cout 
# =============================================================================
def cost_function(x,y,theta):
    m = len(y)
    return 1/(2*m) * np.sum((model(X,theta)-y)**2)

# =============================================================================
# descente de gradient 
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


theta_final,cost_history =gradient_descent(x,y,theta,alpha=0.01,n_iteration=1000)

prediction = model(X,theta_final)
plt.scatter(x,y)
plt.scatter(x,prediction,c='r')

# =============================================================================
# Courbe d'apprentissage 
# =============================================================================

plt.plot(range(1000),cost_history)

# =============================================================================
# Coefficient de determination 
# =============================================================================

def coef_det(y,pred):
    u=((y-pred)**2).sum() 
    v=((y-y.mean())**2).sum()
    return 1 - u/v 


print( coef_det(y,prediction))

