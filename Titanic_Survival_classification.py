# =============================================================================
# Import the modules we need 
# =============================================================================
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier # since it's a classification data
# =============================================================================
# Import gross data 
# =============================================================================

data_gross = pd.read_excel('titanic_data_brut.xlsx') # import data


# =============================================================================
# organize the data 
# =============================================================================

data = data_gross[['survived','sex','age','pclass']] # remove useless columns
data = data.dropna(axis=0)                     # remove rows which contain nan values
data.sex = [1 if x =='female' else 0 for x in data.sex  ] # replace sex info by boolean value
data.head() # to see what it looks like 

# =============================================================================
# Create the target variable and the features 
# =============================================================================

y = data['survived']  # target variable 
X = data.drop('survived', axis =1) # Features matrix 

# =============================================================================
# Create a model, train it and evaluate it 
# =============================================================================
 
 
model = KNeighborsClassifier(n_neighbors=3) # create the model (best score with 3)
model.fit(X,y)                 # fit the model
model.score(X,y)               # evaluate it 
    
# =============================================================================
# Use the model  
# =============================================================================

def survival_func(model,sex,age,pclass,features_nbr):
    """ takes the model trained, the sex,age and the class of the passenger
    """
    x=np.array([sex,age,pclass]).reshape(1,features_nbr)
    message_d = 'you would die if you were on the Titanic ' 
    message_s = 'you would survive  if you were on the Titanic ' 
    
    result = model.predict(x)
    resultproba = model.predict_proba(x)
    """ calculate the prediction with the probability 
    """
    message_proba =  resultproba[0,0] if result ==0 else resultproba[0,1] 
    message = message_d if result == 0 else message_s 
    print(message, ' \n with :', message_proba*100, '%')
    
    """ print the result and the probabilty 
    """
    return result,resultproba
        
result, proba = survival_func(model, sex=1, age=2, pclass=3,features_nbr=3)

