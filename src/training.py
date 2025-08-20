import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import yaml
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from mlflow.models import infer_signature 
import os 
from sklearn.model_selection import train_test_split, GridSearchCV
from urllib.parse import urlparse
import mlflow

#load parameters
params= yaml.safe_load(open('../params.yaml'))['train']

#Create a GridSearchCV object that will take the Random Forest model (rf) 
# and test it with different parameter combinations from param_grid
def hyperparamter_tuning(X_train , y_train, param_grid):
    rf=RandomForestClassifier()
    grid_searchCV= GridSearchCV(estimator=rf,param_grid=param_grid,cv=3,n_jobs=-1,verbose=2)
    grid_searchCV.fit(X_train,y_train)
    return grid_searchCV


#X_train and y_train to teach the model during training 
#X_test and y_test Testing phase (after training, before deployment)
#I compare y_pred with y_test (true answers) to see how well it performs.
#This is where I calculate metrics like accuracy, confusion matrix, classification report, etc.
def split_datasets():
    df=pd.read_csv(params['data_path'])
    X= df.iloc[:, :-1] #input
    y= df.iloc[:,-1:] #output
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20)
    return X_train,X_test,y_train,y_test

def defineParamGrid():
     param_grid ={
         'n_estimators':[100,200],
         'max_depth':[5,10,None],
         'min_samples_split':[2,5],
         'min_samples_leaf':[1,2]
     }
     return param_grid

# Start Training 
def train():
   mlflow.set_tracking_uri('')
   X_train,X_test,y_train,y_test= split_datasets()
   signature=infer_signature(X_train,y_train)
   param_grid=defineParamGrid()
   # start hyperparamter tuning 
   grid_serach=hyperparamter_tuning(X_train,y_train,param_grid)
   best_model=grid_serach.best_estimator_
   return best_model, X_test, y_test

# Start Testing 
def test():
   best_model,X_test,y_test=train()
   y_pred=best_model.predict(X_test) # get the prediction 
   accuracy_score=accuracy_score(y_test,y_pred) # compare prediction with the actual
   print(f'Accuracy Score:{accuracy_score}')
   
  

