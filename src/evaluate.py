#Evaluate how much accuracy with new data 

import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
import yaml
import os
import mlflow 

params= yaml.safe_load(open('../params.yaml'))['train']

os.environ['MLFLOW_TRACKING_URI']=params["mlflow"]["tracking_uri"]
os.environ['MLFLOW_TRACKING_USERNAME']=params["mlflow"]["tracking_username"]
os.environ['MLFLOW_TRACKING_PASSWORD']=params["mlflow"]["tracking_password"]

def evaluate( data_path , model_path):
     data= pd.read_csv(data_path)
     X= data.iloc[:, :-1] #input 2D
     y= data.iloc[:,-1] #output 1D
     mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    #lload the mode from local
     model= pickle.load(open(model_path,'rb'))
     predictions=model.predict(X)
     accuracyScore=accuracy_score(y,predictions)

     #log metrics to mlflow
     mlflow.log_metric("accuracy",accuracyScore)
     print("model accuracy:",accuracyScore)


if __name__=="__main__":
     evaluate(params["data_path"],params["local_model_path"])
     
