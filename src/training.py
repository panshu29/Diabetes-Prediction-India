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

params= yaml.safe_load(open('../params.yaml'))['train']


os.environ['MLFLOW_TRACKING_URI']=params["mlflow"]["tracking_uri"]
os.environ['MLFLOW_TRACKING_USERNAME']=params["mlflow"]["tracking_username"]
os.environ['MLFLOW_TRACKING_PASSWORD']=params["mlflow"]["tracking_password"]

def hyperparamter_tuning(X_train , y_train, param_grid):
    rf=RandomForestClassifier()
    grid_searchCV= GridSearchCV(estimator=rf,param_grid=param_grid,cv=3,n_jobs=-1,verbose=2)
    grid_searchCV.fit(X_train,y_train)
    return grid_searchCV

# Split the dataset 
def split_datasets():
    df=pd.read_csv(params['data_path'])
    X= df.iloc[:, :-1] #input 2D
    y= df.iloc[:,-1] #output 1D
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

# Start Training using hyperparamter tuning 
def train_tuning(X_train,y_train,param_grid):
   # start hyperparamter tuning 
   grid_serach=hyperparamter_tuning(X_train,y_train,param_grid)
   best_model=grid_serach.best_estimator_
   best_params=grid_serach.best_params_
   return best_model, best_params

# Start evaluate that model
def evaluate_model(best_model,X_test,y_test):
   y_pred=best_model.predict(X_test) # get the prediction 
   acc=accuracy_score(y_test,y_pred) # compare prediction with the actual
    # Log the Confusion Metrics
   cm= confusion_matrix(y_test,y_pred)
   cr= classification_report(y_test,y_pred)
   return acc,cm ,cr

# Start Training and tracking using mlflow 
def start_traning_tracking():
    
    X_train,X_test,y_train,y_test= split_datasets()

    # print(X_train.shape)
    # print(y_train.shape)
    # print(X_test)
    # print(y_test)

    param_grid=defineParamGrid()
    signature=infer_signature(X_train,y_train)
    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    best_model,best_params=train_tuning(X_train,y_train,param_grid)
    acc,cm,cr=evaluate_model(best_model,X_test,y_test)
    print(f'Accuracy Score:{acc}')

    #Log Metrics 
    mlflow.log_metric("accuracy",acc)

    #Log Params
    mlflow.log_param("best_n_estimators",best_params["n_estimators"])
    mlflow.log_param("best_max_depth",best_params["max_depth"])
    mlflow.log_param("best_min_samples_split",best_params["min_samples_split"])
    mlflow.log_param("best_min_samples_leaf",best_params["min_samples_leaf"])

    # Log text
    mlflow.log_text(str(cm),"Confusion_matrix.txt")
    mlflow.log_text(str(cr),"classification_report.txt")

    # Log Model 
    track_url_type=urlparse(mlflow.get_tracking_uri()).scheme
    if(track_url_type!='file'): #for a remote server and auto infers i/o signature 
        mlflow.sklearn.log_model(best_model,'model', registered_model_name="Best Model",signature=signature)
    else: #for local file-based logging ./mlruns/<experiment_id>/<run_id>/artifacts/model/
        mlflow.sklearn.log_model(best_model,'model',signature=signature)
        
    # Save the trained machine learning model(serializes that model object)
    model_path=params["local_model_path"]
    os.makedirs(os.path.dirname(model_path),exist_ok=True)
    pickle.dump(best_model,open(model_path,'wb')) #write after converts it into a byte stream
    print(f'Model Saved {model_path}')


if __name__=="__main__":
    #load all training parameters
    start_traning_tracking()