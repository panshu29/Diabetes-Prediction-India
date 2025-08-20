# Data Preprocessing Stage
import pandas as pd
import yaml
import os

#Read paramters from params.yaml

def preprocess(inputPath,outputPath):
    print("...Data Collection (step1)... ")
    data=pd.read_csv(inputPath)
    os.makedirs(os.path.dirname(outputPath),exist_ok=True)
    
    #All following preprocessing logic goes here...
    print("...Data Cleaning (step2)... ")
    print("...Data Transformation (step3)... ")
    print("...Data Integration(step4)... ")
    print("...Data Reduction (step5)... ")
    print(".. Data Splitting (step6)... ")
    
    #Finally write the processed file
    data.to_csv(outputPath,header=None,index=False)
    print(f"Processed File is in {outputPath}")



if __name__ == '__main__':
    print(f"** Data preprocessing started ! **")
    params=yaml.safe_load(open('../params.yaml'))
    inputPath= "../"+params['preprocess']['input']
    outputPath= "../"+params['preprocess']['output']
    preprocess(inputPath,outputPath)
    print("** Data Preprocessing is done ** ")


