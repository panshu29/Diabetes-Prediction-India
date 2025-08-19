# Data Preprocessing Stage
import pandas as pd
import sys
import yaml
import os

#Read paramters from params.yaml

def preprocess(inputPath,outputPath):
    data=pd.read_csv(inputPath)
    os.makedirs(os.path.dirname(outputPath),exist_ok=True)
    #All preprocessing logic ...

    #Finally write the processed file
    data.to_csv(outputPath,header=None,index=False)
    print(f"Processed File is in {outputPath}")


if __name__ == '__main__':
    params=yaml.safe_load(open('../params.yaml'))
    inputPath= "../"+params['preprocess']['input']
    outputPath= "../"+params['preprocess']['output']
    preprocess(inputPath,outputPath)


