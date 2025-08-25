# Diabetes-Prediction-Pima-Indians  [Work in progress!....]

This data science project focuses on predicting the likelihood of diabetes in individuals.  

It also demonstrates building an end-to-end Machine Learning pipeline by training a **Random Forest Classifier (RFC)** on the **Pima Indians Diabetes Dataset**.  

**Population Context**: The dataset was collected from the **Pima Indians**, a Native American tribe in Arizona, USA, who have one of the world’s highest rates of type 2 diabetes.

I intentionally simplified this project to make it easier for beginners to understand.

# Stage 1- Data Preprocessing  

Data preprocessing is a crucial step in AI/ML pipelines because raw data is rarely ready for direct use in model training. Other than Step 1, the remaining 6 steps are optional.
The **Data Preprocessing Stage** ensures that data is:
- Clean
- Consistent
- Structured 



## 1. Data Collection
- Gather raw data from multiple sources (databases, APIs, sensors, web scraping, files, etc.).
- Includes:
  - **Structured** → tables, CSVs
  - **Semi-structured** → JSON, XML
  - **Unstructured** → text, images, video



## 2. Data Cleaning
- Handle **missing values** (imputation or removal).
- Remove **duplicates**.
- Reduce **noise** (filter irrelevant entries).
- Correct **errors** (typos, wrong labels, inconsistent values).



## 3. Data Transformation
- **Normalization/Standardization** → scale features (Min-Max, Z-score).
- **Encoding categorical variables** → One-hot encoding, label encoding.
- **Feature extraction** → derive new features (e.g., "day of week" from timestamps).
- **Feature engineering** → combine or transform features to improve model performance.



## 4. Data Integration
- Merge datasets from different sources.
- Align schemas and remove inconsistencies.



## 5. Data Reduction
- **Dimensionality reduction** → PCA, t-SNE, UMAP.
- **Feature selection** → keep only relevant features.
- **Sampling** → handle class imbalance (oversampling/undersampling).



## 6. Data Splitting
- Divide dataset into:
  - **Training set**
  - **Validation set**
  - **Test set**
- Prevents data leakage and ensures fair evaluation.



## 7. Other Steps
- Remove headers in final file
- Dont add index number


## ✨ Note 
-   Many ML libraries (like scikit-learn, TensorFlow, PyTorch) 
    ssume that the input file only contains raw feature values without column names
    By removing headers, you ensure the file format is uniform and directly consumable by the ML pipeline.
- Dont add index . This avoids an extra column that could confuse the model.

This makes it easier for AI/ML models to learn patterns effectively.  
👉 Poor preprocessing = poor predictions, no matter how advanced the model is.
 
 Logic mentioned in **./src/preprocess.py**

---

# Stage-2 Model Training and Experiment Tracking 

Specifically, it includes:

- **Data Splitting** – Dividing the dataset into training and testing sets.

- **Hyperparameter Tuning** – Using `GridSearchCV` to find the best parameters for the Random Forest model.

- **Model Training** – Fitting the best model on the training data.

- **Model Evaluation** – Calculating metrics like accuracy, confusion matrix, and classification report.

- **Experiment Tracking** – Logging parameters, metrics, and artifacts with MLflow.

- **Model Persistence** – Saving the trained model locally for future inference.


using DVC to tract Datamodel, model versions 

The dvc stage add command creates a stage in your dvc.yaml pipeline.
Here’s what each flag means in your command:

-n preprocess → stage name = preprocess

-p preprocess.input,preprocess.output → track parameters input and output from params.yaml (under the preprocess section)

-d src/preprocess.py → dependency = script

-d data/raw/diabetes.csv → dependency = raw data

-o data/processed/diabetes.csv → output = processed data

python src/preprocess.py → command to execute


A new stage called preprocess gets added (or updated if it already exists) in your project’s dvc.yaml.

That stage will include:

The command to run (python src/preprocess.py)

Dependencies (src/preprocess.py, data/raw/diabetes.csv)

Outputs (data/processed/diabetes.csv)

Parameters (preprocess.input, preprocess.output from params.yaml)

So after running it, you’ll see a new section inside dvc.yaml.
DVC starts tracking the declared dependencies (-d) and outputs (-o).

If any of those dependencies change (file content, params, or code), DVC will know that the stage needs to re-run.

Important: dvc stage add does not execute python src/preprocess.py right away.

It just records the stage definition in dvc.yaml.

preprocessing---> Traing --> Evaluation
1. MLFlow Experimems
2. DVC data versioning
