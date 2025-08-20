# Diabetes-Prediction-Pima-Indians  

This data science project focuses on predicting the likelihood of diabetes in individuals.  

It also demonstrates building an end-to-end Machine Learning pipeline by training a **Random Forest Classifier (RFC)** on the **Pima Indians Diabetes Dataset**.  

**Population Context**: The dataset was collected from the **Pima Indians**, a Native American tribe in Arizona, USA, who have one of the world’s highest rates of type 2 diabetes.


# Data Preprocessing Stage ( 5 steps)

Data preprocessing is a crucial step in AI/ML pipelines because raw data is rarely ready for direct use in model training.  

---

## 1. Data Collection
- Gather raw data from multiple sources (databases, APIs, sensors, web scraping, files, etc.).
- Includes:
  - **Structured** → tables, CSVs
  - **Semi-structured** → JSON, XML
  - **Unstructured** → text, images, video

---

## 2. Data Cleaning
- Handle **missing values** (imputation or removal).
- Remove **duplicates**.
- Reduce **noise** (filter irrelevant entries).
- Correct **errors** (typos, wrong labels, inconsistent values).

---

## 3. Data Transformation
- **Normalization/Standardization** → scale features (Min-Max, Z-score).
- **Encoding categorical variables** → One-hot encoding, label encoding.
- **Feature extraction** → derive new features (e.g., "day of week" from timestamps).
- **Feature engineering** → combine or transform features to improve model performance.

---

## 4. Data Integration
- Merge datasets from different sources.
- Align schemas and remove inconsistencies.

---

## 5. Data Reduction
- **Dimensionality reduction** → PCA, t-SNE, UMAP.
- **Feature selection** → keep only relevant features.
- **Sampling** → handle class imbalance (oversampling/undersampling).

---

## 6. Data Splitting
- Divide dataset into:
  - **Training set**
  - **Validation set**
  - **Test set**
- Prevents data leakage and ensures fair evaluation.

---

## ✨ Summary
The **Data Preprocessing Stage** ensures that data is:
- Clean
- Consistent
- Structured  

This makes it easier for AI/ML models to learn patterns effectively.  
👉 Poor preprocessing = poor predictions, no matter how advanced the model is.
 
 Logic mentioned in **./src/preprocess.py**


