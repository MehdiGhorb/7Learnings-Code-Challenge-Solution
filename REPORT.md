# Technical Report: Snowfall Prediction Using BigQuery and Machine Learning

## Objective
The goal of this project was to predict whether it would snow 20 years ago tomorrow using historical weather data. The dataset contains climate information from over 9000 stations worldwide, and the tasks involved data preprocessing, exploratory data analysis (EDA), feature engineering, and machine learning model training and evaluation.

---

## Data Extraction and Preprocessing

### Data Extraction
The dataset was queried from Google BigQuery's public dataset `bigquery-public-data.samples.gsod`. The following steps were performed:
1. **Date Formatting**: The `year`, `month`, and `day` columns were combined into a single `date` column in the `YYYY-MM-DD` format.
2. **Filtering**: Data was filtered to include only records from 2000 to 2005 for station numbers between 725300 and 726300.

### Missing Value Handling
1. **Dropping Columns**: Columns with more than 80% missing values were dropped.
2. **Imputation**:
   - For numerical columns:
     - Mean imputation was used for normally distributed data (low skewness).
     - Median imputation was used for skewed data.
   - For categorical columns:
     - Mode imputation (most frequent value) was applied.

### Duplicate Removal
Duplicate rows were checked and removed if present.

### Feature Engineering
1. **One-Hot Encoding**: The `wban_number` column, representing station locations, was one-hot encoded to make it suitable for machine learning models.
2. **Column Removal**: Irrelevant columns such as `station_number` (an ID column) were dropped.

---

## Exploratory Data Analysis (EDA)

### Target Variable Analysis
The target variable `snow` was highly imbalanced:
- 85.59% of the records indicated no snow (`False`).
- 14.41% of the records indicated snow (`True`).

### Balancing the Dataset
To address the imbalance, undersampling was applied to the majority class (`False`) to create a balanced dataset with a 50:50 ratio.

### Outlier Detection
Outliers in numerical columns were identified using the Interquartile Range (IQR) method. Columns with heavy-tailed distributions (high kurtosis) and skewed distributions were flagged for further analysis.

### Data Transformation
1. **Quantile Transformation**: Applied to normalize skewed distributions.
2. **Box-Cox Transformation**: Used to stabilize variance and make data more normally distributed.

### Downsampling
For columns with highly frequent values, downsampling was applied to reduce the dominance of these values in the dataset.

### Scaling
Numerical features were standardized using `StandardScaler` to ensure all features had comparable magnitudes.

---

## Data Splitting
The dataset was split into training, evaluation, and test sets:
- **Test Set**: Data after May 13, 2005, was used as the test set.
- **Training and Evaluation Sets**: The remaining data was split into training (90%) and evaluation (10%) sets.

---

## Machine Learning Models

### XGBoost Classifier
1. **Training**: The XGBoost classifier was trained on the training set with the following steps:
   - Columns with potential data leakage (`rain`, `fog`, `hail`, `thunder`, `tornado`) were excluded.
   - The `date` and `snow` columns were also excluded from the features.
2. **Evaluation**:
   - Predictions were made for the test set.
   - Metrics such as accuracy, precision, recall, and F1-score were calculated.
   - A confusion matrix was plotted to visualize the model's performance.

### Hyperparameter Tuning
GridSearchCV was used to optimize the hyperparameters of the XGBoost model. The parameter grid included:
- `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `gamma`, and `min_child_weight`.

Due to computational constraints, the improvement in accuracy after hyperparameter tuning was minimal.

---

### Random Forest Classifier
1. **Training**: A Random Forest classifier was trained as an alternative model.
2. **Evaluation**:
   - Predictions were made for the test set.
   - Metrics such as accuracy, precision, recall, and F1-score were calculated.
   - A confusion matrix was plotted to compare the performance with the XGBoost model.

---

## Results and Observations
1. **XGBoost**:
   - Achieved reasonable accuracy and balanced performance across precision, recall, and F1-score.
   - Hyperparameter tuning did not significantly improve performance due to limited computational resources.

2. **Random Forest**:
   - Provided comparable results to XGBoost.
   - Simpler to train but slightly less interpretable than XGBoost.

3. **Feature Importance**:
   - Both models highlighted the importance of specific weather features in predicting snowfall.

---

## Conclusion
The project demonstrated the end-to-end process of data extraction, preprocessing, EDA, and machine learning model training. Both XGBoost and Random Forest classifiers were effective in predicting snowfall, with XGBoost showing slightly better performance. Future work could involve testing additional models, exploring advanced feature engineering techniques, and leveraging more computational resources for hyperparameter tuning.