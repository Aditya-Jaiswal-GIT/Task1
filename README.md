# 🚢 Task 1 – Data Cleaning & Preprocessing

## 📌 Objective
To learn and apply data preprocessing techniques on the Titanic dataset, preparing it for machine learning tasks. This involves handling missing values, encoding categorical variables, scaling features, and detecting/removing outliers — all using basic Python libraries.

---

## 📁 Dataset
- *Dataset Name*: Titanic Dataset
- *Source*: [Kaggle Titanic Dataset](https://www.kaggle.com/datasets/yasserh/titanic-dataset)
- *Used File*: titanic.csv

---

## 🧰 Tools & Libraries Used
- pandas – Data manipulation
- numpy – Numerical computations
- matplotlib.pyplot – Data visualization (boxplots)

---

## ⚙ Steps Performed

### 1. Data Loading & Exploration
- Loaded dataset using pandas.read_csv()
- Checked column types, missing values, and summary statistics

### 2. Missing Values Handling
- Filled missing *Age* values with the *median*
- Filled missing *Embarked* values with the *mode*
- Dropped *Cabin* column due to too many missing values
- Dropped *Ticket* and *Name* as they are not helpful for basic ML

### 3. Encoding Categorical Variables
- Converted *Sex* to 0 (male) and 1 (female)
- Converted *Embarked*: S → 0, C → 1, Q → 2

### 4. Feature Scaling
- Applied *Min-Max Normalization* on Age and Fare to bring values between 0 and 1

### 5. Outlier Detection & Removal
- Plotted *boxplots* for Age and Fare using matplotlib
- Removed outliers from Fare using the *IQR method*

---

## 📦 Output
- Final cleaned dataset: titanic_cleaned.csv
- All features are numerical and ready for model training

---

## 📷 Visual Example
Boxplots for Age and Fare were created to visualize outliers.

---

## 📌 Notes
- No use of sklearn or seaborn
- Purely based on foundational data science with pandas, numpy, and matplotlib

---

## ✅ Final Dataset Overview
- No missing values
- All features are numerical
- Cleaned and ready for ML algorithms

---

## 📤 Submission
- Upload the code, dataset, and this README.md to your GitHub repository.
- Submit the GitHub link in the provided internship submission form.
