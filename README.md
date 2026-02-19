# 🚢 Titanic Dataset — Exploratory Data Analysis (EDA)

**Data Analyst Internship — Task 5**

## 📌 Objective
Perform exploratory data analysis on the Titanic dataset to extract meaningful insights using visual and statistical techniques.

---

## 🗂️ Repository Structure
```
├── train.csv                  # Training dataset (891 rows × 12 columns)
├── test.csv                   # Test dataset
├── gender_submission.csv      # Sample submission file
├── eda_titanic.py             # Python script — generates all 13 plots
├── titanic_eda.ipynb          # Jupyter Notebook — full EDA with code + observations
├── titanic_eda_report.pdf     # PDF report — charts, findings
└── README.md                  # Project documentation
```

---

## 🛠️ Tools & Libraries

| Tool | Purpose |
|------|---------|
| Python 3.x | Core language |
| Pandas | Data loading, cleaning, aggregation |
| NumPy | Numerical operations |
| Matplotlib | Base plotting |
| Seaborn | Statistical visualizations |
| ReportLab | PDF report generation |

---

## 📊 EDA Coverage

### 1. Dataset Overview
- 891 rows, 12 columns
- Key missing values: **Cabin (77.1%)**, **Age (19.9%)**, Embarked (2)

### 2. Univariate Analysis
- Age distribution (approx. normal, mean = 29.7 years)
- Fare distribution (heavily right-skewed — log transform recommended)
- Survival count (61.6% did not survive, 38.4% survived)

### 3. Bivariate Analysis
- Survival by Gender: Female **74.2%** vs Male **18.9%**
- Survival by Class: 1st **63.0%** → 2nd **47.3%** → 3rd **24.2%**
- Age vs Survival (boxplot)
- Fare vs Survival (boxplot — survivors paid ~£52 vs £22 median)
- Survival by Embarkation Port

### 4. Multivariate Analysis
- Survival Rate by Pclass × Gender (grouped bar chart)
- Age distribution by Passenger Class (violin plot)

### 5. Correlation Heatmap
- Fare ↔ Survived: **+0.26**
- Pclass ↔ Survived: **−0.34**
- Pclass ↔ Fare: **−0.55**

### 6. Pairplot
- Pairwise relationships with Survived as hue

---

## 🔍 Key Findings

1. **Gender is the strongest predictor** — Female survival: 74.2% vs Male: 18.9%
2. **Passenger class strongly affects survival** — 1st class had 2.6× higher survival than 3rd class
3. **Fare correlates with survival** — Survivors paid significantly higher median fares
4. **Sex × Pclass interaction is most powerful** — Female 1st class ~97%; Male 3rd class ~14%
5. **Age has mild effect** — Children had higher survival; overall age separation is weak
6. **Fare is right-skewed** — `np.log1p()` transformation recommended before modelling
7. **Cabin has 77.1% missing data** — Should be dropped or converted to a binary `has_cabin` feature
8. **No severe multicollinearity** — Strongest correlation: Pclass vs Fare (−0.55)
9. **Moderate class imbalance** — Use stratified train/test splits and class weights in ML models

---

## ▶️ How to Run

### Run the Python script (generates all plots):
```bash
pip install pandas matplotlib seaborn reportlab
python eda_titanic.py
```

### Open the Jupyter Notebook:
```bash
pip install jupyter
jupyter notebook titanic_eda.ipynb
```

## 📁 Dataset Source
[Titanic - Machine Learning from Disaster | Kaggle](https://www.kaggle.com/c/titanic/data)
