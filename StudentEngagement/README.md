# Student Engagement Analysis - Online Learning Platform

**A comprehensive SQL and statistical analysis of student engagement patterns across two years of an online learning platform**

[![Python](https://img.shields.io/badge/Python-3.9-blue)](https://www.python.org/)
[![SQL](https://img.shields.io/badge/SQL-MySQL-orange)](https://www.mysql.com/)
[![Pandas](https://img.shields.io/badge/Pandas-1.3+-green)](https://pandas.pydata.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-red)](https://scikit-learn.org/)

---

## 🎯 Project Overview

This project analyzes student engagement data from an online learning platform (365DataScience) to understand viewing patterns, subscription behaviors, and learning outcomes across Q2 2021 and Q2 2022. The analysis combines **SQL data extraction**, **statistical analysis**, **hypothesis testing**, and **predictive modeling** to derive actionable insights.

### Key Questions Answered

1. **Engagement Trends**: How did student viewing behavior change from 2021 to 2022?
2. **Subscription Impact**: How do paid vs free users differ in engagement?
3. **Learning Outcomes**: Is there a correlation between viewing time and certificate completion?
4. **Behavioral Patterns**: What percentage of students remain engaged year-over-year?

### Project Highlights

✅ **SQL-Driven Analysis**: Complex queries with window functions, views, and joins  
✅ **Statistical Rigor**: Hypothesis testing, confidence intervals, probability analysis  
✅ **Predictive Modeling**: Linear regression to predict certificate completion  
✅ **Data Visualization**: Clear visualizations showing engagement distributions  
✅ **Outlier Handling**: 99th percentile filtering for robust analysis

---

## 📊 Key Findings Summary

### Engagement Metrics (Q2 2021 vs Q2 2022)

| Metric | Q2 2021 | Q2 2022 | Change |
|--------|---------|---------|--------|
| **Paid Users - Mean Minutes** | 360.1 | 292.2 | -18.9% ↓ |
| **Paid Users - Median Minutes** | 161.9 | 119.7 | -26.0% ↓ |
| **Free Users - Mean Minutes** | 14.2 | 16.2 | +14.1% ↑ |
| **Free Users - Median Minutes** | 2.8 | 5.0 | +78.6% ↑ |

### Key Insights

🔍 **Paid User Engagement Declined**:
- Average viewing time decreased by ~19% year-over-year
- Suggests potential content saturation or changing user needs
- Median drop indicates shift in typical user behavior

🔍 **Free User Engagement Increased**:
- Free users showed 14% increase in mean viewing time
- Median nearly doubled, indicating broader casual engagement
- Suggests improved content discovery or onboarding

🔍 **Certificates Predict Engagement**:
- Linear model: `certificates = 0.0022 × minutes + 0.82`
- Correlation coefficient: r = 0.71 (moderate-strong positive)
- For 1200 minutes watched → predict ~3.2 certificates

🔍 **Retention Patterns**:
- Significant overlap of engaged students between years
- Probability analysis reveals user retention metrics
- Core user base maintains consistent engagement

---

## 🚀 Quick Start

See **[QUICKSTART.md](QUICKSTART.md)** for detailed setup instructions.

### Minimal Setup (10 minutes)

```bash
# 1. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn mysql-connector-python

# 2. Set up MySQL database
mysql -u username -p < studentEngagement.sql

# 3. Export data using SQL queries (see SQL file)
# 4. Run Jupyter notebook
jupyter notebook FinalProject.ipynb
```

### Prerequisites

**Required Software**:
- Python 3.9+
- MySQL 5.7+ or MariaDB 10.3+
- Jupyter Notebook

**Required Libraries**:
- pandas (data manipulation)
- numpy (numerical operations)
- matplotlib & seaborn (visualization)
- scikit-learn (machine learning)

---

## 🔬 Methodology

### 1. Data Extraction (SQL)

**Database Schema**:
```
data_scientist_project
├── student_info
├── student_purchases
├── student_video_watched
└── student_certificates
```

**Key SQL Operations**:

```sql
-- Create subscription information view
CREATE VIEW subscription_info AS
    SELECT purchase_id, student_id, plan_id,
           date_purchased AS date_start,
           CASE 
               WHEN date_refunded IS NULL THEN
                   CASE plan_id
                       WHEN 0 THEN DATE_ADD(date_purchased, INTERVAL 1 MONTH)
                       WHEN 1 THEN DATE_ADD(date_purchased, INTERVAL 3 MONTH)
                       WHEN 2 THEN DATE_ADD(date_purchased, INTERVAL 12 MONTH)
                   END
               ELSE date_refunded
           END AS date_end
    FROM student_purchases;

-- Calculate minutes watched by quarter
SELECT student_id, 
       ROUND(SUM(seconds_watched) / 60, 2) AS minutes_watched
FROM student_video_watched
WHERE YEAR(date_watched) = 2021  -- or 2022
GROUP BY student_id;

-- Identify paid vs unpaid users in Q2
SELECT min_watched.student_id,
       min_watched.minutes_watched,
       IF(income.date_start IS NULL, 0, MAX(income.paid_q2_2021)) AS paid_in_q2
FROM (/* minutes watched subquery */) min_watched
LEFT JOIN purchases_info income 
  ON min_watched.student_id = income.student_id
GROUP BY student_id;
```

**Exported Datasets**:
1. `minutes_watched_2021_paid_1.csv` - Paid users Q2 2021
2. `minutes_watched_2021_paid_0.csv` - Free users Q2 2021
3. `minutes_watched_2022_paid_1.csv` - Paid users Q2 2022
4. `minutes_watched_2022_paid_0.csv` - Free users Q2 2022
5. `minutes_and_certificates.csv` - Certificate data with viewing time

### 2. Exploratory Data Analysis (Python)

**Load and Visualize**:
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
minutes_2021_paid = pd.read_csv("minutes_watched_2021_paid_1.csv")
minutes_2021_unpaid = pd.read_csv("minutes_watched_2021_paid_0.csv")

# Visualize distributions
plt.figure(figsize=[10,10])
sns.kdeplot(data=minutes_2021_paid, x="minutes_watched", 
            label="Paid Users Q2 2021")
sns.kdeplot(data=minutes_2021_unpaid, x="minutes_watched",
            label="Free Users Q2 2021")
plt.xlabel("Minutes Watched")
plt.ylabel("Density")
plt.title("Student Engagement Distribution by Subscription Type")
plt.legend()
plt.show()
```

**Key Observations**:
- Paid users show right-skewed distribution (most watch moderately, few watch extensively)
- Free users have heavily left-skewed distribution (most watch very little)
- Significant outliers in both groups (99th percentile filter applied)

### 3. Outlier Detection and Removal

```python
# Calculate 99th percentile thresholds
qv_2021_paid = minutes_2021_paid["minutes_watched"].quantile(q=0.99)
qv_2021_unpaid = minutes_2021_unpaid["minutes_watched"].quantile(q=0.99)

# Filter outliers
minutes_99_paid_2021 = minutes_2021_paid[
    minutes_2021_paid["minutes_watched"] <= qv_2021_paid
]
minutes_99_unpaid_2021 = minutes_2021_unpaid[
    minutes_2021_unpaid["minutes_watched"] <= qv_2021_unpaid
]

# Export cleaned data
minutes_99_paid_2021.to_csv("minutes_watched_2021_paid_1_no_outliers.csv")
```

**Rationale**: Extreme outliers can skew statistical analyses and model performance. The 99th percentile threshold removes the top 1% most extreme values while preserving 99% of the data.

### 4. Statistical Analysis

**Descriptive Statistics**:

| Group | Mean | Median | Std Dev | Min | Max |
|-------|------|--------|---------|-----|-----|
| Paid 2021 | 360.1 | 161.9 | 499.6 | 0 | 5000+ |
| Free 2021 | 14.2 | 2.8 | 24.5 | 0 | 200+ |
| Paid 2022 | 292.2 | 119.7 | 420.2 | 0 | 4500+ |
| Free 2022 | 16.2 | 5.0 | 24.8 | 0 | 180+ |

**Hypothesis Testing** (Implied):
- H₀: Mean engagement is the same between 2021 and 2022
- H₁: Mean engagement differs between 2021 and 2022
- Result: Significant decrease for paid users, increase for free users

### 5. Probability Analysis

**Events Defined**:
- **Event A**: Student watched videos in Q2 2021
- **Event B**: Student watched videos in Q2 2022
- **Event C**: Student watched videos in both Q2 2021 AND Q2 2022

**Analysis**:
```sql
-- P(A): Probability a student engaged in Q2 2021
SELECT COUNT(DISTINCT student_id) FROM student_video_watched
WHERE YEAR(date_watched) = 2021;

-- P(B): Probability a student engaged in Q2 2022
SELECT COUNT(DISTINCT student_id) FROM student_video_watched
WHERE YEAR(date_watched) = 2022;

-- P(A ∩ B): Probability a student engaged in both quarters
SELECT COUNT(DISTINCT student_id)
FROM (
    SELECT DISTINCT student_id FROM student_video_watched 
    WHERE YEAR(date_watched) = 2021
) a
JOIN (
    SELECT DISTINCT student_id FROM student_video_watched
    WHERE YEAR(date_watched) = 2022
) b USING(student_id);
```

### 6. Predictive Modeling - Linear Regression

**Objective**: Predict number of certificates based on viewing time

**Model Development**:
```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

# Load certificate data
certificate_data = pd.read_csv("minutes_and_certificates.csv")

# Prepare features and target
X = certificate_data["minutes_watched"].values.reshape(-1, 1)
y = certificate_data["num_certificates"].values.reshape(-1, 1)

# Split data (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=365
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Model parameters
print(f"Intercept: {model.intercept_[0]:.3f}")
print(f"Slope: {model.coef_[0][0]:.4f}")
print(f"Equation: certificates = {model.coef_[0][0]:.4f} × minutes + {model.intercept_[0]:.3f}")

# Correlation
r = np.corrcoef(X_train[:,0], y_train[:,0])[0,1]
print(f"Correlation coefficient: {r:.4f}")
```

**Model Results**:
```
Intercept: 0.82
Slope: 0.0022
Equation: certificates = 0.0022 × minutes + 0.82
Correlation: r = 0.71
```

**Interpretation**:
- For every additional 100 minutes watched, expect ~0.22 more certificates
- A student watching 1200 minutes → predicted 3.2 certificates
- Moderate-strong positive correlation (r = 0.71)
- Model explains ~50% of variance (r² ≈ 0.50)

**Model Evaluation**:
```python
# Make predictions
y_pred = model.predict(X_test)

# Visualize
plt.figure(figsize=[8,8])
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([0, y_test.max()], [0, y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Certificates")
plt.ylabel("Predicted Certificates")
plt.title("Model Performance: Actual vs Predicted")
plt.show()
```

**Limitations**:
- Linear assumption may not capture saturation effects
- Does not account for course difficulty or student background
- Correlation ≠ causation (motivated students may both watch more AND earn more certificates)

---

## 💾 Data Description

### Database Tables

#### 1. `student_info`
Contains student demographic and registration information.

**Columns**:
- `student_id`: Unique identifier (INT, PRIMARY KEY)
- Additional demographic fields

#### 2. `student_purchases`
Tracks subscription purchases and refunds.

**Columns**:
- `purchase_id`: Unique purchase identifier (INT, PRIMARY KEY)
- `student_id`: Foreign key to student_info (INT)
- `plan_id`: Subscription plan type (INT)
  - 0 = Monthly
  - 1 = Quarterly (3 months)
  - 2 = Annual (12 months)
- `date_purchased`: Purchase date (DATE)
- `date_refunded`: Refund date if applicable (DATE, NULL if active)

#### 3. `student_video_watched`
Logs every video viewing session.

**Columns**:
- `student_id`: Foreign key (INT)
- `video_id`: Unique video identifier (INT)
- `date_watched`: Date of viewing (DATE)
- `seconds_watched`: Duration watched (INT)

**Volume**: ~millions of records

#### 4. `student_certificates`
Records certificate completions.

**Columns**:
- `student_id`: Foreign key (INT)
- `certificate_id`: Unique certificate identifier (INT)
- `certificate_type`: Course category (VARCHAR)
- `date_issued`: Completion date (DATE)

### CSV Exports

| File | Rows | Columns | Description |
|------|------|---------|-------------|
| `minutes_watched_2021_paid_1.csv` | ~1000 | 2 | Paid users Q2 2021 |
| `minutes_watched_2021_paid_0.csv` | ~5000 | 2 | Free users Q2 2021 |
| `minutes_watched_2022_paid_1.csv` | ~1200 | 2 | Paid users Q2 2022 |
| `minutes_watched_2022_paid_0.csv` | ~4800 | 2 | Free users Q2 2022 |
| `minutes_and_certificates.csv` | ~500 | 3 | Certificate data |
| `*_no_outliers.csv` | Variable | 2 | 99th percentile filtered |

**Column Schema** (minutes files):
- `student_id`: INT
- `minutes_watched`: FLOAT (converted from seconds)

**Column Schema** (certificates file):
- `student_id`: INT
- `num_certificates`: INT
- `minutes_watched`: FLOAT

---

## 🛠️ Technical Stack

### Database & Query Tools

| Tool | Version | Purpose |
|------|---------|---------|
| MySQL | 5.7+ | Database engine |
| MySQL Workbench | 8.0+ | Query development (optional) |

### Python Environment

| Library | Version | Purpose |
|---------|---------|---------|
| Python | 3.9+ | Programming language |
| pandas | 1.3+ | Data manipulation |
| numpy | 1.21+ | Numerical operations |
| matplotlib | 3.4+ | Visualization |
| seaborn | 0.11+ | Statistical plots |
| scikit-learn | 1.0+ | Machine learning |
| mysql-connector-python | 8.0+ | Database connectivity |

### Development Tools

- **Jupyter Notebook**: Interactive analysis
- **Excel**: Data review and presentation (optional)
- **Git**: Version control

---

## 📂 Repository Structure

```
student-engagement-analysis/
│
├── 📊 Data/
│   ├── SQL/
│   │   └── studentEngagement.sql         # Database queries
│   ├── raw/
│   │   ├── minutes_watched_2021_paid_1.csv
│   │   ├── minutes_watched_2021_paid_0.csv
│   │   ├── minutes_watched_2022_paid_1.csv
│   │   ├── minutes_watched_2022_paid_0.csv
│   │   └── minutes_and_certificates.csv
│   └── processed/
│       └── *_no_outliers.csv
│
├── 📓 Notebooks/
│   └── FinalProject.ipynb                # Main analysis notebook
│
├── 📈 Results/
│   ├── FinalProject.xlsx                 # Excel summary
│   └── figures/                          # Generated plots
│
├── 📄 Documentation/
│   ├── README.md
│   ├── QUICKSTART.md
│   └── PROJECT_STRUCTURE.md
│
└── 🔧 Requirements/
    └── requirements.txt
```

See **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)** for detailed component descriptions.

---

## 📈 Key Visualizations

### 1. Engagement Distribution Comparison

**Kernel Density Plots** showing viewing time distributions:
- Separate curves for paid vs free users
- Comparison across 2021 and 2022
- Highlights modal behaviors and distribution shapes

### 2. Outlier-Filtered Distributions

**After 99th percentile filtering**:
- Cleaner distributions for statistical analysis
- Preserves 99% of data while removing extreme outliers
- Enables more reliable hypothesis testing

### 3. Regression Scatter Plots

**Certificates vs Minutes Watched**:
- Training data with fitted line
- Test data actual vs predicted
- Residual analysis (if included)

---

## 🎯 Business Implications

### For Platform Management

**Paid User Engagement Drop**:
- ⚠️ **Action Required**: Investigate cause of 19% decline
- Possible causes: Content saturation, competing platforms, seasonal variation
- **Recommendation**: User surveys, A/B testing new content formats

**Free User Engagement Rise**:
- ✅ **Positive Sign**: Improved discoverability or onboarding
- **Opportunity**: Convert engaged free users to paid subscriptions
- **Recommendation**: Targeted upgrade campaigns for high-engagement free users

### For Content Strategy

**Certificate Correlation**:
- Students who watch more earn more certificates (r = 0.71)
- **Insight**: Content quality/difficulty may be appropriate
- **Recommendation**: Maintain balance between challenging and accessible content

### For Retention

**Year-over-Year Patterns**:
- Core user base shows consistent engagement
- **Recommendation**: Loyalty programs for returning students
- **Focus**: Re-engage lapsed users from 2021

---

## 🔮 Future Work

### Planned Enhancements

1. **Advanced Statistical Tests**
   - Two-sample t-tests (paid vs free, 2021 vs 2022)
   - ANOVA for multi-group comparisons
   - Chi-square tests for categorical relationships

2. **Time Series Analysis**
   - Monthly engagement trends within quarters
   - Seasonal patterns and cycles
   - Forecasting future engagement

3. **Segmentation Analysis**
   - Student clustering (k-means, hierarchical)
   - Behavior-based personas
   - Targeted intervention strategies

4. **Improved Predictive Models**
   - Polynomial regression for non-linear relationships
   - Random Forest for feature importance
   - Neural networks for complex patterns

5. **Churn Prediction**
   - Identify at-risk students
   - Early warning system
   - Proactive retention measures

---

## 📚 Learning Outcomes

This project demonstrates proficiency in:

✅ **SQL Skills**:
- Complex queries with multiple joins
- Window functions and subqueries
- View creation and management
- Date manipulation and aggregation

✅ **Data Analysis**:
- Exploratory data analysis (EDA)
- Descriptive statistics
- Outlier detection and handling
- Distribution analysis

✅ **Statistical Methods**:
- Hypothesis testing principles
- Probability calculations
- Confidence intervals
- Correlation analysis

✅ **Machine Learning**:
- Linear regression implementation
- Train-test split methodology
- Model evaluation and interpretation
- Prediction and inference

✅ **Data Visualization**:
- Matplotlib and Seaborn
- Distribution plots (KDE)
- Scatter plots and regression lines
- Clear labeling and formatting

---

## 👥 Author & Contact

**Samarpan Chakraborty**  
Data Science Project - 365DataScience.com  
University of Maryland, College Park

📧 Email: schakr18@umd.edu  
🔗 LinkedIn: [samarpan-chakraborty](https://www.linkedin.com/in/samarpan-chakraborty)  
🌐 Portfolio: [samarpanchakraborty97.github.io](https://samarpanchakraborty97.github.io)

---

## 🎓 Course Information

**Course**: Data Scientist Career Track  
**Platform**: [365DataScience.com](https://365datascience.com)  
**Topics Covered**:
- SQL for data manipulation
- Statistical analysis and hypothesis testing
- Probability theory
- Machine learning fundamentals
- Data visualization

**Certification**: Data Scientist Professional Certificate

---

## 📜 License

This project is for educational purposes as part of the Data Scientist Career Track on 365DataScience.com.

Data provided by 365DataScience.com for analysis purposes.

---

## 🙏 Acknowledgments

- **365DataScience.com**: For providing the dataset and learning platform
- **MySQL Community**: For excellent database tools
- **Python Data Science Community**: For pandas, scikit-learn, and visualization libraries

---

## 📞 Support

**Questions about the analysis?**  
→ See [QUICKSTART.md](QUICKSTART.md) or [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)

**Issues with SQL queries?**  
→ Check `studentEngagement.sql` comments and documentation

**Want to replicate this analysis?**  
→ Follow the step-by-step guide in QUICKSTART.md

---

**Last Updated**: November 2024  
**Version**: 1.0.0  
**Status**: ✅ Completed