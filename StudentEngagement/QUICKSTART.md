# Quick Start Guide - Student Engagement Analysis

## Fast Setup (20 minutes)

### 1. Install Dependencies

```bash
# Option A: Using pip
pip install pandas numpy matplotlib seaborn scikit-learn mysql-connector-python jupyter

# Option B: Using conda (recommended)
conda create -n student-engagement python=3.9
conda activate student-engagement
pip install pandas numpy matplotlib seaborn scikit-learn mysql-connector-python jupyter
```

Or use requirements file:
```bash
pip install -r requirements.txt
```

### 2. Set Up MySQL Database

**Option A: Import from SQL file**
```bash
# Create database and import structure/data
mysql -u your_username -p < studentEngagement.sql

# Or using MySQL Workbench:
# File → Run SQL Script → Select studentEngagement.sql
```

**Option B: Manual setup** (if you have existing data)
```sql
CREATE DATABASE data_scientist_project;
USE data_scientist_project;

-- Create tables (see SQL file for full schema)
-- Load your data
```

### 3. Verify Database Connection

```python
import mysql.connector

# Test connection
conn = mysql.connector.connect(
    host="localhost",
    user="your_username",
    password="your_password",
    database="data_scientist_project"
)

print("✅ Database connection successful!")
conn.close()
```

### 4. Export Data from Database

Run SQL queries to generate CSV files:

```sql
-- Export paid users Q2 2021
SELECT student_id, minutes_watched
FROM (/* see studentEngagement.sql for full query */)
WHERE paid_in_q2 = 1 AND YEAR(date_watched) = 2021
INTO OUTFILE '/path/to/minutes_watched_2021_paid_1.csv'
FIELDS TERMINATED BY ',' ENCLOSED BY '"'
LINES TERMINATED BY '\n';

-- Repeat for: paid_0 (free), 2022_paid_1, 2022_paid_0, certificates
```

Or use Python to export:

```python
import pandas as pd
import mysql.connector

conn = mysql.connector.connect(
    host="localhost",
    user="your_username", 
    password="your_password",
    database="data_scientist_project"
)

# Export paid users 2021
query_paid_2021 = """
    SELECT student_id, minutes_watched
    FROM (/* your query here */)
    WHERE paid_in_q2 = 1
"""
df = pd.read_sql(query_paid_2021, conn)
df.to_csv('minutes_watched_2021_paid_1.csv', index=False)

conn.close()
```

### 5. Run the Analysis

```bash
# Open Jupyter notebook
jupyter notebook FinalProject.ipynb

# In Jupyter: Kernel → Restart & Run All
# Or: Cell → Run All
```

---

## 📚 Execution Workflow

### Step-by-Step Process

```
1. Database Setup (studentEngagement.sql)
   ├─> Create database and tables
   ├─> Load student data
   ├─> Create views (subscription_info, purchase_info)
   └─> Export CSVs for analysis
   
2. Data Analysis (FinalProject.ipynb)
   ├─> Load CSV files
   ├─> Exploratory data analysis (EDA)
   ├─> Visualize distributions
   ├─> Outlier detection & removal
   ├─> Statistical analysis
   ├─> Linear regression model
   └─> Generate insights
   
3. Results & Reporting (FinalProject.xlsx)
   └─> Excel summary with statistics
```

---

## ⚡ Expected Results

### After Database Setup

✅ **Database Created**:
- Database: `data_scientist_project`
- Tables: 4 (student_info, student_purchases, student_video_watched, student_certificates)
- Views: 2 (subscription_info, purchase_info)

✅ **CSV Files Generated**:
```
✓ minutes_watched_2021_paid_1.csv   (~1000 rows)
✓ minutes_watched_2021_paid_0.csv   (~5000 rows)
✓ minutes_watched_2022_paid_1.csv   (~1200 rows)
✓ minutes_watched_2022_paid_0.csv   (~4800 rows)
✓ minutes_and_certificates.csv      (~500 rows)
```

### After Running Analysis

✅ **Visualizations Created**:
- KDE plots comparing paid vs free users
- Distribution plots for 2021 vs 2022
- Scatter plots for regression analysis
- Outlier-filtered distributions

✅ **Statistical Outputs**:
```
Paid Users Q2 2021:
  Mean: 360.10 minutes
  Median: 161.93 minutes
  Std Dev: 499.62 minutes

Free Users Q2 2021:
  Mean: 14.21 minutes
  Median: 2.79 minutes
  Std Dev: 24.48 minutes

Paid Users Q2 2022:
  Mean: 292.22 minutes
  Median: 119.75 minutes
  Std Dev: 420.19 minutes

Free Users Q2 2022:
  Mean: 16.25 minutes
  Median: 5.02 minutes
  Std Dev: 24.83 minutes
```

✅ **Machine Learning Model**:
```
Linear Regression: certificates vs minutes_watched
  Equation: y = 0.0022x + 0.82
  Correlation: r = 0.71
  
Prediction Example:
  Input: 1200 minutes
  Output: 3.24 certificates
```

---

## 🔧 Configuration Quick Reference

### SQL Query Parameters

```sql
-- Change year for analysis
WHERE YEAR(date_watched) = 2021  -- or 2022

-- Filter by subscription status
WHERE paid_in_q2 = 1  -- Paid users
WHERE paid_in_q2 = 0  -- Free users

-- Date range (Q2 = April-June)
WHERE date_watched BETWEEN '2021-04-01' AND '2021-06-30'
```

### Python Analysis Parameters

```python
# In FinalProject.ipynb

# Outlier threshold (adjust as needed)
outlier_quantile = 0.99  # Remove top 1%

# Regression parameters
test_size = 0.2          # 20% for testing
random_state = 365       # For reproducibility

# Visualization settings
figsize = (10, 10)       # Figure dimensions
color_paid = 'blue'      # Color for paid users
color_free = 'red'       # Color for free users
```

---

## 🚀 Quick Test After Setup

### Test 1: Database Connection

```python
import mysql.connector

try:
    conn = mysql.connector.connect(
        host="localhost",
        user="your_username",
        password="your_password",
        database="data_scientist_project"
    )
    cursor = conn.cursor()
    
    # Test query
    cursor.execute("SELECT COUNT(*) FROM student_video_watched")
    result = cursor.fetchone()
    print(f"✅ Total video watch records: {result[0]}")
    
    cursor.close()
    conn.close()
    
except Exception as e:
    print(f"❌ Error: {e}")
```

### Test 2: Load and Inspect Data

```python
import pandas as pd

# Load one dataset
df = pd.read_csv('minutes_watched_2021_paid_1.csv')

print(f"✅ Data loaded successfully!")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nFirst 5 rows:")
print(df.head())

# Basic statistics
print(f"\nStatistics:")
print(df.describe())
```

### Test 3: Quick Visualization

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
paid_2021 = pd.read_csv('minutes_watched_2021_paid_1.csv')

# Create plot
plt.figure(figsize=(10, 6))
sns.histplot(data=paid_2021, x='minutes_watched', bins=50, kde=True)
plt.title('Distribution of Viewing Time - Paid Users Q2 2021')
plt.xlabel('Minutes Watched')
plt.ylabel('Frequency')
plt.show()

print("✅ Visualization working!")
```

---

## 📊 Understanding the SQL Queries

### Query 1: Create Subscription View

**Purpose**: Calculate subscription end dates based on plan type

```sql
CREATE VIEW subscription_info AS
    SELECT 
        purchase_id,
        student_id,
        plan_id,
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
```

**Explanation**:
- If not refunded: Calculate end date based on plan
  - Plan 0 (Monthly): +1 month
  - Plan 1 (Quarterly): +3 months
  - Plan 2 (Annual): +12 months
- If refunded: Use refund date as end date

### Query 2: Identify Paid Users in Q2

**Purpose**: Flag students who had active subscriptions in Q2

```sql
CREATE VIEW purchase_info AS
    SELECT 
        student_id,
        CASE
            WHEN date_end < '2021-04-01' THEN 0        -- Expired before Q2
            WHEN date_start > '2021-06-30' THEN 0       -- Started after Q2
            ELSE 1                                       -- Active during Q2
        END AS paid_q2_2021
    FROM subscription_info;
```

**Explanation**:
- Returns 1 if subscription overlapped with Q2 2021 (Apr 1 - Jun 30)
- Returns 0 if subscription ended before or started after Q2

### Query 3: Calculate Minutes Watched

**Purpose**: Aggregate viewing time by student for a year

```sql
SELECT 
    student_id,
    ROUND(SUM(seconds_watched) / 60, 2) AS minutes_watched
FROM student_video_watched
WHERE YEAR(date_watched) = 2021
GROUP BY student_id;
```

**Explanation**:
- Sums all `seconds_watched` per student
- Converts to minutes (÷ 60)
- Rounds to 2 decimal places
- Filters by year

### Query 4: Join Minutes with Payment Status

**Purpose**: Combine viewing data with subscription information

```sql
SELECT 
    min_watched.student_id,
    min_watched.minutes_watched,
    IF(income.date_start IS NULL, 0, MAX(income.paid_q2_2021)) AS paid_in_q2
FROM (
    -- Subquery: get minutes watched
    SELECT student_id, ROUND(SUM(seconds_watched)/60, 2) AS minutes_watched
    FROM student_video_watched
    WHERE YEAR(date_watched) = 2021
    GROUP BY student_id
) min_watched
LEFT JOIN purchase_info income 
    ON min_watched.student_id = income.student_id
GROUP BY student_id;
```

**Explanation**:
- Inner query: Calculates minutes per student
- LEFT JOIN: Keeps all students, even without purchases
- IF statement: Returns 0 if no purchase found, otherwise payment status
- MAX: Handles multiple purchases (takes most recent status)

---

## 🎨 Visualization Guide

### Plot 1: KDE Distribution Comparison

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
paid_2021 = pd.read_csv('minutes_watched_2021_paid_1.csv')
free_2021 = pd.read_csv('minutes_watched_2021_paid_0.csv')

# Create figure
plt.figure(figsize=(10, 8))

# Plot distributions
sns.kdeplot(
    data=paid_2021, 
    x='minutes_watched',
    color='blue',
    label='Paid Users Q2 2021',
    linewidth=2
)

sns.kdeplot(
    data=free_2021,
    x='minutes_watched', 
    color='red',
    label='Free Users Q2 2021',
    linewidth=2
)

# Formatting
plt.xlabel('Minutes Watched', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.title('Student Engagement Distribution by Subscription Type', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### Plot 2: Year-over-Year Comparison

```python
# Load all datasets
paid_2021 = pd.read_csv('minutes_watched_2021_paid_1.csv')
paid_2022 = pd.read_csv('minutes_watched_2022_paid_1.csv')

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 2021
axes[0].hist(paid_2021['minutes_watched'], bins=50, color='skyblue', edgecolor='black')
axes[0].set_title('Paid Users - Q2 2021', fontsize=13)
axes[0].set_xlabel('Minutes Watched')
axes[0].set_ylabel('Frequency')

# 2022
axes[1].hist(paid_2022['minutes_watched'], bins=50, color='lightcoral', edgecolor='black')
axes[1].set_title('Paid Users - Q2 2022', fontsize=13)
axes[1].set_xlabel('Minutes Watched')
axes[1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()
```

### Plot 3: Regression Analysis

```python
from sklearn.linear_model import LinearRegression
import numpy as np

# Load certificate data
cert_data = pd.read_csv('minutes_and_certificates.csv')

# Prepare data
X = cert_data['minutes_watched'].values.reshape(-1, 1)
y = cert_data['num_certificates'].values

# Fit model
model = LinearRegression()
model.fit(X, y)

# Create predictions for plotting
X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_plot = model.predict(X_plot)

# Plot
plt.figure(figsize=(10, 8))
plt.scatter(X, y, alpha=0.5, s=50, label='Actual data')
plt.plot(X_plot, y_plot, color='red', linewidth=2, 
         label=f'y = {model.coef_[0]:.4f}x + {model.intercept_:.2f}')

plt.xlabel('Minutes Watched', fontsize=12)
plt.ylabel('Number of Certificates', fontsize=12)
plt.title('Certificate Completion vs Viewing Time', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Print stats
r = np.corrcoef(X[:, 0], y)[0, 1]
print(f'Correlation coefficient: {r:.4f}')
print(f'R-squared: {r**2:.4f}')
```

---

## 🔬 Analysis Workflow

### Phase 1: Data Preparation (SQL)

**Time**: 30-60 minutes

```
Step 1: Create database and load tables
Step 2: Create subscription_info view
Step 3: Create purchase_info view
Step 4: Export CSV files
        ├─> 2021 paid users
        ├─> 2021 free users
        ├─> 2022 paid users
        ├─> 2022 free users
        └─> Certificate data
```

**Key SQL Commands**:
```sql
USE data_scientist_project;
CREATE VIEW subscription_info AS ...;
CREATE VIEW purchase_info AS ...;
SELECT ... INTO OUTFILE ...;
```

### Phase 2: Exploratory Analysis (Python)

**Time**: 45-60 minutes

```
Step 1: Load all CSV files
Step 2: Calculate descriptive statistics
        ├─> Mean, median, std dev
        ├─> Min, max, quartiles
        └─> Compare 2021 vs 2022
        
Step 3: Visualize distributions
        ├─> KDE plots
        ├─> Histograms
        └─> Box plots (optional)
        
Step 4: Identify outliers
        ├─> Calculate 99th percentile
        ├─> Filter extreme values
        └─> Export cleaned data
```

### Phase 3: Statistical Analysis

**Time**: 30 minutes

```
Step 1: Hypothesis testing setup
        H₀: μ_2021 = μ_2022
        H₁: μ_2021 ≠ μ_2022
        
Step 2: Calculate test statistics
        ├─> t-statistic
        ├─> p-value
        └─> Confidence intervals
        
Step 3: Probability analysis
        ├─> P(watched 2021)
        ├─> P(watched 2022)
        └─> P(watched both)
```

### Phase 4: Predictive Modeling

**Time**: 30 minutes

```
Step 1: Prepare features and target
        X = minutes_watched
        y = num_certificates
        
Step 2: Split data (80/20)

Step 3: Train linear regression model

Step 4: Evaluate performance
        ├─> Correlation coefficient
        ├─> R-squared
        ├─> Residual analysis
        └─> Predictions
```

---

## 🛠️ Troubleshooting

### Problem 1: Database Connection Error

```
Error: Can't connect to MySQL server
```

**Solutions**:
```bash
# Check MySQL is running
sudo systemctl status mysql

# Start MySQL if stopped
sudo systemctl start mysql

# Verify credentials
mysql -u your_username -p
```

### Problem 2: CSV Export Permission Denied

```
Error: Access denied for user to file
```

**Solutions**:

**Option A: Use Python instead**
```python
import pandas as pd
import mysql.connector

conn = mysql.connector.connect(...)
df = pd.read_sql("SELECT ...", conn)
df.to_csv('output.csv', index=False)
```

**Option B: Grant file privileges**
```sql
GRANT FILE ON *.* TO 'your_username'@'localhost';
FLUSH PRIVILEGES;
```

**Option C: Change secure_file_priv**
```sql
-- Check current setting
SHOW VARIABLES LIKE 'secure_file_priv';

-- Use allowed directory or disable restriction
-- (Edit my.cnf and restart MySQL)
```

### Problem 3: Missing CSV Files

```
Error: FileNotFoundError: minutes_watched_2021_paid_1.csv
```

**Solutions**:
```python
# Check working directory
import os
print(os.getcwd())
print(os.listdir('.'))

# Change directory if needed
os.chdir('/path/to/csv/files')

# Or use full paths
df = pd.read_csv('/full/path/to/file.csv')
```

### Problem 4: Import Errors

```
Error: No module named 'seaborn'
```

**Solutions**:
```bash
# Install missing package
pip install seaborn

# Or install all requirements
pip install -r requirements.txt

# Verify installation
python -c "import seaborn; print(seaborn.__version__)"
```

### Problem 5: Memory Error with Large Dataset

```
Error: MemoryError
```

**Solutions**:
```python
# Read data in chunks
chunks = pd.read_csv('large_file.csv', chunksize=10000)
result = pd.concat([chunk.describe() for chunk in chunks])

# Or reduce data size
df_sample = df.sample(n=10000, random_state=42)

# Or increase available memory
# Use 64-bit Python, close other applications
```

### Problem 6: Visualization Not Displaying

```
Warning: No display found
```

**Solutions**:
```python
# For non-interactive environments
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Save instead of show
plt.savefig('output.png')

# For Jupyter notebooks
%matplotlib inline
```

---

## 📈 Performance Optimization Tips

### For Faster SQL Queries

```sql
-- Add indexes
CREATE INDEX idx_student_date ON student_video_watched(student_id, date_watched);
CREATE INDEX idx_purchase_date ON student_purchases(student_id, date_purchased);

-- Use EXPLAIN to analyze
EXPLAIN SELECT ...;

-- Optimize joins
-- Use indexes on join columns
-- Filter early (WHERE before JOIN)
```

### For Faster Python Processing

```python
# Use vectorized operations
df['minutes'] = df['seconds'] / 60  # Fast
# vs
df['minutes'] = df['seconds'].apply(lambda x: x / 60)  # Slow

# Read only needed columns
df = pd.read_csv('file.csv', usecols=['student_id', 'minutes_watched'])

# Use appropriate data types
df['student_id'] = df['student_id'].astype('int32')  # vs int64
```

---

## ✅ Success Checklist

**Before starting:**
- [ ] MySQL installed and running
- [ ] Python 3.9+ installed
- [ ] Required libraries installed
- [ ] Database created
- [ ] Tables loaded with data

**After SQL phase:**
- [ ] Views created (subscription_info, purchase_info)
- [ ] CSV files exported (5 files minimum)
- [ ] Data spot-checked for correctness
- [ ] File paths noted for Python analysis

**After Python analysis:**
- [ ] All CSV files load without errors
- [ ] Visualizations display correctly
- [ ] Statistics calculated (mean, median, std)
- [ ] Outliers identified and removed
- [ ] Linear regression model trained
- [ ] Results interpreted and documented

**Quality checks:**
- [ ] No missing values in key columns
- [ ] Reasonable data ranges (no negative minutes)
- [ ] Model correlation makes sense (r > 0.5)
- [ ] Visualizations have labels and titles
- [ ] Findings match expectations

---

## 🎯 Next Steps

1. **📊 Extend Analysis**
   - Add more years of data
   - Segment by demographics
   - Analyze course-specific engagement

2. **📈 Advanced Statistics**
   - Conduct formal hypothesis tests
   - Calculate confidence intervals
   - Perform ANOVA for multi-group comparison

3. **🤖 Improve Models**
   - Try polynomial regression
   - Add more features (age, location, course type)
   - Use Random Forest or XGBoost

4. **📱 Create Dashboard**
   - Build interactive Tableau dashboard
   - Use Plotly for interactive visualizations
   - Deploy Streamlit app

5. **📝 Report Findings**
   - Create executive summary
   - Present to stakeholders
   - Make data-driven recommendations

---

## 📚 Additional Resources

**SQL References**:
- [MySQL Documentation](https://dev.mysql.com/doc/)
- [W3Schools SQL Tutorial](https://www.w3schools.com/sql/)
- [SQLZoo Practice](https://sqlzoo.net/)

**Python Data Analysis**:
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Seaborn Tutorial](https://seaborn.pydata.org/tutorial.html)
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)

**Statistical Methods**:
- [Khan Academy Statistics](https://www.khanacademy.org/math/statistics-probability)
- [StatQuest YouTube](https://www.youtube.com/c/joshstarmer)
- [365DataScience Course](https://365datascience.com/)

---

## 💬 Getting Help

**SQL Issues?**  
→ Check `studentEngagement.sql` comments  
→ Review MySQL error logs

**Python Errors?**  
→ Check notebook cell outputs  
→ Review troubleshooting section above

**Analysis Questions?**  
→ See main [README](README_STUDENT_ENGAGEMENT.md)  
→ Review [PROJECT_STRUCTURE](PROJECT_STRUCTURE_STUDENT.md)

**Contact**:  
📧 schakr18@umd.edu

---

**Pro Tip**: Start by running just the first few cells of the notebook to verify your data loads correctly before running the entire analysis!

**Last Updated**: November 2024  
**Version**: 1.0.0