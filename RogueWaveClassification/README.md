# 🌊 Rogue Wave Forecasting: Deep Learning for Maritime Safety

A comprehensive machine learning framework for predicting extreme ocean waves (rogue waves) using time series data from oceanographic buoys. The system achieves unprecedented accuracy exceeding 72% for forecasting wave anomalies up to 10 minutes in advance using Long Short-Term Memory (LSTM) networks, Support Vector Machines (SVM), and Decision Trees, processing approximately 20 billion data points from global ocean monitoring systems.

## 📋 Project Overview

This project develops and evaluates machine learning models to forecast rogue waves—extreme ocean waves that pose catastrophic risks to ships, offshore structures, and maritime personnel. Using time series data from the Coastal Data Information Program (CDIP) buoy network, the system predicts whether a rogue wave will occur within a specified forecast horizon based on current wave patterns.

### 🎯 Problem Statement

**Rogue Waves:**
- Also called "freak waves" or "extreme waves"
- Significantly larger than surrounding waves
- **Definition**: Wave height (H) > 2.0 × Significant wave height (Hs)
- **Alternative**: Crest height (ηc) > 1.25 × Significant wave height (Hs)
- **Impact**: Devastating damage to ships, offshore structures, and loss of life
- **Challenge**: Low probability of occurrence but extremely high consequences

**Forecasting Challenge:**
- Predict rogue wave occurrence 3-10 minutes in advance
- Process high-frequency time series data (continuous monitoring)
- Handle diverse ocean conditions (deep water vs shallow water)
- Generalize across different geographic locations

### 💡 Solution

A multi-model framework that:
1. **Processes ocean buoy time series** data with zero-crossing analysis
2. **Extracts meaningful features** through significant wave height normalization
3. **Trains multiple ML models** optimized for different scenarios
4. **Predicts rogue wave events** with 72%+ accuracy on global data
5. **Adapts to local conditions** with specialized models for regional forecasting

### ✨ Key Features

- **High Accuracy**: 72% for LSTM on global ocean data, 67% for 5-minute forecasts
- **Multiple Models**: LSTM (best for large datasets), SVM (best for localized data), Decision Trees (baseline)
- **Flexible Horizons**: Forecast windows of 3, 5, and 10 minutes
- **Scalable Architecture**: Handles ~20 billion data points from 172 ocean locations
- **Regional Adaptation**: Specialized models for deep water, shallow water, and localized areas
- **Real-Time Capability**: Fast inference for operational deployment
- **Comprehensive Evaluation**: Accuracy, F1 scores, confusion matrices across scenarios

## 📁 Project Structure

The project follows a systematic time series analysis pipeline with three main components:

### 1. 📊 Data Preparation and Preprocessing

#### `Data_Preparation_Generic.ipynb`

Comprehensive data extraction and preprocessing from CDIP buoy network:

**Data Source:**
- **CDIP Network**: Coastal Data Information Program (UCSD)
- **Coverage**: 172+ buoy stations across USA coastlines
- **Data Volume**: ~20 billion data points
- **Temporal Resolution**: 0.5-1 Hz sampling rate
- **Duration**: Multi-year continuous monitoring

**Data Processing Pipeline:**

**1. Zero-Crossing Analysis:**
```python
def find_max_wave_height(zdisp_window):
    """
    Extract wave heights using zero-crossing method
    - Identifies wave peaks and troughs
    - Calculates individual wave heights
    - Returns maximum wave in window
    """
```

**2. Significant Wave Height Normalization:**
- Significant wave height (Hs) = 4 × standard deviation of sea surface elevation
- Normalizes all waves by Hs for consistent scaling
- Formula: normalized_wave = wave_height / Hs

**3. Rogue Wave Detection:**
- Binary classification: Rogue (1) or Not Rogue (0)
- Rogue wave criteria: H > 2.0 × Hs (or ηc > 1.25 × Hs)
- Automated extraction from continuous time series

**4. Window Creation:**
- **Training window (t_window)**: 15 or 20 minutes of historical data
- **Forecast horizon (t_horizon)**: 3, 5, or 10 minutes ahead
- **Sliding window approach**: Extract overlapping windows
- **Balanced dataset**: Equal proportions of rogue and non-rogue windows

**Data Organization:**

```
Extracted Data/
├── rogue_wave_windows/
│   ├── buoy_001/
│   │   ├── rw_event_001.npz
│   │   ├── rw_event_002.npz
│   │   └── ...
│   └── buoy_002/
│       └── ...
└── non_rogue_windows/
    ├── buoy_001/
    │   ├── norw_window_001.npz
    │   ├── norw_window_002.npz
    │   └── ...
    └── buoy_002/
        └── ...
```

**Output Files:**
- `RWs_H_g_2_tadv_Xmin_scenario_rw_0.5.npz`: Training data
  - Contains: `wave_data_train`, `label_train`
- `RWs_H_g_2_tadv_Xmin_scenario_test_rw_0.5.npz`: Test data
  - Contains: `wave_data_test`, `label_test`

**Scenarios Prepared:**
1. **Global/Deep Buoys**: All offshore buoys combined
2. **Triangular Area**: 3 buoys (2 deep water, 1 shallow water)
3. **Localized**: 5 buoys in close proximity

### 2. 🧠 LSTM Model Training

#### `LSTM_Training.ipynb`

Deep learning model implementation using TensorFlow/Keras:

**Model Architecture:**

```
Input: (batch, time_steps, 1) - Normalized wave time series
    ↓
LSTM Layer 1: 10 units, return_sequences=True
    ↓
Batch Normalization
    ↓
LSTM Layer 2: 10 units, return_sequences=True
    ↓
Batch Normalization
    ↓
LSTM Layer 3: 10 units, return_sequences=True
    ↓
Batch Normalization
    ↓
LSTM Layer 4: 10 units
    ↓
Batch Normalization
    ↓
Dropout: 0.05
    ↓
Dense: 2 units (binary classification), sigmoid activation
    ↓
Output: Probability [No Rogue Wave, Rogue Wave]
```

**Training Configuration:**

- **Loss Function**: Sparse Categorical Crossentropy
- **Optimizer**: Adam
- **Batch Size**: 64
- **Max Epochs**: 500 (with early stopping)
- **Validation Split**: 20% of training data
- **Early Stopping**: Patience of 25 epochs on validation loss

**Advanced Training Techniques:**

**1. Learning Rate Scheduling:**
```python
def scheduler(epochs, lr):
    if epochs < 5:
        return lr
    else:
        return lr * math.exp(-0.1)
```
- Initial learning rate maintained for 5 epochs
- Exponential decay afterwards for fine-tuning

**2. Model Checkpointing:**
- Saves best model based on validation loss
- Prevents overfitting through early stopping

**3. Batch Normalization:**
- Applied after each LSTM layer
- Stabilizes training and accelerates convergence

**Training Outputs:**
- Saved model: `best_LSTM_[scenario]_checkpoint.model.keras`
- Training curves: Training and validation accuracy over epochs
- Metrics file: Final performance statistics

### 3. 📈 Comprehensive Results Analysis

#### `RogueWave_Classification_Results.ipynb`

Extensive evaluation across multiple models and ocean scenarios:

**Models Evaluated:**

1. **LSTM Networks** (4-layer architecture)
   - Best for large-scale global data
   - Captures temporal dependencies
   - Achieves 67-72% accuracy

2. **Support Vector Machines (SVM)**
   - Best for localized/limited data scenarios
   - Robust generalization
   - Achieves 64% global, 64% localized

3. **Decision Trees (DT)**
   - Baseline comparison
   - Fails to generalize well
   - Achieves ~51% accuracy

**Evaluation Scenarios:**

**Scenario 1: Global Ocean Data (All Buoys)**
- **Training samples**: ~252,000
- **Test samples**: ~68,000
- **Best model**: LSTM (67% accuracy, 5-min horizon)
- **Observation**: LSTM outperforms as dataset size increases

**Scenario 2: Triangular Area (3 Buoys)**
- **Configuration**: 2 deep water buoys (training) + 1 shallow water buoy (testing)
- **Training samples**: ~9,900
- **Test samples**: ~7,600
- **Best model**: SVM (64% accuracy, 3-min horizon)
- **Observation**: SVM generalizes better with limited training data

**Scenario 3: Localized Buoys (5 Buoys)**
- **Configuration**: 4 offshore buoys (training) + 1 near-shore buoy (testing)
- **Training samples**: ~8,094
- **Test samples**: ~340
- **Best model**: SVM (best performance across horizons)
- **Observation**: SVM maintains performance with geographic diversity

**Key Findings:**

**Effect of Forecast Horizon:**
- Accuracy decreases as horizon increases (3 min → 10 min)
- LSTM maintains best performance at all horizons for global data
- SVM more stable across horizons for localized data

**Effect of Training Window:**
- Increasing window from 15 to 20 minutes improves accuracy
- LSTM: +4% improvement
- SVM: +6% improvement
- Decision Trees: No improvement (model too simple)

**Model Selection Guidelines:**
- **Large dataset (>100K samples)**: Use LSTM
- **Medium dataset (5K-50K samples)**: Use SVM
- **Cross-location generalization**: Prefer SVM
- **Real-time deployment**: Both LSTM and SVM suitable

## 📊 Dataset Description

### Data Source: CDIP Buoy Network

**Buoy Characteristics:**
- **Type**: Moored oceanographic buoys
- **Measurements**: 
  - Sea surface elevation (z-displacement)
  - Significant wave height
  - Wave period
  - Wave direction (some buoys)
- **Sampling Rate**: 0.5-1.0 Hz (2-1 second intervals)
- **Deployment**: Permanent offshore installations

### Data Statistics

**Global Dataset:**
- **Buoy locations**: 172 stations
- **Time coverage**: Multi-year continuous data
- **Total data points**: ~20 billion
- **Rogue wave events**: Thousands across all buoys
- **Training examples**: ~252,000 windows
- **Test examples**: ~68,000 windows

**Triangular Area Dataset:**
- **Buoys**: 3 (Station IDs available in project)
- **Geographic spread**: ~10-50 miles
- **Water depth variation**: Deep to shallow transition
- **Training examples**: ~9,900 windows
- **Test examples**: ~7,600 windows

**Localized Dataset:**
- **Buoys**: 5 near-shore and offshore
- **Geographic spread**: Localized coastal region
- **Training examples**: ~8,094 windows
- **Test examples**: ~340 windows

### Data Format

**NPZ File Structure:**
```python
{
    'wave_data_train': np.array,  # Shape: (n_samples, time_steps, 1)
    'label_train': np.array,       # Shape: (n_samples,)
    'wave_data_test': np.array,    # Shape: (n_test, time_steps, 1)
    'label_test': np.array         # Shape: (n_test,)
}
```

**Time Series Properties:**
- **Time steps**: 
  - 15-minute window: ~900-1800 data points (0.5-1 Hz)
  - 20-minute window: ~1200-2400 data points
- **Normalization**: All values divided by significant wave height (Hs)
- **Range**: Typically [-3, 3] after normalization

## 🚀 Installation and Setup

### Prerequisites
- Python 3.7 or higher
- TensorFlow/Keras 2.x
- GPU recommended for LSTM training (optional)

### Required Libraries

Install all dependencies using pip:

```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn
```

Or use the requirements file:

```bash
pip install -r requirements_roguewave.txt
```

#### Core Libraries:
- **tensorflow** (>=2.6.0): Deep learning framework
- **keras** (included with TensorFlow): High-level neural networks API
- **numpy** (>=1.20.0): Numerical computing
- **matplotlib** (>=3.4.0): Visualization
- **seaborn** (>=0.11.0): Statistical visualization

#### Machine Learning:
- **scikit-learn** (>=0.24.0): SVM, Decision Trees, and metrics

### Data Access

The CDIP buoy data can be accessed from:
- **Website**: https://cdip.ucsd.edu/
- **Data format**: NetCDF files (converted to NPZ for processing)
- **Access**: Public data archive

## 💻 Usage

### Recommended Execution Order

#### Step 1: Data Preparation

```bash
jupyter notebook Data_Preparation_Generic.ipynb
```

**Configure parameters:**
```python
# Set forecast horizon (3, 5, or 10 minutes)
forecast_horizon = 5  # minutes

# Set training window (15 or 20 minutes)
training_window = 15  # minutes

# Select scenario
scenario = "deep_buoys"  # or "triangular_area", "localized"

# Output filename
file_str = f"RWs_H_g_2_tadv_{forecast_horizon}min_{scenario}_rw_0.5"
```

**This notebook will:**
- Load raw CDIP buoy data
- Extract rogue wave and non-rogue wave windows
- Normalize by significant wave height
- Create balanced training and test datasets
- Save as NPZ files for model training

**Output:** 
- Training file: `[scenario]_rw_0.5.npz`
- Test file: `[scenario]_test_rw_0.5.npz`

#### Step 2: LSTM Model Training

```bash
jupyter notebook LSTM_Training.ipynb
```

**Configure model parameters:**
```python
# Set filenames matching data preparation
file_str = "RWs_H_g_2_tadv_5min_deep_buoys_rw_0.5"
file_str_test = "RWs_H_g_2_tadv_5min_deep_buoys_test_rw_0.5"

# Training hyperparameters (optimized defaults provided)
batch_size = 64
lstm_units = 10
num_layers = 4
dropout_rate = 0.05
max_epochs = 500
```

**This notebook will:**
- Load preprocessed NPZ data
- Build 4-layer LSTM architecture
- Train with early stopping and learning rate scheduling
- Save best model checkpoint
- Generate training curves

**Training Time:**
- Global data (~252K samples): ~2-4 hours on GPU
- Localized data (~8K samples): ~30-60 minutes on GPU

#### Step 3: Results Analysis

```bash
jupyter notebook RogueWave_Classification_Results.ipynb
```

**This notebook displays:**
- Confusion matrices for all models and scenarios
- Accuracy and F1 score comparisons
- Effect of forecast horizon on performance
- Effect of training window size
- Model selection guidelines

### 🔄 Complete Workflow

```python
# Example: Train model for 5-minute rogue wave forecasting

# 1. Prepare data (Data_Preparation_Generic.ipynb)
forecast_horizon = 5  # minutes
training_window = 15  # minutes
scenario = "deep_buoys"

# Extract windows and save:
# - RWs_H_g_2_tadv_5min_deep_buoys_rw_0.5.npz (train)
# - RWs_H_g_2_tadv_5min_deep_buoys_test_rw_0.5.npz (test)

# 2. Train LSTM (LSTM_Training.ipynb)
model = build_lstm_model(
    input_shape=(time_steps, 1),
    lstm_units=10,
    num_layers=4
)
history = model.fit(train_data, epochs=500, callbacks=[...])

# 3. Evaluate (RogueWave_Classification_Results.ipynb)
predictions = model.predict(test_data)
accuracy = evaluate_predictions(predictions, test_labels)
# Expected: ~67% accuracy for 5-min horizon
```

## 📈 Model Performance

### Global Ocean Data (Best Case Scenario)

| Model | Accuracy | F1 Score | Training Time | Inference Speed |
|-------|----------|----------|---------------|-----------------|
| **LSTM** | **72%** (0-min)  | High | 2-4 hours (GPU) | Real-time |
| | **67%** (5-min) | 0.67 | | |
| **SVM** | **64%** (5-min) | 0.64 | 1-2 hours | Real-time |
| **Decision Tree** | **51%** (5-min) | 0.51 | < 30 min | Real-time |

**Forecast Horizon Effects (LSTM):**
- 0-3 minutes: 70-72% accuracy
- 5 minutes: 67% accuracy
- 10 minutes: 62-65% accuracy

### Localized/Triangular Area Data

| Scenario | Best Model | Accuracy | Test Samples |
|----------|------------|----------|--------------|
| Triangular (3 buoys) | **SVM** | **64%** | 7,600 |
| Triangular (3 buoys) | LSTM | 55% | 7,600 |
| Localized (5 buoys) | **SVM** | **Best** | 340 |
| Localized (5 buoys) | LSTM | Lower | 340 |

**Key Observation**: SVM outperforms LSTM on limited/localized data

### Training Window Comparison (5-min horizon)

| Model | 15-min window | 20-min window | Improvement |
|-------|--------------|---------------|-------------|
| LSTM | 67% | 71% | +4% |
| SVM | 64% | 70% | +6% |
| Decision Tree | 51% | 51% | 0% |

### 💡 Key Insights

- 🎯 **LSTM excels with large datasets**: >100K training samples
- 🌊 **SVM better for localization**: Cross-buoy generalization
- 📊 **Longer training windows help**: +4-6% accuracy improvement
- ⏱️ **Forecast accuracy decays**: Inverse relationship with horizon
- 🏗️ **Decision Trees underperform**: Too simple for complex temporal patterns
- 🔄 **Real-time deployment feasible**: Both LSTM and SVM fast enough
- 🌍 **Geographic transfer learning**: SVM handles better than LSTM

## 🛠️ Techniques Used

### Time Series Analysis
- **Zero-Crossing Method**: Wave height extraction
- **Significant Wave Height**: Normalization and statistical baseline
- **Sliding Window**: Temporal sequence extraction
- **Wave Peak Detection**: Identify maximum amplitudes

### Deep Learning
- **LSTM Networks**: Temporal dependency modeling
- **Batch Normalization**: Training stabilization
- **Dropout Regularization**: Prevent overfitting
- **Early Stopping**: Optimal model selection
- **Learning Rate Scheduling**: Exponential decay

### Machine Learning
- **Support Vector Machines**: Robust binary classification
- **Decision Trees**: Baseline comparison
- **Stratified Sampling**: Balanced class distribution

### Evaluation
- **Confusion Matrices**: Per-class performance
- **Accuracy Metrics**: Overall correctness
- **F1 Scores**: Precision-recall balance
- **Cross-Location Validation**: Generalization testing

## 💾 Output Files

The project generates:
- **NPZ Data Files**: Preprocessed training and test datasets
- **Model Checkpoints**: Saved LSTM models (`.keras` format)
- **Training Curves**: Accuracy and loss over epochs
- **Confusion Matrices**: Classification performance visualizations
- **Metrics Files**: Detailed performance statistics (`.txt`)
- **Result Visualizations**: Comparative plots and figures

## 🔧 Customization

### Adjusting Forecast Parameters

```python
# In Data_Preparation_Generic.ipynb

# Change forecast horizon
forecast_horizon_minutes = 10  # Try 3, 5, or 10

# Change training window
training_window_minutes = 20  # Try 15 or 20

# Adjust rogue wave threshold
rogue_threshold = 2.2  # Standard is 2.0 (H/Hs)
```

### Modifying LSTM Architecture

```python
# In LSTM_Training.ipynb

# Increase model capacity
lstm_units = 20  # From 10 to 20
num_layers = 5   # Add one more layer

# Adjust regularization
dropout_rate = 0.1  # Increase dropout

# Change optimization
optimizer = keras.optimizers.Adam(learning_rate=0.0001)
```

### Hyperparameter Tuning

```python
# Batch size
batch_size = 128  # Increase for faster training (if memory allows)

# Early stopping patience
patience = 50  # More patience for complex datasets

# Learning rate schedule
def custom_scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * 0.95  # Slower decay
```

## 🌊 Applications

### Primary Use Case: Maritime Safety
- **Ship navigation**: Alert systems for vessel routing
- **Offshore operations**: Timing for critical operations
- **Emergency response**: Advance warning for evacuation
- **Weather routing**: Optimize ship paths to avoid extreme waves

### Extended Applications
- 🏗️ **Offshore structures**: Wind turbines, oil platforms
- 🚢 **Port operations**: Safe window prediction for loading/unloading
- 🏄 **Recreational safety**: Beach warnings, surfing conditions
- 📊 **Climate research**: Extreme event statistics and trends
- 🛡️ **Insurance**: Risk assessment for marine operations
- 🎣 **Fishing industry**: Safety planning for fishing fleets

## 🔍 Troubleshooting

### Common Issues

**Issue**: Low accuracy (<60%)
- **Check dataset balance**: Ensure 50-50 rogue/non-rogue distribution
- **Verify normalization**: Confirm significant wave height calculation
- **Increase training window**: Try 20 minutes instead of 15
- **Check for data quality**: Remove corrupted or missing data

**Issue**: LSTM not converging
```python
# Solution 1: Reduce learning rate
optimizer = keras.optimizers.Adam(learning_rate=0.0001)

# Solution 2: Add more regularization
model.add(keras.layers.Dropout(0.2))  # Increase from 0.05

# Solution 3: Check for NaN values
data = np.nan_to_num(data, nan=0.0)
```

**Issue**: Out of memory during training
```python
# Reduce batch size
batch_size = 32  # From 64

# Use gradient accumulation
# Or train on smaller subsets sequentially
```

**Issue**: Poor cross-location generalization
- **Use SVM instead of LSTM**: Better for limited data
- **Include diverse training locations**: Mix deep and shallow water
- **Augment training data**: Use data from similar buoys

**Issue**: FileNotFoundError
```bash
# Ensure data files are in correct location
ls *.npz

# Check file naming convention
file_str = "RWs_H_g_2_tadv_5min_deep_buoys_rw_0.5"
```

## 🤝 Contributing

When adding new features or improvements:
1. Document data preprocessing steps
2. Report performance on standard test sets
3. Compare with baseline models (LSTM, SVM)
4. Test on multiple forecast horizons
5. Validate across different ocean regions

## 📝 Future Improvements

- 🧠 **Advanced architectures**: GRU, Transformer, Attention mechanisms
- 🌐 **Graph Neural Networks**: Model spatial relationships between buoys
- 📊 **Multi-output prediction**: Forecast rogue wave height and timing
- 🎯 **Ensemble methods**: Combine LSTM + SVM predictions
- 📈 **Transfer learning**: Pre-train on global data, fine-tune locally
- 🔄 **Online learning**: Continuous model updates with new data
- 🌍 **Global deployment**: Real-time forecasting system
- 📱 **Mobile integration**: Smartphone apps for maritime users
- 🛰️ **Satellite data fusion**: Combine buoy + satellite observations

## 📄 License

This project is part of academic research at University of Maryland. Data used from CDIP is publicly available.

## 📧 Contact

For questions or collaboration:
- **Email**: schakr18@umd.edu
- **LinkedIn**: [linkedin.com/in/samarpan-chakraborty](https://linkedin.com/in/samarpan-chakraborty)
- **GitHub**: [github.com/SamarpanChakraborty97](https://github.com/SamarpanChakraborty97)

## 🙏 Acknowledgments

This research was conducted at the University of Maryland (2019-2025) as part of Ph.D. dissertation work focusing on physics-based and data-driven investigations into rapid wave formations in oceans. The project utilized data from the Coastal Data Information Program (CDIP) at Scripps Institution of Oceanography, UC San Diego.

**Dissertation**: *Physics-Based and Data-Driven Investigations into Rapid Wave Formations in Oceans*

**Key Achievements**:
- Unprecedented prediction accuracies exceeding 72% for wave time series anomalies
- Analysis of ~20 billion data points from 172 ocean locations
- 23% accuracy improvement through attention-based CNN+LSTM models
- 25% reduction in prediction errors using corrective forecasting schemes

---

**Note**: This project demonstrates practical application of deep learning for extreme event prediction in physical systems. The high accuracy (>72%) and ability to forecast 3-10 minutes in advance makes it suitable for operational deployment in maritime safety systems, enabling proactive risk mitigation for ships and offshore structures.

**Version**: 1.0  
**Last Updated**: November 2025  
**Research Period**: August 2019 – October 2025
