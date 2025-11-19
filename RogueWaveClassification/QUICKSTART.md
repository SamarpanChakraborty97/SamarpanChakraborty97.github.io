# Quick Start Guide - Rogue Wave Forecasting

## Fast Setup (20 minutes)

### 1. Install Dependencies

```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn scipy IPython
```

Or use requirements file:
```bash
pip install -r requirements_roguewave.txt
```

### 2. Verify Installation

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print(f"TensorFlow version: {tf.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"GPU available: {tf.config.list_physical_devices('GPU')}")
```

### 3. Obtain Data

**Option A: Use CDIP Public Data**
- Visit: https://cdip.ucsd.edu/
- Download historical buoy data (NetCDF format)
- Convert to NPZ using provided scripts

**Option B: Use Preprocessed Data** (if available)
- Place NPZ files in project directory
- Files should be named: `RWs_H_g_2_tadv_[horizon]min_[scenario]_rw_0.5.npz`

### 4. Run the Pipeline

**Quick Training (Pre-processed Data)**

```bash
# Step 1: Skip data prep if you have .npz files
# Step 2: Train LSTM model
jupyter notebook LSTM_Training.ipynb
# Run all cells (~2-4 hours on GPU for global data)

# Step 3: View results
jupyter notebook RogueWave_Classification_Results.ipynb
```

**Full Pipeline (From Raw Data)**

```bash
# Step 1: Prepare data
jupyter notebook Data_Preparation_Generic.ipynb
# Configure forecast horizon and window size
# Run all cells (~1-2 hours depending on data volume)

# Step 2: Train LSTM
jupyter notebook LSTM_Training.ipynb

# Step 3: Analyze results
jupyter notebook RogueWave_Classification_Results.ipynb
```

## Expected Results

After running the complete pipeline:

✅ **LSTM Model (Global Data)**:
- Accuracy: 67-72%
- F1 Score: 0.67-0.72
- Training time: 2-4 hours (GPU)

✅ **SVM Model (Localized Data)**:
- Accuracy: 64%
- Better cross-location generalization
- Training time: 1-2 hours

✅ **Saved Outputs**:
- Model checkpoint: `.keras` file
- Training curves: Accuracy/loss plots
- Confusion matrices: Performance visualization
- Metrics: Text file with detailed statistics

## Notebook Execution Order

```
1. Data_Preparation_Generic.ipynb (If starting from raw data)
   ├─> Loads CDIP buoy data
   ├─> Extracts wave windows
   ├─> Normalizes by significant wave height
   └─> Generates: [scenario]_rw_0.5.npz files
   
2. LSTM_Training.ipynb (REQUIRED)
   ├─> Loads NPZ data
   ├─> Builds 4-layer LSTM
   ├─> Trains with early stopping
   └─> Saves: best_LSTM_[scenario]_checkpoint.model.keras
   
3. RogueWave_Classification_Results.ipynb (Analysis)
   ├─> Displays confusion matrices
   ├─> Compares LSTM, SVM, DT
   └─> Shows effect of horizons and windows
```

## Key Files

| File | Purpose | Time | Required |
|------|---------|------|----------|
| `Data_Preparation_Generic.ipynb` | Data prep | 1-2 hrs | If no NPZ files |
| `LSTM_Training.ipynb` | Model training | 2-4 hrs | Always |
| `RogueWave_Classification_Results.ipynb` | Analysis | 5 min | View results |

## Configuration Quick Reference

### Data Preparation Parameters

```python
# In Data_Preparation_Generic.ipynb

# Forecast horizon (how far ahead to predict)
forecast_horizon = 5  # Options: 3, 5, or 10 minutes

# Training window (historical data used)
training_window = 15  # Options: 15 or 20 minutes

# Scenario
scenario = "deep_buoys"  # Options: 
                         # - "deep_buoys" (global)
                         # - "triangular_area" (3 buoys)
                         # - "localized" (5 buoys)

# Rogue wave threshold
rogue_threshold = 2.0  # H/Hs ratio (standard is 2.0)
```

### LSTM Training Parameters

```python
# In LSTM_Training.ipynb

# Model architecture (optimized defaults)
lstm_units = 10       # Units per LSTM layer
num_layers = 4        # Number of LSTM layers
dropout_rate = 0.05   # Dropout after final LSTM

# Training
batch_size = 64       # Batch size for training
max_epochs = 500      # Maximum epochs (early stopping active)
validation_split = 0.2  # 20% for validation

# Filenames
file_str = "RWs_H_g_2_tadv_5min_deep_buoys_rw_0.5"
```

## Quick Test After Training

```python
import tensorflow as tf
import numpy as np

# Load trained model
model = tf.keras.models.load_model('best_LSTM_[scenario]_checkpoint.model.keras')

# Load test data
data_test = np.load('RWs_H_g_2_tadv_5min_deep_buoys_test_rw_0.5.npz')
X_test = data_test['wave_data_test']
y_test = data_test['label_test']

# Make predictions
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)

# Calculate accuracy
accuracy = np.mean(predicted_classes == y_test)
print(f"Test Accuracy: {accuracy:.2%}")

# Expected output: ~67% for 5-min horizon on global data
```

## Understanding Performance by Scenario

### Global/Deep Buoys (Best Performance)
- **Training samples**: ~252,000
- **LSTM accuracy**: 67-72%
- **When to use**: Large dataset available
- **Training time**: 2-4 hours (GPU)

### Triangular Area (3 Buoys)
- **Training samples**: ~9,900
- **SVM accuracy**: 64%
- **LSTM accuracy**: 55%
- **When to use**: Limited data, cross-location testing
- **Training time**: 30-60 minutes

### Localized (5 Buoys)
- **Training samples**: ~8,094
- **SVM best**: Outperforms LSTM
- **When to use**: Regional forecasting
- **Training time**: 30-60 minutes

## Performance Benchmarks

| Scenario | Model | Accuracy | F1 Score | Training Time |
|----------|-------|----------|----------|---------------|
| Global, 5-min | LSTM | 67% | 0.67 | 2-4 hrs (GPU) |
| Global, 0-min | LSTM | 72% | 0.72 | 2-4 hrs (GPU) |
| Triangle, 3-min | SVM | 64% | 0.64 | 1-2 hrs |
| Triangle, 3-min | LSTM | 55% | 0.55 | 1 hr (GPU) |
| Localized | SVM | Best | Best | 1 hr |

## Troubleshooting

**Problem**: NPZ files not found
```bash
# Check file naming
ls RWs_H_g_2_tadv_*min_*.npz

# Ensure data prep completed successfully
# Re-run Data_Preparation_Generic.ipynb
```

**Problem**: CUDA/GPU errors
```python
# Force CPU usage if GPU issues
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Or install CPU-only TensorFlow
pip install tensorflow-cpu
```

**Problem**: Out of memory during training
```python
# Reduce batch size in LSTM_Training.ipynb
batch_size = 32  # From 64

# Or reduce data size
# Use subset of training data
```

**Problem**: Low accuracy (<50%)
```python
# Check data balance
print(np.bincount(label_train))
# Should be approximately equal

# Verify normalization
print(f"Data range: [{wave_data_train.min()}, {wave_data_train.max()}]")
# Should be roughly [-3, 3]

# Increase training window
training_window = 20  # Instead of 15 minutes
```

**Problem**: Model not improving
```python
# Check learning rate
# Reduce initial learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

# Increase patience for early stopping
keras.callbacks.EarlyStopping(patience=50)  # From 25

# Verify data quality
# Remove NaN values
data = np.nan_to_num(data, nan=0.0)
```

## Customization Examples

### Change Forecast Horizon

```python
# In Data_Preparation_Generic.ipynb
forecast_horizon = 10  # Predict 10 minutes ahead instead of 5

# Then update filename in LSTM_Training.ipynb
file_str = "RWs_H_g_2_tadv_10min_deep_buoys_rw_0.5"
```

### Increase Model Capacity

```python
# In LSTM_Training.ipynb
model_LSTM.add(keras.layers.LSTM(20, return_sequences=True))  # 20 instead of 10
model_LSTM.add(keras.layers.LSTM(20, return_sequences=True))
model_LSTM.add(keras.layers.LSTM(20, return_sequences=True))
model_LSTM.add(keras.layers.LSTM(20))
```

### Use Different Rogue Wave Threshold

```python
# In Data_Preparation_Generic.ipynb
# Standard: H/Hs > 2.0
# Alternative: H/Hs > 2.2 (stricter definition)
rogue_threshold = 2.2

# Or use crest height definition
# ηc/Hs > 1.25
```

### Train SVM Instead of LSTM

```python
# Add to Data_Preparation_Generic.ipynb or create new notebook
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

# Load data
data = np.load('RWs_H_g_2_tadv_5min_deep_buoys_rw_0.5.npz')
X_train = data['wave_data_train'].reshape(len(data['wave_data_train']), -1)
y_train = data['label_train']

# Train SVM
svm = SVC(kernel='rbf', C=1.0, gamma='scale')
svm.fit(X_train, y_train)

# Test
data_test = np.load('RWs_H_g_2_tadv_5min_deep_buoys_test_rw_0.5.npz')
X_test = data_test['wave_data_test'].reshape(len(data_test['wave_data_test']), -1)
y_test = data_test['label_test']

predictions = svm.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"SVM Accuracy: {accuracy:.2%}")
```

## Understanding Output Files

### NPZ Data Files
```python
import numpy as np
data = np.load('RWs_H_g_2_tadv_5min_deep_buoys_rw_0.5.npz')

# Contents:
# wave_data_train: (n_samples, time_steps, 1)
# label_train: (n_samples,) - Binary: 0 (no rogue) or 1 (rogue)
```

### Model Checkpoint
```python
# Saved LSTM model
model = tf.keras.models.load_model('best_LSTM_[scenario]_checkpoint.model.keras')

# Model architecture
model.summary()

# Make predictions
predictions = model.predict(test_data)
```

### Training Curves
- **Plot**: Training vs validation accuracy over epochs
- **X-axis**: Epoch number
- **Y-axis**: Accuracy (0-1)
- **Ideal**: Training and validation curves close together (no overfitting)

### Confusion Matrix
```
              Predicted
           No Rogue  Rogue
Actual No   TN       FP
       Yes  FN       TP

Good model: High TN and TP, Low FP and FN
```

## Next Steps

1. 📊 **Analyze results**: Review confusion matrices and accuracy plots
2. 🎯 **Tune hyperparameters**: Adjust LSTM units, layers, dropout
3. ⏱️ **Try different horizons**: Test 3, 5, 10-minute forecasts
4. 🌊 **Test on new locations**: Evaluate cross-location generalization
5. 🚀 **Deploy**: Integrate best model into warning system
6. 📈 **Collect feedback**: Validate with domain experts

## Performance Tips

### For Best Accuracy
- Use **20-minute training window** instead of 15
- Train on **global dataset** for LSTM
- Use **SVM for localized** scenarios (<10K samples)
- Try **shorter forecast horizons** (3-5 minutes)

### For Faster Training
- Reduce **batch size** to 32
- Use **fewer LSTM layers** (3 instead of 4)
- **Early stopping** usually triggers before 500 epochs
- Train on **subset** of data for prototyping

### For Better Generalization
- Include **diverse buoy locations** in training
- Use **SVM** for cross-location testing
- **Augment data** with different weather conditions
- **Ensemble models**: Combine LSTM + SVM predictions

## Need Help?

- Check main README_ROGUE_WAVE.md for detailed documentation
- Review RogueWave_Classification_Results.ipynb for expected outputs
- Verify data format matches NPZ structure
- Contact: schakr18@umd.edu

## Success Criteria

✅ **LSTM Global**: 67-72% accuracy  
✅ **SVM Localized**: 64% accuracy  
✅ **Training converges**: Loss decreases steadily  
✅ **No overfitting**: Val accuracy close to train accuracy  
✅ **Confusion matrix**: Balanced TP and TN  
✅ **F1 Score**: >0.60 for operational use  

---

**Pro Tip**: Start with the global dataset and 5-minute forecast horizon using LSTM for best initial results. Once working, experiment with different scenarios and models!
