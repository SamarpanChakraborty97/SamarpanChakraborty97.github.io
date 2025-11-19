# Quick Start Guide - Missing Wave Data Imputation

## Fast Setup (15 minutes)

### 1. Install Dependencies

```bash
# Option A: Using pip
pip install torch tensorflow numpy pandas matplotlib seaborn scipy scikit-learn jupyter

# Option B: Using conda (recommended)
conda create -n wave-impute python=3.9
conda activate wave-impute
pip install torch tensorflow numpy pandas matplotlib seaborn scipy scikit-learn jupyter
```

Or use requirements file:
```bash
pip install -r requirements.txt
```

### 2. Verify Installation

```python
import torch
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print(f"PyTorch version: {torch.__version__}")
print(f"TensorFlow version: {tf.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"GPU available (PyTorch): {torch.cuda.is_available()}")
print(f"GPU available (TensorFlow): {len(tf.config.list_physical_devices('GPU')) > 0}")
```

### 3. Obtain Data

**Option A: Use CDIP Public Data**
- Visit: https://cdip.ucsd.edu/
- Select buoy stations: 067 (San Nicholas), 071 (Harvest), 076 (Diablo Canyon)
- Download historical buoy data (NetCDF or ASCII format)
- Format: Wave elevation time series, 1.28 Hz sampling

**Option B: Use Preprocessed Data** (if available)
- Place data files in `data/` directory
- Required format: NumPy arrays or CSV files
- Columns: timestamp, wave_elevation, significant_wave_height

### 4. Quick Notebook Tour

**Start Here: Results Notebook** (5-10 minutes)

```bash
jupyter notebook Wave_Data_Imputation_Results.ipynb
```

This notebook provides:
- ✅ Complete methodology overview
- ✅ Performance comparisons across all models
- ✅ Visualizations and heatmaps
- ✅ Extreme wave event case study
- ✅ No training required - view results immediately

**Then Explore: Implementation Notebooks**

```bash
# LSTM Implementation
jupyter notebook Wave_data_imputation_using_LSTM.ipynb

# CNN+LSTM Implementation  
jupyter notebook Wave_data_imputation_using_CNN_LSTM.ipynb
```

---

## 📚 Notebook Execution Guide

### Recommended Order

```
1. Wave_Data_Imputation_Results.ipynb (START HERE ⭐)
   ├─> Understanding: Methodology and results
   ├─> Visualizations: Performance heatmaps, correlations
   ├─> Time: 5-10 minutes
   └─> Training: None required
   
2. Wave_data_imputation_using_LSTM.ipynb
   ├─> Implementation: LSTM architecture
   ├─> Code review: Data preprocessing, training loops
   ├─> Time: 15 minutes (review) OR 2-4 hours (training)
   └─> Best for: Understanding recurrent networks
   
3. Wave_data_imputation_using_CNN_LSTM.ipynb
   ├─> Implementation: CNN+LSTM with attention
   ├─> Code review: Hybrid architecture, attention mechanism
   ├─> Time: 15 minutes (review) OR 3-5 hours (training)
   └─> Best for: State-of-the-art architecture
```

---

## 🎯 Expected Results

### After Reviewing Results Notebook

✅ **Performance Understanding**:
- CNN+LSTM: 23% better than baseline
- LSTM: Competitive, especially for shallow water
- SSA: Baseline comparison method

✅ **Visual Insights**:
- Normalized performance heatmaps
- True vs predicted peak correlations
- Time series reconstruction examples

✅ **Key Takeaways**:
- Deep water → CNN+LSTM excels
- Shallow water → LSTM performs best
- Extreme waves → Neural networks significantly outperform SSA

### After Training Models (If You Choose To)

✅ **LSTM Model**:
- Training time: 2-4 hours (GPU) / 8-12 hours (CPU)
- MAE: ~0.25m for 5-minute gaps
- Correlation: r > 0.85

✅ **CNN+LSTM Model**:
- Training time: 3-5 hours (GPU) / 12-16 hours (CPU)
- MAE: ~0.20m for 5-minute gaps
- Correlation: r > 0.90

✅ **Saved Outputs**:
- Trained model weights (`.pth` or `.h5` files)
- Training curves (loss and metric plots)
- Imputation examples (comparison figures)
- Performance metrics (CSV or text files)

---

## 🔧 Configuration Quick Reference

### Data Preprocessing Parameters

```python
# In data preprocessing section

# Wave data parameters
sampling_rate = 1.28  # Hz (CDIP standard)
gap_duration = 300    # seconds (5 minutes)
n_samples_gap = int(gap_duration * sampling_rate)  # 384 samples for 5 min

# Training/test split
train_ratio = 0.8
validation_ratio = 0.1
test_ratio = 0.1

# Normalization
normalize_by = "significant_wave_height"  # Or "std_dev"
```

### LSTM Training Parameters

```python
# In Wave_data_imputation_using_LSTM.ipynb

# Model architecture
lstm_units_layer1 = 64
lstm_units_layer2 = 64
dense_units = 32
dropout_rate = 0.2

# Training hyperparameters
batch_size = 32
learning_rate = 0.001
max_epochs = 200
patience = 20  # Early stopping

# Data parameters
sequence_length = 150  # Input sequence length (time steps)
prediction_horizon = 77  # Output length (1 minute at 1.28 Hz)
```

### CNN+LSTM Training Parameters

```python
# In Wave_data_imputation_using_CNN_LSTM.ipynb

# CNN architecture
conv_filters = 32
kernel_size = 3
pool_size = 2

# LSTM architecture  
lstm_units_layer1 = 64
lstm_units_layer2 = 32

# Attention mechanism
attention_units = 64

# Training hyperparameters
batch_size = 32
learning_rate = 0.001
max_epochs = 250
patience = 20

# Data parameters
sequence_length = 150
prediction_horizon = 77  # For 1-min gaps (or 384 for 5-min)
```

---

## 🚀 Quick Test After Setup

### Test 1: Load and Visualize Data

```python
import numpy as np
import matplotlib.pyplot as plt

# Load sample wave data (modify path as needed)
wave_data = np.load('data/sample_wave_elevation.npy')
time = np.arange(len(wave_data)) / 1.28  # Convert to seconds

# Plot
plt.figure(figsize=(12, 4))
plt.plot(time, wave_data)
plt.xlabel('Time (seconds)')
plt.ylabel('Wave Elevation (m)')
plt.title('Sample Wave Time Series')
plt.grid(True, alpha=0.3)
plt.show()

print(f"Data shape: {wave_data.shape}")
print(f"Duration: {len(wave_data)/1.28:.1f} seconds")
print(f"Mean elevation: {np.mean(wave_data):.2f} m")
print(f"Std deviation: {np.std(wave_data):.2f} m")
```

### Test 2: Peak Detection

```python
from scipy.signal import find_peaks

# Detect peaks
peaks, _ = find_peaks(wave_data, distance=10, prominence=0.5)
troughs, _ = find_peaks(-wave_data, distance=10, prominence=0.5)

# Visualize
plt.figure(figsize=(12, 4))
plt.plot(time, wave_data, label='Wave elevation')
plt.plot(time[peaks], wave_data[peaks], 'ro', label='Peaks', markersize=6)
plt.plot(time[troughs], wave_data[troughs], 'bo', label='Troughs', markersize=6)
plt.xlabel('Time (seconds)')
plt.ylabel('Wave Elevation (m)')
plt.title('Peak and Trough Detection')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"Number of peaks: {len(peaks)}")
print(f"Number of troughs: {len(troughs)}")
print(f"Mean peak height: {np.mean(wave_data[peaks]):.2f} m")
```

### Test 3: Load Pre-trained Model (If Available)

```python
import torch

# Load PyTorch model (CNN+LSTM)
model = torch.load('models/cnn_lstm_best_model.pth')
model.eval()
print("Model loaded successfully!")
print(model)

# Or load TensorFlow model (LSTM)
import tensorflow as tf
model_tf = tf.keras.models.load_model('models/lstm_best_model.h5')
print("\nTensorFlow model loaded!")
model_tf.summary()
```

---

## 📊 Understanding the Results

### Performance Metrics Explained

**Mean Absolute Error (MAE)**:
```
MAE = mean(|predicted_peaks - true_peaks|)
```
- Lower is better
- Interpretation: Average prediction error in meters
- Good: MAE < 0.25m for 5-minute gaps
- Excellent: MAE < 0.15m for 1-minute gaps

**Mean Squared Error (MSE)**:
```
MSE = mean((predicted_peaks - true_peaks)²)
```
- Lower is better
- Penalizes large errors more heavily
- Good: MSE < 0.10 for 5-minute gaps

**Correlation Coefficient (r)**:
```
r = correlation(predicted_peaks, true_peaks)
```
- Range: -1 to 1 (higher is better)
- Good: r > 0.85
- Excellent: r > 0.90
- Perfect: r = 1.0

### Confusion Matrix for Classification (If Applicable)

```
              Predicted
           Normal  Extreme
Actual  
Normal     TN       FP      ← False alarms
Extreme    FN       TP      ← Missed detections
           ↑        ↑
         Miss    Success
```

---

## 🎨 Visualization Guide

### Plot 1: Normalized Performance Heatmap

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Performance data (example)
models = ['CNN+LSTM', 'LSTM', 'SSA', 'Baseline']
metrics = ['MAE', 'MSE', 'Correlation']

# Normalized scores (0=worst, 1=best for each metric)
data = np.array([
    [0.95, 0.93, 0.92],  # CNN+LSTM
    [0.85, 0.83, 0.87],  # LSTM
    [0.60, 0.58, 0.65],  # SSA
    [0.40, 0.38, 0.45]   # Baseline
])

# Plot
plt.figure(figsize=(8, 6))
sns.heatmap(data, annot=True, fmt='.2f', cmap='RdYlGn', 
            xticklabels=metrics, yticklabels=models,
            vmin=0, vmax=1, cbar_kws={'label': 'Normalized Score'})
plt.title('Model Performance Comparison (Higher = Better)')
plt.tight_layout()
plt.show()
```

### Plot 2: True vs Predicted Peaks

```python
# Example: Create scatter plot with correlation
predicted_peaks = model.predict(X_test)
true_peaks = y_test

plt.figure(figsize=(8, 8))
plt.scatter(true_peaks, predicted_peaks, alpha=0.5, s=20)
plt.plot([true_peaks.min(), true_peaks.max()], 
         [true_peaks.min(), true_peaks.max()], 
         'r--', lw=2, label='Perfect prediction')

# Calculate and display correlation
r = np.corrcoef(true_peaks, predicted_peaks)[0, 1]
plt.text(0.05, 0.95, f'r = {r:.3f}', 
         transform=plt.gca().transAxes, 
         fontsize=14, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.xlabel('True Peak Height (m)', fontsize=12)
plt.ylabel('Predicted Peak Height (m)', fontsize=12)
plt.title('Peak Imputation Performance', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.tight_layout()
plt.show()
```

### Plot 3: Time Series with Imputed Gap

```python
# Example: Visualize imputation results
gap_start = 500
gap_end = 577  # 77 samples = 1 minute gap

plt.figure(figsize=(14, 5))

# Plot original data
plt.plot(time[:gap_start], wave_data[:gap_start], 'b-', 
         label='Observed (before gap)', linewidth=1.5)
plt.plot(time[gap_end:gap_end+200], wave_data[gap_end:gap_end+200], 'b-', 
         label='Observed (after gap)', linewidth=1.5)

# Plot true values in gap (for comparison)
plt.plot(time[gap_start:gap_end], wave_data[gap_start:gap_end], 'k--', 
         label='True values', linewidth=1.5, alpha=0.7)

# Plot imputed values
plt.plot(time[gap_start:gap_end], imputed_values, 'r-', 
         label='CNN+LSTM imputation', linewidth=2)

# Shade gap region
plt.axvspan(time[gap_start], time[gap_end], alpha=0.2, color='gray', 
            label='Missing data region')

plt.xlabel('Time (seconds)', fontsize=12)
plt.ylabel('Wave Elevation (m)', fontsize=12)
plt.title('Wave Data Imputation Example (1-minute gap)', fontsize=14)
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Calculate error
mae = np.mean(np.abs(wave_data[gap_start:gap_end] - imputed_values))
print(f"MAE for this gap: {mae:.3f} m")
```

---

## 🔬 Model Comparison by Scenario

### Scenario 1: Deep Water, Short Gap

**Conditions**: 262m depth, 1-minute gap  
**Best model**: CNN+LSTM  
**Expected performance**: MAE < 0.15m, r > 0.92

```python
# Training configuration
scenario = "deep_water_short_gap"
water_depth = 262  # meters
gap_duration = 60  # seconds
sequence_length = 150
model_type = "CNN+LSTM"
```

### Scenario 2: Deep Water, Long Gap

**Conditions**: 550m depth, 5-minute gap  
**Best model**: CNN+LSTM  
**Expected performance**: MAE < 0.32m, r > 0.85

```python
scenario = "deep_water_long_gap"
water_depth = 550
gap_duration = 300
sequence_length = 200  # Longer context for long gaps
model_type = "CNN+LSTM"
```

### Scenario 3: Shallow Water

**Conditions**: 27m depth, 1-5 minute gaps  
**Best model**: LSTM  
**Expected performance**: MAE < 0.25m, r > 0.88

```python
scenario = "shallow_water"
water_depth = 27
gap_duration = 300
sequence_length = 150
model_type = "LSTM"
```

### Scenario 4: Extreme Wave Event

**Conditions**: High wave peak (>2.5× mean)  
**Best model**: CNN+LSTM  
**Expected performance**: Peak capture rate > 90%

```python
scenario = "extreme_wave"
rogue_wave_threshold = 2.5  # H/Hs ratio
model_type = "CNN+LSTM"
attention_enabled = True  # Critical for extreme events
```

---

## 🛠️ Troubleshooting

### Problem 1: Import Errors

```bash
# Error: "No module named 'torch'"
pip install torch

# Error: "No module named 'tensorflow'"
pip install tensorflow

# Error: "No module named 'scipy'"
pip install scipy scikit-learn
```

### Problem 2: CUDA/GPU Issues

```python
# Check GPU availability
import torch
print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
print(f"PyTorch CUDA version: {torch.version.cuda}")

# Force CPU usage if GPU issues
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Or install CPU-only versions
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install tensorflow-cpu
```

### Problem 3: Out of Memory

```python
# Reduce batch size
batch_size = 16  # From 32

# Reduce sequence length
sequence_length = 100  # From 150

# Use gradient accumulation
accumulation_steps = 2  # Effective batch = batch_size * accumulation_steps
```

### Problem 4: Poor Model Performance

**Checklist**:
```python
# 1. Check data normalization
print(f"Data mean: {X_train.mean():.3f}")
print(f"Data std: {X_train.std():.3f}")
# Should be close to mean=0, std=1

# 2. Check for NaN values
print(f"NaN values in training: {np.isnan(X_train).sum()}")
print(f"NaN values in labels: {np.isnan(y_train).sum()}")

# 3. Verify data shapes
print(f"X_train shape: {X_train.shape}")  # Should be (n_samples, seq_length, features)
print(f"y_train shape: {y_train.shape}")  # Should be (n_samples, prediction_horizon)

# 4. Check learning rate
# If loss not decreasing, reduce learning rate
learning_rate = 0.0001  # From 0.001

# 5. Increase training epochs
max_epochs = 300  # From 200
```

### Problem 5: Notebook Kernel Crashes

```bash
# Increase Jupyter memory limit
jupyter notebook --NotebookApp.max_buffer_size=1000000000

# Or use JupyterLab
jupyter lab --ResourceUseDisplay.mem_limit=8589934592  # 8GB
```

### Problem 6: Data Loading Issues

```python
# Check file exists
import os
data_path = 'data/wave_elevation.npy'
if os.path.exists(data_path):
    print(f"File found: {data_path}")
else:
    print(f"File not found: {data_path}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Available files: {os.listdir('data/')}")

# Try different loading methods
try:
    data = np.load(data_path)
except Exception as e:
    print(f"Error loading NPY: {e}")
    # Try CSV instead
    import pandas as pd
    data = pd.read_csv(data_path.replace('.npy', '.csv')).values
```

---

## 🎓 Understanding the Physics

### Wave Model Decomposition

The key innovation is decomposing waves into **slowly varying amplitudes**:

```
Original: η(t) = complex wave elevation
          ↓ (Wave model)
Decomposed: A₁(t), A₂(t), ..., Aₙ(t)  (slowly varying amplitudes)
          ↓ (Neural network learns patterns)
Predicted: A₁(t+Δt), A₂(t+Δt), ..., Aₙ(t+Δt)
          ↓ (Reconstruct)
Imputed: η(t+Δt) = reconstructed wave elevation
```

**Why this works**:
- Amplitudes vary more slowly than raw wave elevation
- Easier for neural networks to learn smooth functions
- Physics-based → better generalization

### Attention Mechanism

The attention layer learns which past time steps are most important:

```
Input sequence: [t-150, t-149, ..., t-1, t]
                   ↓
Attention weights: [0.02, 0.01, ..., 0.15, 0.35]  (sums to 1.0)
                   ↓
Weighted sum → Focuses on recent + relevant patterns
                   ↓
Better predictions for gaps
```

**When attention helps most**:
- Long input sequences (>100 time steps)
- Extreme wave events (non-stationary behavior)
- Deep water scenarios (more complex dynamics)

---

## 📈 Performance Optimization Tips

### For Best Accuracy

1. **Use longer training sequences**:
   ```python
   sequence_length = 200  # Instead of 150
   ```

2. **Increase model capacity** (if sufficient data):
   ```python
   lstm_units = 128  # Instead of 64
   num_layers = 3    # Instead of 2
   ```

3. **Enable attention** (for CNN+LSTM):
   ```python
   use_attention = True
   attention_units = 64
   ```

4. **Use appropriate model for scenario**:
   - Deep water → CNN+LSTM
   - Shallow water → LSTM
   - Short gaps (<2 min) → Either model
   - Long gaps (>3 min) → CNN+LSTM

### For Faster Training

1. **Reduce batch size** (paradoxically can speed up):
   ```python
   batch_size = 16  # Faster gradient updates
   ```

2. **Use learning rate scheduling**:
   ```python
   from torch.optim.lr_scheduler import ReduceLROnPlateau
   scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
   ```

3. **Early stopping**:
   ```python
   early_stopping_patience = 15  # Stop if no improvement for 15 epochs
   ```

4. **Mixed precision training** (GPU only):
   ```python
   from torch.cuda.amp import autocast, GradScaler
   scaler = GradScaler()
   ```

### For Better Generalization

1. **Data augmentation**:
   ```python
   # Add noise to training data
   noise_level = 0.01
   X_train_augmented = X_train + np.random.normal(0, noise_level, X_train.shape)
   ```

2. **Dropout regularization**:
   ```python
   dropout_rate = 0.3  # Increase from 0.2
   ```

3. **Cross-validation**:
   ```python
   from sklearn.model_selection import KFold
   kfold = KFold(n_splits=5, shuffle=True)
   ```

4. **Ensemble predictions**:
   ```python
   # Train multiple models, average predictions
   predictions = (model1.predict(X) + model2.predict(X) + model3.predict(X)) / 3
   ```

---

## 🔄 Workflow Examples

### Example 1: Quick Analysis (No Training)

```bash
# Time: 10 minutes

# 1. Open Results notebook
jupyter notebook Wave_Data_Imputation_Results.ipynb

# 2. Run all cells (Kernel → Restart & Run All)

# 3. Review:
#    - Performance heatmaps
#    - Model comparisons
#    - Extreme wave case study
```

### Example 2: Train LSTM Model

```bash
# Time: 2-4 hours (GPU) or 8-12 hours (CPU)

# 1. Prepare data (if not done)
python prepare_data.py --buoy 067 --gap-duration 300

# 2. Train LSTM
jupyter notebook Wave_data_imputation_using_LSTM.ipynb
# Configure parameters, run training cells

# 3. Evaluate
python evaluate_model.py --model lstm --test-data data/test.npz
```

### Example 3: Train CNN+LSTM with Custom Configuration

```bash
# Time: 3-5 hours (GPU)

# 1. Modify configuration in notebook
jupyter notebook Wave_data_imputation_using_CNN_LSTM.ipynb

# In notebook, set:
sequence_length = 200
batch_size = 32
learning_rate = 0.0005
attention_units = 128

# 2. Run training

# 3. Save best model
torch.save(model.state_dict(), 'models/custom_cnn_lstm.pth')
```

### Example 4: Cross-Validation Study

```python
# Compare performance across multiple buoys

buoys = ['067', '071', '076']
results = {}

for buoy in buoys:
    # Load data
    data = load_buoy_data(buoy)
    
    # Train model
    model = train_cnn_lstm(data, epochs=200)
    
    # Evaluate
    mae, r = evaluate_model(model, data['test'])
    results[buoy] = {'MAE': mae, 'correlation': r}
    
    print(f"Buoy {buoy}: MAE={mae:.3f}, r={r:.3f}")

# Compare results
import pandas as pd
results_df = pd.DataFrame(results).T
print(results_df)
results_df.to_csv('cross_validation_results.csv')
```

---

## 📚 Additional Resources

### Documentation
- Main README: [README.md](README.md)
- Project Structure: [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)
- Research Paper: [Ocean Engineering](https://www.sciencedirect.com/science/article/pii/S0029801824034838)

### External Resources
- CDIP Data Portal: https://cdip.ucsd.edu/
- PyTorch Documentation: https://pytorch.org/docs/
- TensorFlow Guide: https://www.tensorflow.org/guide
- Wave Physics Tutorial: [CDIP Wave Physics](https://cdip.ucsd.edu/themes/wave_heights/)

### Related Projects
- SSA Implementation: [GitHub](https://github.com/SamarpanChakraborty97/Missing-wave-data-imputation-/tree/main/Singular%20Spectrum%20Analysis)
- Rogue Wave Forecasting: Contact for details

---

## ✅ Success Checklist

Before considering your setup complete:

- [ ] All dependencies installed and verified
- [ ] Jupyter notebooks open without errors
- [ ] Results notebook runs completely
- [ ] Sample visualizations display correctly
- [ ] Understand performance metrics (MAE, MSE, correlation)
- [ ] Can load and plot wave data
- [ ] Familiar with notebook structure and workflow

Before starting training:

- [ ] Sufficient data obtained (from CDIP or other source)
- [ ] Data properly formatted (time series, peaks extracted)
- [ ] GPU available (recommended) or patient for CPU training
- [ ] Training parameters configured appropriately
- [ ] Output directories created
- [ ] Enough disk space for model checkpoints (~500MB per model)

After training completes:

- [ ] Training converged (loss decreased steadily)
- [ ] Validation loss close to training loss (no severe overfitting)
- [ ] Model saved successfully
- [ ] Performance metrics meet expectations
- [ ] Visualizations generated
- [ ] Results documented

---

## 🎯 Next Steps

1. **Understand the Results** → Start with Wave_Data_Imputation_Results.ipynb
2. **Explore Implementation** → Review LSTM and CNN+LSTM notebooks
3. **Experiment** → Modify hyperparameters, try different configurations
4. **Apply to New Data** → Test on different buoy locations or conditions
5. **Compare Models** → Train both LSTM and CNN+LSTM, compare performance
6. **Read the Paper** → Deep dive into methodology and theory
7. **Extend the Work** → Try longer gaps, real-time implementation, etc.

---

## 💬 Getting Help

**Quick questions?**  
→ Check [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)

**Technical issues?**  
→ Review troubleshooting section above

**Research questions?**  
→ Read the [Ocean Engineering paper](https://www.sciencedirect.com/science/article/pii/S0029801824034838)

**Want to collaborate?**  
→ Email: schakr18@umd.edu

---

**Pro Tip**: The Results notebook is designed to run without any training - perfect for quickly understanding the project's outcomes before diving into implementation details!

**Last Updated**: November 2024  
**Version**: 1.0.0