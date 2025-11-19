# Project Structure

## Repository Organization

```
missing-wave-data-imputation/
├── Wave_data_imputation_using_LSTM.ipynb
├── Wave_data_imputation_using_CNN_LSTM.ipynb
├── Wave_Data_Imputation_Results.ipynb
├── README.md
└── PROJECT_STRUCTURE.md
```

---

## File Descriptions

### `Wave_data_imputation_using_LSTM.ipynb`

**Purpose**: Implementation of Long Short-Term Memory networks for wave data imputation

**Contents**:
- Data loading and preprocessing from CDIP buoy sources
- Peak and trough extraction from wave time series
- Wave model implementation for amplitude decomposition
- LSTM network architecture definition and training
- Model evaluation on different gap durations (1-5 minutes)
- Testing across multiple buoy locations

**Key Features**:
- Handles variable-length gaps in time series
- Optimized for temporal sequence modeling
- Strong performance in shallow water scenarios (27m depth)
- Outperforms traditional SSA methods

**Technologies**: TensorFlow, NumPy, pandas, Matplotlib

---

### `Wave_data_imputation_using_CNN_LSTM.ipynb`

**Purpose**: Hybrid deep learning architecture combining CNN and LSTM with attention mechanism

**Contents**:
- Enhanced data preprocessing pipeline
- 1D Convolutional layer implementation for spatial feature extraction
- LSTM layer integration for temporal modeling
- Attention mechanism for adaptive feature weighting
- Hyperparameter tuning and optimization
- Cross-validation across multiple test scenarios

**Architecture Highlights**:
1. **CNN layers**: Extract local patterns from amplitude sequences
2. **LSTM layers**: Capture long-range temporal dependencies
3. **Attention layer**: Dynamically weight important time steps
4. **Dense layers**: Final prediction outputs

**Performance**:
- 23% accuracy improvement over baseline
- Best overall performer for deep water scenarios (262m, 550m depths)
- Successfully captures extreme wave events within data gaps
- Higher correlation coefficients than standalone LSTM and SSA

**Technologies**: PyTorch, NumPy, pandas, Matplotlib, Seaborn

---

### `Wave_Data_Imputation_Results.ipynb`

**Purpose**: Comprehensive comparison and visualization of all imputation methods

**Contents**:

#### Workflow Visualization
- Complete imputation process diagram
- Data flow from extraction to evaluation

#### Model Comparisons
**Models Evaluated**:
- Singular Spectrum Analysis (SSA) - Linear baseline
- LSTM - Recurrent neural network
- CNN+LSTM with Attention - Hybrid architecture
- Baseline - Mean amplitude fitting

**Performance Metrics**:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Correlation coefficients
- Normalized heatmaps for cross-model comparison

#### Test Scenarios
**Gap Durations**: 1 minute, 5 minutes

**Buoy Locations**:
- San Nicholas Island (262m depth)
- Harvest (550m depth)
- Diablo Canyon (27m depth)

#### Key Findings

**1-Minute Gaps**:
- CNN+LSTM consistently outperformed all models across all metrics
- Neural networks significantly exceeded SSA performance
- Fitting errors validate physics-based wave model accuracy

**5-Minute Gaps**:
- Deep water: CNN+LSTM achieved best performance
- Shallow water: LSTM showed optimal results
- Model performance varies by water depth and wave characteristics

**Extreme Events**:
- Case study: High wave event imputation at Diablo Canyon
- CNN+LSTM successfully captured wave peaks significantly larger than surrounding amplitudes
- Demonstrates robustness for anomaly prediction

#### Visualizations
- Normalized performance heatmaps across models and metrics
- True vs. predicted peak correlations with density plots
- Time series imputation examples with error bands
- High wave event case study with comparative predictions

**Technologies**: pandas, Matplotlib, Seaborn

---

## Data Sources

All wave measurements sourced from [CDIP (Coastal Data Information Program)](https://cdip.ucsd.edu/):
- Real-time and historical ocean wave data
- Buoy measurements at varying water depths
- High-frequency sampling rates suitable for detailed analysis

## Related Publications

**Primary Reference**:  
S. Chakraborty, K. Ide, and B. Balachandran, "Missing values imputation in ocean buoy time series data", *Ocean Engineering*, v. 315, 2025, pp. 120145.

**SSA Implementation Details**:  
Available at [GitHub Repository](https://github.com/SamarpanChakraborty97/Missing-wave-data-imputation-/tree/main/Singular%20Spectrum%20Analysis)
