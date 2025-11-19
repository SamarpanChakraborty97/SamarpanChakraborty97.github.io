# Missing Wave Data Imputation

A deep learning approach to impute missing wave peaks and troughs in ocean buoy time series data using neural networks and statistical methods.

## Overview

This project addresses the challenge of filling gaps in ocean wave measurements by decomposing wave surface elevation into slowly varying amplitudes using a physics-based wave model, then applying machine learning techniques for imputation.

## Methodology

**Data Processing Pipeline:**
1. Extract peaks and troughs from wave time series
2. Obtain slowly varying amplitudes via wave model optimization
3. Impute missing data using neural networks (extrapolation) or SSA (interpolation)

**Models Implemented:**

- **LSTM**: Long Short-Term Memory network for temporal sequence modeling
  - Captures long-term dependencies in wave amplitude evolution
  - Trained on historical amplitude sequences to predict future values
  - Best performer for shallow water scenarios

- **CNN+LSTM with Attention**: Hybrid architecture combining spatial and temporal features
  - CNN layers extract local spatial patterns from amplitude sequences
  - LSTM layers model temporal dependencies
  - Attention mechanism dynamically weights important time steps
  - 23% improvement over baseline methods
  - Best overall performer for deep water scenarios

- **Singular Spectrum Analysis (SSA)**: Linear statistical baseline method
  - Traditional interpolation approach for comparison

## Key Results

### Performance Across Gap Durations

**1-Minute Gaps (San Nicholas Island, 262m depth):**
- CNN+LSTM outperformed all models across MAE, MSE, and correlation metrics
- Both neural networks significantly exceeded SSA and baseline performance
- Higher correlation coefficients demonstrated better prediction accuracy

**5-Minute Gaps (Multiple Locations):**
- **Deep water (Harvest, 550m)**: CNN+LSTM achieved best performance across all metrics
- **Shallow water (Diablo Canyon, 27m)**: LSTM showed best results
- Neural networks successfully captured extreme wave events within gaps
- Successfully imputed wave peaks significantly larger than surrounding amplitudes

### Accuracy Improvements
- 23% improvement over baseline using attention-based CNN+LSTM architecture
- Successfully predicted wave anomalies exceeding surrounding amplitudes
- Robust performance across different water depths and wave characteristics

## Dataset

Wave buoy measurements from CDIP (Coastal Data Information Program):
- [San Nicholas Island (Buoy 067)](https://cdip.ucsd.edu/m/products/?stn=067p1) - 262m depth
- [Harvest (Buoy 071)](https://cdip.ucsd.edu/m/products/?stn=071p1) - 550m depth  
- [Diablo Canyon (Buoy 076)](https://cdip.ucsd.edu/m/products/?stn=076p1) - 27m depth

Gap durations tested: 1 minute, 5 minutes

## Technologies

- **Deep Learning**: PyTorch, TensorFlow
- **Data Processing**: NumPy, pandas, scikit-learn
- **Visualization**: Matplotlib, Seaborn
- **Wave Modeling**: MATLAB

## Project Notebooks

- `Wave_data_imputation_using_LSTM.ipynb` - LSTM implementation and training
- `Wave_data_imputation_using_CNN_LSTM.ipynb` - CNN+LSTM with attention architecture
- `Wave_Data_Imputation_Results.ipynb` - Comparative analysis, visualizations, and results

See `PROJECT_STRUCTURE.md` for detailed file descriptions.

## Citation

For detailed methodology and results, see:

S. Chakraborty, K. Ide, and B. Balachandran, "Missing values imputation in ocean buoy time series data", *Ocean Engineering*, v. 315, 2025, pp. 120145.  
[Article Link](https://www.sciencedirect.com/science/article/pii/S0029801824034838)

## Author

**Samarpan Chakraborty**  
University of Maryland, College Park  
[LinkedIn](https://www.linkedin.com/in/samarpan-chakraborty) | [Portfolio](https://samarpanchakraborty97.github.io)
