# Quick Start Guide - GNN Rogue Wave Forecasting

## Fast Setup (30 minutes)

### 1. Install PyTorch with CUDA

```bash
# Install PyTorch (CUDA 11.3 example)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu113

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### 2. Install PyTorch Geometric

```bash
# Install PyTorch Geometric and dependencies
pip install torch-geometric torch-scatter torch-sparse torch-cluster

# Or install from wheels (if above fails)
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
pip install torch-geometric
```

### 3. Install Other Dependencies

```bash
pip install numpy matplotlib seaborn scikit-learn networkx scipy tensorflow
```

Or use requirements file:
```bash
pip install -r requirements_gnn_roguewave.txt
```

### 4. Verify Installation

```python
import torch
import torch_geometric
from torch_geometric.nn import GINConv

print(f"PyTorch: {torch.__version__}")
print(f"PyG: {torch_geometric.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
# Expected: All imports successful, CUDA available = True
```

### 5. Run the Pipeline

**Quick Training (Use PCA version for speed)**

```bash
# Step 1: Prepare wave group data
jupyter notebook Data_Preparation_different_wave_groups.ipynb
# Run all cells (~1-2 hours for data extraction)

# Step 2: Train GNN with PCA (faster)
jupyter notebook GraphLevelClassification_PCA_GIN.ipynb
# Run all cells (~30-60 minutes)

# Expected: >70% accuracy for largest wave groups
```

**Full Pipeline (Best accuracy)**

```bash
# Step 1: Data preparation
jupyter notebook Data_Preparation_different_wave_groups.ipynb

# Step 2: Train GNN on original data
jupyter notebook GraphLevelClassification_OriginalLength_GIN.ipynb
# Run all cells (~2-4 hours)

# Expected: >72% accuracy for largest wave groups (Group 4)
```

## Expected Results

After running the complete pipeline:

✅ **GNN Performance (Group 4 - Largest Waves)**:
- Accuracy: >72%
- Improvement over LSTM: +5-7%
- Training time: 2-4 hours (GPU)

✅ **GNN with PCA**:
- Accuracy: 70-72% (only ~2% drop)
- Training time: 30-60 minutes (80% faster)
- Memory usage: 67% reduction

✅ **Generated Outputs**:
- Wave group NPZ files (4 groups)
- Trained GNN model checkpoints
- Training curves (loss and accuracy)
- Confusion matrices
- Performance metrics

## Notebook Execution Order

```
1. Data_Preparation_different_wave_groups.ipynb (REQUIRED)
   ├─> Segregates waves into magnitude groups
   ├─> Groups: 2.0-2.5, 2.5-3.0, 3.0-3.5, ≥3.5 (H/Hs)
   └─> Generates: tadv_5min_wave_group_window_15mins_[1-4].npz
   
2A. GraphLevelClassification_OriginalLength_GIN.ipynb (Best accuracy)
   ├─> Converts time series to graphs
   ├─> Trains GIN with skip connections
   └─> Achieves: >72% accuracy (Group 4)
   
2B. GraphLevelClassification_PCA_GIN.ipynb (Faster alternative)
   ├─> Applies PCA dimensionality reduction
   ├─> Trains GIN on smaller graphs
   └─> Achieves: 70-72% accuracy (80% faster)
```

## Key Files

| File | Purpose | Time | Required |
|------|---------|------|----------|
| `Data_Preparation_different_wave_groups.ipynb` | Wave segregation | 1-2 hrs | Always |
| `GraphLevelClassification_OriginalLength_GIN.ipynb` | GNN training | 2-4 hrs | Best accuracy |
| `GraphLevelClassification_PCA_GIN.ipynb` | GNN+PCA | 30-60 min | Faster option |

## Configuration Quick Reference

### Data Preparation Parameters

```python
# In Data_Preparation_different_wave_groups.ipynb

# Wave group thresholds (H/Hs ratios)
wave_groups = {
    1: (2.0, 2.5),   # Moderate rogue waves
    2: (2.5, 3.0),   # Strong rogue waves
    3: (3.0, 3.5),   # Extreme rogue waves
    4: (3.5, inf)    # Largest rogue waves (best performance)
}

# Forecast horizon and window
forecast_horizon = 5   # minutes
training_window = 15   # minutes

# Output directory
output_dir = "wave_groups/"
```

### GNN Architecture Parameters

```python
# In GraphLevelClassification notebooks

# Graph construction
num_neighbours = 70    # K-NN edges (70 for original, 20 for PCA)

# GIN architecture
dim_pre_MLP = 64       # Pre-processing dimension
dim_post_MLP = 32      # Post-processing dimension
num_graph_layers = 3   # Number of GIN layers
dropout_prob = 0.3     # Dropout rate

# Training
batch_size = 32
learning_rate = 5e-4
weight_decay = 0.01
patience = 10-20       # Early stopping patience

# Wave group selection
file_str = "tadv_5min_wave_group_window_15mins_4"  # Group 4 = best results
```

### PCA Configuration

```python
# In GraphLevelClassification_PCA_GIN.ipynb

# PCA parameters
n_components = 100     # Reduce from ~900 to 100 dimensions
# Achieves >95% explained variance

# Adjusted graph construction
num_neighbours = 20    # Fewer neighbors for smaller graphs
```

## Quick Test After Training

```python
import torch
from torch_geometric.loader import DataLoader

# Load trained model
model = GraphIsomorphismNetworkWithSkipConnections(
    dim_pre_MLP=64,
    dim_post_MLP=32,
    dim_graphLin=2,
    num_pre_layers=2,
    num_post_layers=2,
    dropout_prob=0.3,
    num_graph_layers=3,
    training=False
)

model.load_state_dict(torch.load('best_model_wave_group_4.pt'))
model.eval()

# Load test data (graph objects)
test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)

# Evaluate
correct = 0
total = 0

with torch.no_grad():
    for batch in test_loader:
        outputs = model(batch.x, batch.edge_index, batch.edge_weight, batch.batch)
        predictions = (outputs > 0.5).float()
        correct += (predictions == batch.y).sum().item()
        total += batch.y.size(0)

accuracy = correct / total
print(f"Test Accuracy: {accuracy:.2%}")
# Expected: >72% for Group 4 (largest waves)
```

## Understanding Wave Groups

### Performance by Wave Group

| Group | H/Hs Range | Severity | GNN Accuracy | LSTM Accuracy | Improvement |
|-------|------------|----------|--------------|---------------|-------------|
| 1 | 2.0-2.5 | Moderate | 68-70% | 65-67% | +3-5% |
| 2 | 2.5-3.0 | Strong | 70-71% | 66-68% | +3-4% |
| 3 | 3.0-3.5 | Extreme | 71-73% | 67-69% | +3-4% |
| **4** | **≥3.5** | **Largest** | **>72%** | **65-67%** | **+5-7%** |

### Why Group 4 Performs Best

- **Clearer patterns**: Largest waves have more distinct formation signatures
- **Graph benefit**: Complex temporal dependencies captured better
- **Critical importance**: Most dangerous waves receive best predictions
- **Practical value**: Where accuracy matters most for safety

## Troubleshooting

**Problem**: PyTorch Geometric installation errors
```bash
# Solution 1: Install from wheels
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.11.0+cu113.html

# Solution 2: Build from source (if wheels unavailable)
pip install git+https://github.com/pyg-team/pytorch_geometric.git

# Solution 3: Use conda
conda install pyg -c pyg
```

**Problem**: CUDA out of memory
```python
# Solution 1: Use PCA version
# Switch to GraphLevelClassification_PCA_GIN.ipynb

# Solution 2: Reduce batch size
batch_size = 16  # From 32

# Solution 3: Fewer neighbors
num_neighbours = 50  # From 70

# Solution 4: Use CPU (slower)
device = torch.device('cpu')
```

**Problem**: Low accuracy (<60%)
```python
# Check data balance
print(f"Rogue waves: {sum(labels)}")
print(f"Non-rogue: {len(labels) - sum(labels)}")
# Should be approximately equal

# Verify wave group
# Group 4 typically performs best (>72%)
# Try different groups if using Group 1 or 2

# Check normalization
print(f"Data range: [{data.min()}, {data.max()}]")
# Should be roughly [-3, 3] after normalization

# Increase training epochs
num_epochs = 150  # From default
```

**Problem**: Training not converging
```python
# Reduce learning rate
learning_rate = 1e-4  # From 5e-4

# Add gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Check for NaN values
if torch.isnan(loss):
    print("NaN detected in loss!")
    # Check data preprocessing
```

**Problem**: Graph construction too slow
```python
# Use vectorized k-NN
import torch_cluster
edge_index = torch_cluster.knn_graph(node_features, k=num_neighbours)

# Or use PCA version (smaller graphs)
# GraphLevelClassification_PCA_GIN.ipynb
```

## Customization Examples

### Change Wave Group

```python
# Train on different wave magnitude ranges
file_str = "tadv_5min_wave_group_window_15mins_1"  # Moderate waves (2.0-2.5)
file_str = "tadv_5min_wave_group_window_15mins_2"  # Strong waves (2.5-3.0)
file_str = "tadv_5min_wave_group_window_15mins_3"  # Extreme waves (3.0-3.5)
file_str = "tadv_5min_wave_group_window_15mins_4"  # Largest waves (≥3.5) ⭐ Best
```

### Adjust Graph Structure

```python
# More/fewer neighbors
num_neighbours = 100  # Larger receptive field (from 70)
num_neighbours = 40   # Smaller receptive field

# Different edge weighting
def gaussian_edge_weight(distance, sigma=10):
    return np.exp(-(distance**2) / (2 * sigma**2))
```

### Modify GIN Architecture

```python
# Deeper network
num_graph_layers = 5  # From 3

# Wider layers
dim_pre_MLP = 128     # From 64
dim_post_MLP = 64     # From 32

# More aggressive regularization
dropout_prob = 0.5    # From 0.3
weight_decay = 0.1    # From 0.01
```

### PCA Component Selection

```python
# More components (more detail)
n_components = 150    # From 100
# Slower but potentially higher accuracy

# Fewer components (faster)
n_components = 50     # From 100
# Faster but may lose important information
```

## Performance Optimization Tips

### For Best Accuracy
- Use **original data** (not PCA) version
- Train on **Wave Group 4** (largest waves)
- Use **70 neighbors** for graph construction
- Train for **100-150 epochs** with early stopping
- Use **skip connections** in GIN architecture

### For Fastest Training
- Use **PCA version** (80% faster)
- Use **20 neighbors** (smaller graphs)
- **Batch size 64** (if GPU memory allows)
- Train on **Wave Group 4** (converges faster)

### For Limited GPU Memory
- Use **PCA version** automatically reduces memory
- Reduce **batch size to 16**
- Reduce **num_neighbours to 40-50**
- Use **gradient checkpointing** (if implemented)

## Understanding Output Files

### NPZ Wave Group Files
```python
import numpy as np
data = np.load('wave_groups/tadv_5min_wave_group_window_15mins_4.npz')

# Contents:
# wave_data_train: (n_samples, time_steps, 1)
# wave_data_test: (n_test, time_steps, 1)
# label_train: (n_samples,) - Binary: 0 or 1
# label_test: (n_test,) - Binary: 0 or 1
```

### PyTorch Geometric Graph Objects
```python
# Each graph contains:
graph.x           # Node features (time_steps, 1)
graph.edge_index  # Edge connections (2, num_edges)
graph.edge_weight # Edge weights (num_edges,)
graph.y           # Graph label (1,) - Binary
graph.batch       # Batch assignment for each node
```

### Model Checkpoint
```python
# Saved GIN model
checkpoint = torch.load('best_model_wave_group_4.pt')
model.load_state_dict(checkpoint)

# Model contains:
# - Learned GIN layer weights
# - Pre/post MLP parameters
# - Batch normalization statistics
```

## Next Steps

1. 📊 **Analyze results**: Review training curves and confusion matrices
2. 🎯 **Try different wave groups**: Compare performance across Groups 1-4
3. 🔄 **Experiment with architecture**: Adjust layers, dimensions, dropout
4. 🌐 **Multi-buoy graphs**: Extend to spatial GNNs across locations
5. 🚀 **Deploy**: Integrate best model (Group 4) into warning system
6. 📈 **Research**: Publish findings on GNN for oceanographic forecasting

## Comparison with Other Methods

| Method | Accuracy (Group 4) | Training Time | Memory | Best For |
|--------|-------------------|---------------|--------|----------|
| **GNN (Original)** | **>72%** | 2-4 hrs | High | Best accuracy |
| **GNN (PCA)** | **70-72%** | 30-60 min | Low | Fast training |
| LSTM | 65-67% | 2-3 hrs | Medium | Sequential baseline |
| SVM | 64% | 1-2 hrs | Low | Limited data |

## Need Help?

- Check main README_GNN_ROGUE_WAVE.md for detailed documentation
- Verify PyTorch and PyG versions match
- Ensure wave group data is balanced (50-50 split)
- Contact: schakr18@umd.edu

## Success Criteria

✅ **Accuracy**: >72% on Wave Group 4 test set  
✅ **Training converges**: Loss decreases steadily  
✅ **No overfitting**: Val loss doesn't increase  
✅ **Graph construction**: Successfully creates k-NN graphs  
✅ **GPU utilization**: CUDA operational for faster training  
✅ **Improvement over LSTM**: +5-7% accuracy gain  

---

**Pro Tip**: Start with Wave Group 4 (largest waves) and the PCA version for quick initial results. Once validated, train on original data for maximum accuracy!
