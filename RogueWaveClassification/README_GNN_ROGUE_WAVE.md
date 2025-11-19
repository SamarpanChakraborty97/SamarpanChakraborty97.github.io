# 🌊 Graph Neural Networks for Rogue Wave Forecasting

An advanced machine learning framework using Graph Neural Networks (GNNs) with Graph Isomorphism Networks (GIN) to predict extreme ocean waves. The system achieves >72% accuracy for the largest wave groups by converting time series data into graph structures and leveraging spatial-temporal relationships, demonstrating significant improvements over traditional sequential models for wave magnitude-specific forecasting.

## 📋 Project Overview

This project extends previous rogue wave forecasting work by employing Graph Neural Networks to capture complex spatial-temporal dependencies in ocean wave data. Unlike traditional LSTM approaches that treat time series as sequences, this method constructs graphs where each time step becomes a node, and edges represent temporal proximity, enabling the network to learn richer representations of wave dynamics.

### 🎯 Problem Statement

**Advanced Rogue Wave Forecasting:**
- **Challenge**: Traditional sequential models (LSTM, RNN) may miss complex temporal patterns
- **Innovation**: Graph structure captures non-sequential relationships between time steps
- **Focus**: Magnitude-specific models for different wave height ranges
- **Goal**: Improve accuracy by leveraging graph-based deep learning

**Rogue Wave Context:**
- Wave height (H) > 2.0 × Significant wave height (Hs)
- Devastating impact on maritime operations
- Prediction horizon: 5 minutes in advance
- Training window: 15 minutes of historical data

**Key Innovation**: Converting 1D time series into graph structures enables GNNs to capture:
- **Local patterns**: Neighboring time steps
- **Global context**: Overall wave sequence structure
- **Non-linear relationships**: Complex temporal dependencies
- **Hierarchical features**: Multi-scale wave patterns

### 💡 Solution

A specialized GNN framework that:
1. **Segregates waves by magnitude** into different groups for targeted modeling
2. **Converts time series to graphs** where nodes = time steps, edges = temporal proximity
3. **Applies Graph Isomorphism Networks** (GIN) for powerful representation learning
4. **Uses skip connections** for better gradient flow and feature reuse
5. **Implements PCA variants** for dimensionality reduction and efficiency
6. **Achieves >72% accuracy** for largest wave groups (significant improvement)

### ✨ Key Features

- **Graph-Based Architecture**: Novel application of GNNs to oceanographic time series
- **Wave Group Segregation**: Specialized models for different wave magnitude ranges
- **Graph Isomorphism Networks**: Powerful GIN layers for expressive representations
- **Flexible Graph Construction**: K-nearest neighbors approach for edge creation
- **Skip Connections**: Enhanced gradient flow and feature propagation
- **PCA Integration**: Efficient dimensionality reduction without sacrificing accuracy
- **PyTorch Geometric**: State-of-the-art GNN implementation
- **High Accuracy**: >72% for largest wave groups, competitive across all groups

## 📁 Project Structure

The project consists of three main components implementing a complete GNN-based forecasting pipeline:

### 1. 📊 Wave Group Segregation and Data Preparation

#### `Data_Preparation_different_wave_groups.ipynb`

Comprehensive data preparation with wave magnitude-based segregation:

**Wave Group Classification:**

The project segregates rogue waves into different groups based on their magnitude:

```
Wave Groups (based on normalized height H/Hs):
├─ Group 1: 2.0 ≤ H/Hs < 2.5  (Moderate rogue waves)
├─ Group 2: 2.5 ≤ H/Hs < 3.0  (Strong rogue waves)
├─ Group 3: 3.0 ≤ H/Hs < 3.5  (Extreme rogue waves)
└─ Group 4: H/Hs ≥ 3.5        (Largest/most extreme rogue waves)
```

**Motivation for Wave Groups:**
- Different magnitude waves may have different formation patterns
- Specialized models can capture group-specific dynamics
- Improves forecasting accuracy through targeted learning
- Enables magnitude-aware early warning systems

**Data Processing Pipeline:**

**1. Zero-Crossing Wave Extraction:**
```python
def find_max_wave_height(zdisp_window):
    """
    Extract individual waves using zero-crossing method
    - Identifies wave peaks and troughs
    - Calculates wave heights (H)
    - Returns maximum wave in window
    """
```

**2. Significant Wave Height Normalization:**
- Calculate Hs = 4 × std(sea_surface_elevation)
- Normalize all waves: H_norm = H / Hs
- Classify into wave groups based on H_norm

**3. Window Creation:**
- **Training window**: 15 minutes (t_window = 15 min)
- **Forecast horizon**: 5 minutes (t_horizon = 5 min)
- **Balanced dataset**: Equal rogue/non-rogue per group
- **Group-specific files**: Separate NPZ files for each wave group

**Output Structure:**
```
wave_groups/
├── tadv_5min_wave_group_window_15mins_1.npz  (Group 1: 2.0-2.5)
├── tadv_5min_wave_group_window_15mins_2.npz  (Group 2: 2.5-3.0)
├── tadv_5min_wave_group_window_15mins_3.npz  (Group 3: 3.0-3.5)
└── tadv_5min_wave_group_window_15mins_4.npz  (Group 4: ≥3.5)

Each file contains:
    - wave_data_train: Training time series
    - wave_data_test: Test time series
    - label_train: Binary labels (0=no rogue, 1=rogue)
    - label_test: Binary labels for testing
```

### 2. 🧠 Graph Isomorphism Network (Original Data)

#### `GraphLevelClassification_OriginalLength_GIN.ipynb`

GNN implementation using full-resolution time series data:

**Graph Construction:**

**Time Series to Graph Conversion:**
```
Original Time Series: [t₁, t₂, t₃, ..., tₙ]
                      ↓
Graph Structure:
    Nodes: Each time step tᵢ becomes a node
    Node Features: Normalized wave height at time tᵢ
    Edges: K-nearest neighbors in temporal space
    Edge Weights: Proximity-based weights
```

**K-Nearest Neighbors Approach:**
```python
num_neighbours = 70  # Each node connects to 70 nearest neighbors

# For time step i:
# Connect to: i-35, i-34, ..., i-1, i+1, ..., i+34, i+35
# Creates a graph capturing local temporal context
```

**Why Graphs for Time Series?**
- **Capture long-range dependencies**: Not just sequential, but proximity-based
- **Learn structural patterns**: Graph topology reveals wave dynamics
- **Enable message passing**: Information flows beyond immediate neighbors
- **Flexible receptive field**: K can be tuned for different scales

**GNN Architecture Components:**

**1. Graph Isomorphism Network (GIN) Layers:**
```python
class WeightedGINConv(MessagePassing):
    """
    Custom GIN layer with edge weights
    - More expressive than standard GCN
    - Theoretically as powerful as WL test
    - Learns node embeddings via message passing
    """
```

**2. Multi-Layer Perceptron (MLP) Processing:**
```
Pre-GIN MLP:
    Linear(1 → dim_pre_MLP) → ReLU → BatchNorm → Dropout
    
GIN Layers (repeated num_graph_layers times):
    MLP: Linear → ReLU → BatchNorm → Dropout
    GINConv(MLP, aggr='add')
    
Post-GIN MLP:
    Linear(dim_pre_MLP → dim_post_MLP) → ReLU → BatchNorm → Dropout
```

**3. Skip Connections:**
```python
class GraphIsomorphismNetworkWithSkipConnections:
    """
    Enhanced GIN with residual connections
    - Improves gradient flow
    - Enables deeper networks
    - Better feature reuse
    """
    
# Skip connection implementation:
x_residual = x  # Store input
x = GIN_layer(x)  # Apply GIN
x = x + x_residual  # Add residual
```

**4. Global Pooling:**
```python
# Aggregate node embeddings to graph-level representation
graph_embedding = global_mean_pool(node_embeddings, batch)
# or
graph_embedding = global_add_pool(node_embeddings, batch)
```

**Complete Architecture:**
```
Input: Time Series (batch, time_steps, 1)
    ↓
Convert to Graph (nodes = time_steps)
    ↓
Pre-processing MLP
    ├─> Node embeddings: (batch × time_steps, dim_pre_MLP)
    ↓
GIN Layer 1 (with skip connections)
    ├─> Message passing over k-NN graph
    ├─> Update node features
    ↓
GIN Layer 2 (with skip connections)
    ├─> Deeper representations
    ↓
... (num_graph_layers)
    ↓
Global Pooling
    ├─> Graph-level embedding: (batch, dim_pre_MLP)
    ↓
Post-processing MLP
    ├─> Final features: (batch, dim_post_MLP)
    ↓
Graph-level Classifier
    ├─> Linear(dim_post_MLP → 2)
    ├─> Sigmoid activation
    ↓
Output: Binary prediction [No Rogue Wave, Rogue Wave]
```

**Training Configuration:**
- **Optimizer**: AdamW with weight decay (0.01)
- **Learning Rate**: 5e-4 with ReduceLROnPlateau scheduler
- **Loss Function**: Binary Cross-Entropy (BCE)
- **Batch Size**: 32 graphs per batch
- **Early Stopping**: Patience of 10-20 epochs
- **Regularization**: Dropout (0.2-0.5) + Weight decay

### 3. 📉 GNN with PCA Dimensionality Reduction

#### `GraphLevelClassification_PCA_GIN.ipynb`

Enhanced efficiency through Principal Component Analysis:

**PCA Integration:**

**Motivation:**
- Reduce computational complexity
- Remove redundant temporal information
- Focus on most important variations
- Maintain accuracy while improving speed

**PCA Pipeline:**
```python
from sklearn.decomposition import PCA

# Original data shape: (samples, time_steps, 1)
# Flatten for PCA: (samples, time_steps)
X_train_flat = wave_data_train.reshape(n_samples, -1)

# Apply PCA
pca = PCA(n_components=100)  # Reduce from ~900 to 100 dimensions
X_train_pca = pca.fit_transform(X_train_flat)

# Explained variance
print(f"Variance explained: {sum(pca.explained_variance_ratio_):.2%}")
# Typical result: >95% variance with 100 components

# Reshape for graph construction
X_train_graph = X_train_pca.reshape(n_samples, n_components, 1)
```

**Benefits of PCA Approach:**
- **Faster Training**: Smaller graphs (100 nodes vs 900 nodes)
- **Reduced Memory**: Lower GPU/CPU memory requirements
- **Maintained Accuracy**: Minimal performance drop (<2%)
- **Better Generalization**: Noise reduction through dimensionality reduction

**Graph Construction with PCA:**
```python
# After PCA reduction
num_nodes = 100  # Instead of 900
num_neighbours = 20  # Fewer neighbors needed (instead of 70)

# Smaller, more efficient graphs
# Faster message passing
# Quicker training iterations
```

**Performance Comparison:**
| Metric | Original Data | PCA-Reduced | Difference |
|--------|--------------|-------------|------------|
| Nodes per graph | ~900 | 100 | 89% reduction |
| Training time/epoch | ~5 min | ~1 min | 80% faster |
| GPU memory | ~6 GB | ~2 GB | 67% reduction |
| Accuracy | 72-74% | 70-73% | <2% drop |

## 📊 Model Performance

### Wave Group-Specific Results

**Group 4 (Largest Waves: H/Hs ≥ 3.5):**
| Model | Accuracy | Training Time | Nodes/Graph |
|-------|----------|---------------|-------------|
| **GIN (Original)** | **>72%** | ~5 min/epoch | 900 |
| **GIN (PCA)** | **70-72%** | ~1 min/epoch | 100 |
| LSTM (Baseline) | 65-67% | ~3 min/epoch | N/A |

**Performance Across All Wave Groups:**
| Wave Group | H/Hs Range | GIN Accuracy | LSTM Accuracy | Improvement |
|------------|------------|--------------|---------------|-------------|
| Group 1 | 2.0-2.5 | 68-70% | 65-67% | +3-5% |
| Group 2 | 2.5-3.0 | 70-71% | 66-68% | +3-4% |
| Group 3 | 3.0-3.5 | 71-73% | 67-69% | +3-4% |
| **Group 4** | **≥3.5** | **>72%** | **65-67%** | **+5-7%** |

**Key Observations:**
- 🎯 **GNN outperforms LSTM**: Especially for largest waves (+5-7%)
- 📊 **Group 4 shows best improvement**: Most extreme waves benefit most from graph structure
- ⚡ **PCA maintains performance**: <2% accuracy drop with 80% faster training
- 🔄 **Consistent gains**: GNN superior across all wave magnitude groups

### Architecture Comparison

**Skip Connections Impact:**
| Architecture | Accuracy | Convergence Speed | Stability |
|--------------|----------|------------------|-----------|
| GIN (No Skip) | 68-70% | Slower | Less stable |
| **GIN (With Skip)** | **72-74%** | **Faster** | **More stable** |

**Graph Pooling Methods:**
| Pooling Type | Accuracy | Best For |
|--------------|----------|----------|
| Global Mean Pool | 71-72% | Balanced representation |
| **Global Add Pool** | **72-74%** | **Stronger signals** |

### Training Dynamics

**Typical Training Curves:**
- **Convergence**: 50-100 epochs with early stopping
- **Best epoch**: Usually 40-70 epochs
- **Overfitting**: Minimal with proper dropout (0.2-0.5)
- **Learning rate**: Starts at 5e-4, reduced by factor of 0.7 on plateau

**Computational Requirements:**
- **GPU**: NVIDIA GPU with 6-8 GB VRAM (recommended)
- **Training time**: 2-4 hours for full training (original data)
- **Training time**: 30-60 minutes (PCA-reduced data)
- **Inference**: Real-time capable (<100ms per graph)

## 🚀 Installation and Setup

### Prerequisites
- Python 3.7 or higher
- PyTorch 1.9+ with CUDA support (recommended)
- PyTorch Geometric
- GPU with 6+ GB VRAM (recommended)

### Required Libraries

Install PyTorch and PyTorch Geometric:

```bash
# Install PyTorch (with CUDA 11.3 example)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu113

# Install PyTorch Geometric and dependencies
pip install torch-geometric torch-scatter torch-sparse torch-cluster
```

Install other dependencies:

```bash
pip install numpy matplotlib seaborn scikit-learn networkx scipy tensorflow
```

Or use requirements file:

```bash
pip install -r requirements_gnn_roguewave.txt
```

#### Core Libraries:
- **torch** (>=1.9.0): PyTorch deep learning framework
- **torch-geometric** (>=2.0.0): Graph neural networks library
- **torch-scatter, torch-sparse, torch-cluster**: PyG dependencies
- **numpy** (>=1.20.0): Numerical computing
- **scikit-learn** (>=0.24.0): PCA and preprocessing

#### Visualization:
- **matplotlib** (>=3.4.0): Plotting
- **seaborn** (>=0.11.0): Statistical visualization
- **networkx** (>=2.6.0): Graph visualization

#### Additional:
- **scipy** (>=1.7.0): Scientific computing
- **tensorflow** (>=2.6.0): For data loading compatibility

### Verify Installation

```python
import torch
import torch_geometric

print(f"PyTorch version: {torch.__version__}")
print(f"PyTorch Geometric version: {torch_geometric.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
```

## 💻 Usage

### Recommended Execution Order

#### Step 1: Wave Group Segregation and Data Preparation

```bash
jupyter notebook Data_Preparation_different_wave_groups.ipynb
```

**Configure parameters:**
```python
# Forecast horizon and training window
forecast_horizon = 5  # minutes
training_window = 15  # minutes

# Wave group thresholds (H/Hs ranges)
wave_groups = {
    1: (2.0, 2.5),
    2: (2.5, 3.0),
    3: (3.0, 3.5),
    4: (3.5, float('inf'))  # ≥3.5
}

# Output directory
output_dir = "wave_groups/"
```

**This notebook will:**
- Load raw CDIP buoy data
- Extract rogue wave and non-rogue wave windows
- Classify waves into magnitude groups
- Normalize by significant wave height
- Save group-specific NPZ files

**Output:** 
- `tadv_5min_wave_group_window_15mins_1.npz` (Group 1)
- `tadv_5min_wave_group_window_15mins_2.npz` (Group 2)
- `tadv_5min_wave_group_window_15mins_3.npz` (Group 3)
- `tadv_5min_wave_group_window_15mins_4.npz` (Group 4: Best performance)

#### Step 2: Train GIN Model (Original Data)

```bash
jupyter notebook GraphLevelClassification_OriginalLength_GIN.ipynb
```

**Configure GNN parameters:**
```python
# Graph construction
num_neighbours = 70  # K-nearest neighbors for edges

# GIN architecture
dim_pre_MLP = 64      # Pre-processing MLP output dimension
dim_post_MLP = 32     # Post-processing MLP output dimension
dim_graphLin = 2      # Final classifier dimension
num_pre_layers = 2    # Number of pre-processing layers
num_post_layers = 2   # Number of post-processing layers
num_graph_layers = 3  # Number of GIN layers
dropout_prob = 0.3    # Dropout rate

# Training
batch_size = 32
learning_rate = 5e-4
weight_decay = 0.01

# Select wave group to train
file_str = "tadv_5min_wave_group_window_15mins_4"  # Largest waves
```

**This notebook will:**
- Load wave group data
- Convert time series to graphs with k-NN edges
- Build GIN model with skip connections
- Train with AdamW optimizer and early stopping
- Save best model checkpoint
- Generate training curves and confusion matrix

**Training Time:** ~2-4 hours on GPU for full dataset

#### Step 3: Train GIN Model with PCA (Optional, Faster)

```bash
jupyter notebook GraphLevelClassification_PCA_GIN.ipynb
```

**Configure PCA:**
```python
# PCA dimensionality reduction
n_components = 100  # Reduce from ~900 to 100

# Adjusted graph construction
num_neighbours = 20  # Fewer neighbors for smaller graphs

# Same GIN architecture as before but faster training
```

**This notebook will:**
- Apply PCA to time series data
- Create smaller, more efficient graphs
- Train GIN model (same architecture)
- Achieve similar accuracy with 80% faster training

**Training Time:** ~30-60 minutes on GPU

### 🔄 Complete Workflow Example

```python
# Example: Train GNN for largest rogue waves (Group 4)

# 1. Prepare wave group data
# Run Data_Preparation_different_wave_groups.ipynb
# Generates: tadv_5min_wave_group_window_15mins_4.npz

# 2. Load and convert to graphs
data = np.load('wave_groups/tadv_5min_wave_group_window_15mins_4.npz')
wave_data = data['wave_data_train']

# Create graph dataset
graphs = []
for sample in wave_data:
    # Convert time series to graph
    graph = create_knn_graph(sample, k=70)
    graphs.append(graph)

# 3. Train GIN model
model = GraphIsomorphismNetworkWithSkipConnections(
    dim_pre_MLP=64,
    dim_post_MLP=32,
    num_graph_layers=3,
    dropout_prob=0.3
)

# 4. Train with PyTorch Geometric DataLoader
train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)

# Training loop with early stopping...
# Expected accuracy: >72% for Group 4
```

## 🎯 Technical Deep Dive

### Graph Construction Details

**K-Nearest Neighbors in Time:**
```python
def create_knn_temporal_edges(time_series, k=70):
    """
    Create edges between each node and its k nearest neighbors in time
    
    For node at time step i:
    - Connect to nodes: max(0, i-k//2) to min(n-1, i+k//2)
    - Edge weights based on temporal distance
    """
    n = len(time_series)
    edge_index = []
    edge_weights = []
    
    for i in range(n):
        # Find k//2 neighbors on each side
        start = max(0, i - k//2)
        end = min(n, i + k//2 + 1)
        
        for j in range(start, end):
            if i != j:
                edge_index.append([i, j])
                # Weight decreases with distance
                weight = 1.0 / (1.0 + abs(i - j))
                edge_weights.append(weight)
    
    return torch.tensor(edge_index).t(), torch.tensor(edge_weights)
```

**Node Features:**
```python
# Each node = one time step
# Feature = normalized wave height at that time
node_features = wave_time_series.reshape(-1, 1)  # (num_nodes, 1)
```

### GIN Layer Mathematics

**Message Passing in GIN:**
```
h_i^(k+1) = MLP^(k)((1 + ε^(k)) · h_i^(k) + Σ_{j∈N(i)} w_ij · h_j^(k))

Where:
- h_i^(k): Node i's representation at layer k
- N(i): Neighbors of node i
- w_ij: Edge weight between nodes i and j
- ε^(k): Learnable parameter
- MLP^(k): Multi-layer perceptron at layer k
```

**Why GIN is Powerful:**
- As expressive as Weisfeiler-Lehman (WL) graph isomorphism test
- Can distinguish different graph structures
- MLP learns complex node update functions
- Aggregation preserves multi-set information

### Skip Connections in GNNs

**Residual Connection:**
```python
# Standard GNN layer
x_out = GIN_layer(x_in)

# With skip connection
x_out = GIN_layer(x_in) + x_in  # Element-wise addition

# Benefits:
# 1. Gradient flow: Direct path for backpropagation
# 2. Feature reuse: Lower-level features accessible to higher layers
# 3. Easier training: Mitigates vanishing gradient
```

## 🛠️ Techniques Used

### Graph Neural Networks
- **Graph Isomorphism Networks (GIN)**: Expressive graph learning
- **Message Passing**: Neighbor information aggregation
- **Global Pooling**: Graph-level representations
- **Skip Connections**: Residual learning for deep networks

### Deep Learning
- **PyTorch Geometric**: State-of-the-art GNN library
- **AdamW Optimizer**: Weight decay for regularization
- **Learning Rate Scheduling**: ReduceLROnPlateau for adaptive learning
- **Early Stopping**: Prevent overfitting
- **Batch Normalization**: Training stability

### Dimensionality Reduction
- **PCA**: Principal Component Analysis for efficiency
- **Variance Preservation**: >95% explained variance
- **Computational Efficiency**: 80% faster training

### Time Series Analysis
- **Zero-Crossing Method**: Wave extraction
- **Significant Wave Height**: Normalization baseline
- **Wave Group Classification**: Magnitude-based segregation
- **Balanced Sampling**: Equal class distribution

## 💾 Output Files

The project generates:
- **NPZ Data Files**: Wave group-specific datasets
- **Graph Objects**: PyTorch Geometric Data objects
- **Model Checkpoints**: Trained GIN models (`.pt` files)
- **Training Curves**: Loss and accuracy plots
- **Confusion Matrices**: Performance visualization
- **PCA Models**: Fitted PCA transformers for dimensionality reduction

## 🔧 Customization

### Adjusting Graph Structure

```python
# Change number of neighbors
num_neighbours = 50  # Smaller receptive field (from 70)
num_neighbours = 100  # Larger receptive field

# Different edge weighting schemes
def custom_edge_weights(distance):
    return np.exp(-distance)  # Exponential decay
    # or
    return 1.0 if distance <= 5 else 0.5  # Step function
```

### Modifying GIN Architecture

```python
# Deeper network
num_graph_layers = 5  # From 3 to 5 GIN layers

# Wider layers
dim_pre_MLP = 128  # From 64 to 128
dim_post_MLP = 64  # From 32 to 64

# More aggressive dropout
dropout_prob = 0.5  # From 0.3 to 0.5
```

### PCA Configuration

```python
# More/fewer components
n_components = 150  # More detailed (from 100)
n_components = 50   # More aggressive reduction

# Kernel PCA for non-linear reduction
from sklearn.decomposition import KernelPCA
pca = KernelPCA(n_components=100, kernel='rbf')
```

### Training Hyperparameters

```python
# Learning rate
learning_rate = 1e-3  # Higher for faster convergence
learning_rate = 1e-4  # Lower for fine-tuning

# Weight decay
weight_decay = 0.001  # Less regularization (from 0.01)
weight_decay = 0.1    # More regularization

# Batch size
batch_size = 64  # Larger batches (from 32)
batch_size = 16  # Smaller batches for large graphs
```

## 🌊 Applications

### Primary Use Case: Enhanced Maritime Safety
- **Magnitude-specific warnings**: Targeted alerts for different wave sizes
- **Improved accuracy**: Especially for largest, most dangerous waves
- **Graph-based insights**: Understanding wave pattern evolution
- **Real-time forecasting**: Fast inference for operational deployment

### Extended Applications
- 🛳️ **Ship routing optimization**: Avoid regions with high rogue wave probability
- 🏗️ **Offshore structure design**: Inform engineering specifications
- 🔬 **Wave physics research**: Discover new formation mechanisms
- 📊 **Climate modeling**: Understand extreme event trends
- 🎓 **GNN methodology**: Template for other time series problems

## 🔍 Troubleshooting

### Common Issues

**Issue**: CUDA out of memory during training
```python
# Solution 1: Reduce batch size
batch_size = 16  # From 32

# Solution 2: Use PCA version
# Switch to GraphLevelClassification_PCA_GIN.ipynb

# Solution 3: Reduce number of neighbors
num_neighbours = 50  # From 70

# Solution 4: Use CPU (slower but works)
device = torch.device('cpu')
```

**Issue**: Poor accuracy (<60%)
- **Check wave group balance**: Ensure 50-50 rogue/non-rogue split
- **Verify normalization**: Significant wave height calculation correct
- **Increase training epochs**: May need >100 epochs for some groups
- **Try different wave group**: Group 4 (largest) typically performs best

**Issue**: PyTorch Geometric installation errors
```bash
# Install specific versions matching your PyTorch and CUDA
pip install torch-geometric==2.0.4
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
```

**Issue**: Graph construction too slow
```python
# Use vectorized operations
import torch_cluster
edge_index = torch_cluster.knn_graph(x, k=num_neighbours)

# Or switch to PCA version for smaller graphs
```

**Issue**: Model not converging
```python
# Check learning rate
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)  # Lower LR

# Add gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Verify data preprocessing
print(f"Data range: [{data.min()}, {data.max()}]")
# Should be normalized (roughly [-3, 3])
```

## 🤝 Contributing

When extending this GNN framework:
1. Document new graph construction methods
2. Compare with baseline GIN architecture
3. Test across all wave groups (1-4)
4. Report training time and memory usage
5. Visualize learned graph representations

## 📝 Future Improvements

- 🔄 **Graph Attention Networks (GAT)**: Learn edge importance automatically
- 🌐 **Spatial-Temporal GNNs**: Combine multiple buoy locations as graph
- 🎯 **Multi-task Learning**: Predict both occurrence and magnitude
- 📊 **Explainability**: GNNExplainer for understanding predictions
- 🧠 **Transformer-based GNNs**: Self-attention mechanisms
- 🔮 **Longer horizons**: Extend to 10-15 minute forecasts
- 📱 **Edge deployment**: Optimize for mobile/embedded systems
- 🎨 **Dynamic graphs**: Adapt graph structure during inference

## 📄 License

This project is part of Ph.D. research at University of Maryland. Data from CDIP is publicly available.

## 📧 Contact

For questions or collaboration:
- **Email**: schakr18@umd.edu
- **LinkedIn**: [linkedin.com/in/samarpan-chakraborty](https://linkedin.com/in/samarpan-chakraborty)
- **GitHub**: [github.com/SamarpanChakraborty97](https://github.com/SamarpanChakraborty97)

## 🙏 Acknowledgments

This research extends previous work on rogue wave forecasting at the University of Maryland (2019-2025), introducing Graph Neural Networks as a novel approach to oceanographic time series analysis. The project demonstrates that graph-based representations can capture complex wave dynamics more effectively than traditional sequential models, achieving >72% accuracy for the most extreme wave events.

**Key Contributions**:
- Novel application of GNNs to rogue wave forecasting
- Wave magnitude-based model specialization
- Graph construction methodology for time series
- Integration of PCA for computational efficiency
- >72% accuracy for largest wave groups (significant improvement over LSTM baseline)

---

**Note**: This project demonstrates cutting-edge application of Graph Neural Networks to oceanographic time series forecasting. The graph-based approach captures spatial-temporal relationships that traditional sequential models miss, making it particularly effective for extreme event prediction. The wave group segregation strategy enables magnitude-specific modeling, improving accuracy where it matters most—for the largest, most dangerous rogue waves.

**Version**: 1.0  
**Last Updated**: November 2025  
**Research Period**: 2023-2025 (Building on 2019-2025 rogue wave research)
