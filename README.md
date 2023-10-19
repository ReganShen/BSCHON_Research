# Training Graph Neural Networks using Metaheuristics

Welcome to the repository! Here, you'll find the code and experiments supporting my Computer Science honours research. I've delved into the realm of training Graph Neural Networks (GNNs) using metaheuristics, and the results are promising.

During my investigations, I discovered that particular PSO algorithms which leveraged gradient information were adept at accurately training GNN models.

## Repository Structure

- **GAT**: This folder contains the Graph Attention Network model. The code has been sourced from the original GAT paper and adopted from this [GitHub repository](https://github.com/Diego999/pyGAT).
  
- **GCN1**: Here, you'll find the code for the Graph Convolutional Network model used in the "GCN for text classification" paper. It's sourced from this [github repo](https://github.com/yao8839836/text_gcn).

- **GCN2**: A custom Graph Convolutional model that I personally implemented using PyTorch to incorporate custom convolutional layers.

## Setup and Usage

1. Install the required dependencies listed in `requirements.txt`.
2. Configure parameters using the `parameters.json` file for optimal hyperparameter tuning.
3. Simply run `main.py` to get started.

### Changing Datasets

To alter the datasets for each model:

- **GAT**: Navigate to `train.py` and locate line 74. You can choose from Citeseer, Pubmed, and Cora datasets. Uncomment your desired dataset and comment out the others.

- **GCN1**: In `main.py`, go to line 20. The available datasets are MR and 20NG. Toggle between them by commenting out the unwanted one.

- **GCN2**: Refer to `train.py` line 20. Similar to GAT, choose between Citeseer, Pubmed, and Cora.

### Optimization Algorithms

Each model is equipped to deploy five different optimization algorithms on a specified dataset:

- **Adam Optimizer**: (Short description about Adam Optimizer)
  
- **SGD**: (Short description about Stochastic Gradient Descent)
  
- **PSO**: (Short description about Particle Swarm Optimizer)
  
- **GDPSO**: A hybrid algorithm utilizing both Gradient Descent and Particle Swarm Optimization. The algorithm maintains the conventional velocity and position update equations. Yet, the swarm's best particle employs the Gradient Descent method for position updates. For reference, check out this [thesis](https://repository.up.ac.za/bitstream/handle/2263/24965/Complete.pdf?sequence=8&isAllowed=y) by Barla-Szabó, Daniel (2010).
  
- **AdamPSO**: A novel variation of GDPSO, introduced in our research. It aligns with GDPSO's principles but uses Adam Optimizer instead of Gradient Descent.

### Output

The application generates an output for each optimization algorithm in a text file, detailing its training journey and final testing results.
