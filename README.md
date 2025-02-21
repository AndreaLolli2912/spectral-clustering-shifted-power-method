# Spectral Clustering Homework (HW SC)

This repository, **spectral-clustering-shifted-power-sparse**, hosts all the materials and code for the mandatory spectral clustering homework in the course *Computational Linear Algebra for Large Scale Problems* (A.A. 2024–2025). It features a from-scratch Shifted Inverse Power Method with Deflation for eigenvalue computations using SciPy sparse matrices.

---

## Contents

- **Technical Report (`HW_SpectralClustering.pdf`)**: Explains the methods, results, and comparisons with other clustering approaches.
- **Instructions (`instruction.txt`)**: Shows how to run the code and reproduce the results.
- **Source Code**: Contains Python scripts for each step of the homework.

---

## Task Overview

1. **k-NN Similarity Graph & Adjacency Matrix**  
   [`knn_similarity_graph.py`](knn_similarity_graph.py) builds a k-nearest-neighbor graph from the data and forms the weighted adjacency matrix.

2. **Degree Matrix & Laplacian**  
   [`laplacian_matrix.py`](laplacian_matrix.py) constructs the degree matrix and Laplacian for spectral clustering.

3. **Connectivity**  
   [`connected_components.py`](connected_components.py) checks how many connected components exist.

4. **Finding Clusters**  
   [`iterative_power_methods.py`](iterative_power_methods.py) computes small eigenvalues of the Laplacian using a shifted inverse power method plus deflation.

5. **Eigenvectors & Embedding**  
   [`eigen_pairs.py`](eigen_pairs.py) extracts the corresponding eigenvectors to embed the data.

6. **k-Means on Embedding**  
   Groups the embedded points into clusters.

7. **Visualization**  
   Plots clusters in distinct colors.

8. **Compare Methods**  
   Evaluates spectral clustering against other clustering techniques.

---

## Libraries & Tools

- **NumPy / SciPy** for numerical operations and sparse matrix support.
- **Custom Iterative Power Methods** for eigenvalue approximation.

---

## Getting Started

1. **Install Requirements**: Check [`requirements.txt`](requirements.txt).
2. **Run Scripts**: See [`instruction.txt`](instruction.txt) for steps. But just install the requirements and run the main notebook and that's it.
3. **Experiment**: Adjust parameters like `k` in k-NN or the number of eigenvalues or use different datasets.

---

## Contributions & Contact

Use or adapt this code for academic or personal research. For questions or suggestions, open an issue or reach out directly.

