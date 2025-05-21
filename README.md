# Simav Geothermal & Seismic AI Clustering

This project applies **AI-based unsupervised clustering** to geochemical and seismic data from the **Simav geothermal region** using a **Deep Embedded Clustering (DEC)** model. It produces spatial cluster maps overlaid with **borehole locations** and **ML ≥ 4 earthquake events**.

##  Overview

- **Model**: Deep Embedded Clustering (DEC)
- **Features**: Geochemical and geophysical measurements
- **Spatial Context**: Boreholes and earthquakes plotted with clusters
- **Data Sources**:
  - `simav_geothermal.xlsx`
  - `kuyular simav son.xlsx` (boreholes)
  - `Simav_Region_Earthquake_Datas_35S_UTM.xlsx` (earthquakes)

## 🤖 What is Deep Embedded Clustering (DEC)?

**Deep Embedded Clustering (DEC)** is an unsupervised learning algorithm that combines deep feature learning with clustering. It works in two main phases:

1. **Autoencoder Pretraining**:
   - A deep autoencoder is trained to compress high-dimensional input data (e.g., geochemical features) into a low-dimensional latent space.
   - The encoder learns representations that preserve important structure while filtering noise.

2. **Cluster Refinement via KL Divergence**:
   - The latent features are passed into a learnable clustering layer with trainable cluster centroids.
   - The model minimizes a **KL divergence loss** between soft assignments (based on a Student's t-distribution) and a sharpened "target distribution" to refine cluster boundaries.
   - This encourages each point to move closer to its most probable cluster center while separating distinct clusters.

Unlike standard clustering (e.g. KMeans), **DEC jointly learns both the feature representation and the clusters**, leading to more meaningful groups in complex, high-dimensional data like geochemistry.

📘 *Reference: Xie et al., "Unsupervised Deep Embedding for Clustering Analysis", ICML 2016*


##  Files

- `AI-based-cluster.py`: Main DEC training and visualization script
- `DEC_cluster_results.csv`: Clustered data with coordinates
- `DEC_Clusters_Map.jpeg`: Map with DEC clusters, boreholes, and earthquakes
- `DEC_performance_log.txt`: Silhouette and Davies-Bouldin scores
- `DEC_latent_space.csv`: Latent representations (DEC encoder output)

##  Output Preview

- Convex hull visualizations of clusters
- Overlay of:
  - Boreholes (black `+`)
  - Earthquakes (red unfilled circles)

## ⚙️ Usage

```bash
python AI-based-cluster.py
