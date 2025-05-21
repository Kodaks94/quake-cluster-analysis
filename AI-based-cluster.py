
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Layer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import KLDivergence
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import os

# ==== Parameters ====
target_var_row_no = 5
use_only_chems_with_ppm = True
encoding_dim = 10
n_clusters = 4
maxiter = 5000
update_interval = 100
path_to_save = os.getcwd() + '/AI_Model_Results/'
os.makedirs(path_to_save, exist_ok=True)
# ==== Load Geothermal Data ====
features_raw = pd.read_excel("data/simav_geothermal.xlsx", sheet_name="Sayfa1")
features_raw = features_raw.set_index("NUMUNE").T.reset_index(drop=True)

# Load borehole data (Sayfa2)
df_Kuyular_Simav = pd.read_excel("data/kuyular simav son.xlsx", sheet_name="Sayfa2")
df_Kuyular_Simav.columns = ['Index', 'ID', 'latitude', 'longitude']
df_Kuyular_Simav = df_Kuyular_Simav[['longitude', 'latitude']].dropna().reset_index(drop=True)

# Load earthquake data
df_quakes = pd.read_excel("data/Simav_Region_Earthquake_Datas_35S_UTM.xlsx", sheet_name="Sayfa1")

# Filter earthquakes: ML ≥ 4
df_high_magnitude_earthquakes = df_quakes[df_quakes["ML"] >= 4][["Longitude (X)", "Latitude (Y)"]].dropna().reset_index(drop=True)
df_high_magnitude_earthquakes.columns = ['longitude', 'latitude']



# Coordinates
coords = features_raw.iloc[:2].astype(float)

# Target row
target = features_raw.iloc[target_var_row_no].astype(float).reset_index(drop=True)

# Feature matrix
features = features_raw.iloc[2:].T.reset_index(drop=True)

# Optionally filter columns for ppm-only
if use_only_chems_with_ppm:
    ppm_cols = [col for col in features.columns if 'ppm' in str(col) or '(ppm)' in str(col)]
    if not ppm_cols:
        print("Warning: No ppm columns found. Using all features.")
    else:
        features = features[ppm_cols]

# Coordinates DataFrame
coordinates = pd.DataFrame({
    "Longitude": coords.iloc[0].values,
    "Latitude": coords.iloc[1].values
}).reset_index(drop=True)

# ==== Preprocess ====
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)
X_pca = PCA(n_components=min(30, X_scaled.shape[1])).fit_transform(X_scaled)

# ==== Clustering Layer ====
class ClusteringLayer(Layer):
    def __init__(self, n_clusters, **kwargs):
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters

    def build(self, input_shape):
        self.cluster_centers = self.add_weight(
            shape=(self.n_clusters, input_shape[1]),
            initializer='glorot_uniform',
            trainable=True,
            name='cluster_centers'
        )

    def call(self, inputs):
        q = 1.0 / (1.0 + tf.reduce_sum(tf.square(tf.expand_dims(inputs, axis=1) - self.cluster_centers), axis=2))
        q = q ** ((1 + 1.0) / 2.0)
        q = tf.transpose(tf.transpose(q) / tf.reduce_sum(q, axis=1))
        return q

def target_distribution(q):
    weight = q ** 2 / tf.reduce_sum(q, axis=0)
    return tf.transpose(tf.transpose(weight) / tf.reduce_sum(weight, axis=1))

def build_autoencoder(input_dim, encoding_dim=10):
    input_layer = Input(shape=(input_dim,))
    x = Dense(64, activation='relu')(input_layer)
    x = Dense(32, activation='relu')(x)
    latent = Dense(encoding_dim, name='latent')(x)
    x = Dense(32, activation='relu')(latent)
    x = Dense(64, activation='relu')(x)
    output_layer = Dense(input_dim, activation='linear')(x)
    return Model(input_layer, output_layer), Model(input_layer, latent)

def train_DEC(X, n_clusters=4, encoding_dim=10, maxiter=5000, update_interval=100):
    input_dim = X.shape[1]
    autoencoder, encoder = build_autoencoder(input_dim, encoding_dim)
    autoencoder.compile(optimizer=Adam(1e-3), loss='mse')
    autoencoder.fit(X, X, epochs=50, batch_size=16, shuffle=True, verbose=0)

    latent = encoder.predict(X)
    kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=42).fit(latent)
    clustering_input = Input(shape=(encoding_dim,))
    clustering_output = ClusteringLayer(n_clusters)(clustering_input)
    clustering_model = Model(clustering_input, clustering_output)
    clustering_model.get_layer(index=1).set_weights([kmeans.cluster_centers_])

    q_output = clustering_model(encoder(autoencoder.input))
    dec_model = Model(autoencoder.input, [q_output, autoencoder.output])

    optimizer = Adam(1e-3)
    loss_fn = KLDivergence()

    for i in range(maxiter):
        latent = encoder.predict(X)
        q = clustering_model.predict(latent)
        p = target_distribution(q)
        with tf.GradientTape() as tape:
            q_pred, x_recon = dec_model(X)
            loss = loss_fn(p, q_pred) + tf.reduce_mean(tf.square(X - x_recon))
        grads = tape.gradient(loss, dec_model.trainable_weights)
        optimizer.apply_gradients(zip(grads, dec_model.trainable_weights))
        if i % update_interval == 0:
            print(f"Iter {i}: Loss = {loss.numpy():.4f}")

    final_q = clustering_model.predict(encoder.predict(X))
    return np.argmax(final_q, axis=1), encoder.predict(X)

# ==== Run DEC ====
dec_clusters, latent = train_DEC(X_pca, n_clusters=n_clusters, encoding_dim=encoding_dim)

# ==== Combine & Plot ====
result = coordinates.copy()
result["Target"] = target
result["Cluster"] = dec_clusters

plt.figure(figsize=(10, 8))
colors = plt.cm.Set2.colors
for i in sorted(result["Cluster"].unique()):
    group = result[result["Cluster"] == i]
    plt.scatter(group["Longitude"], group["Latitude"], color=colors[i % len(colors)],
                label=f"Cluster {i}", edgecolor='black', s=80, facecolors='none')
    if len(group) >= 3:
        try:
            hull = ConvexHull(group[["Longitude", "Latitude"]])
            for simplex in hull.simplices:
                plt.plot(group.iloc[simplex, 0], group.iloc[simplex, 1], color=colors[i % len(colors)])
        except:
            pass

# === Add boreholes (sondaj kuyulari) ===
if 'df_Kuyular_Simav' in locals():
    plt.scatter(
        df_Kuyular_Simav['longitude'],
        df_Kuyular_Simav['latitude'],
        marker='+',
        color='black',
        linewidth=2,
        s=100,
        label='Boreholes',
        zorder=5,
        alpha=0.5
    )

# === Add earthquakes (ML ≥ 4) ===
if 'df_high_magnitude_earthquakes' in locals():
    for i, row in df_high_magnitude_earthquakes.iterrows():
        x, y = row[0], row[1]  # Assumes Longitude, Latitude are first two columns
        plt.scatter(
            x, y,
            facecolors='none',
            edgecolors='red',
            s=40,
            linewidth=1.5,
            label='Earthquake (ML ≥ 4)' if i == 0 else "",
            zorder=5
        )

# === Map styling to match theirs ===
fnt_size = 16
plt.xlabel("Longitude", fontsize=fnt_size)
plt.ylabel("Latitude", fontsize=fnt_size)
plt.grid(True)
plt.legend(fontsize=fnt_size - 2)
plt.xticks(fontsize=fnt_size)
plt.yticks(fontsize=fnt_size)
plt.xlim(630000, 690000)
plt.ylim(4.325e6, 4.345e6)
plt.tight_layout()

# Save figure as JPEG (plus other formats if you want)
output = 'DEC_AI_model.jpeg'
plt.savefig(path_to_save + output, format='jpeg', dpi=300, bbox_inches='tight')


plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("DEC Clustering with Convex Hulls")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(path_to_save + "DEC_Clusters_Map.jpeg", dpi=300)
plt.show()


# Save merged result table
result.to_csv(path_to_save + "DEC_cluster_results.csv", index=False)

# Save a version that includes all input features and cluster labels
full_result = features.copy()
full_result["Cluster"] = dec_clusters
full_result["Longitude"] = coordinates["Longitude"]
full_result["Latitude"] = coordinates["Latitude"]
full_result["Target"] = target.values
full_result.to_csv(path_to_save + "DEC_full_data_with_clusters.csv", index=False)

# Save cluster mean profiles (for interpretability)
cluster_means = full_result.groupby("Cluster").mean(numeric_only=True)
cluster_means.to_csv(path_to_save + "DEC_cluster_feature_means.csv")

# Save performance metrics if needed
from sklearn.metrics import silhouette_score, davies_bouldin_score

silhouette = silhouette_score(X_pca, dec_clusters)
db_score = davies_bouldin_score(X_pca, dec_clusters)

with open(path_to_save + "/DEC_performance_log.txt", "w") as f:
    f.write(f"DEC Performance Summary\\n")
    f.write(f"Silhouette Score: {silhouette:.4f}\\n")
    f.write(f"Davies-Bouldin Index: {db_score:.4f}\\n")
    f.write(f"Clusters: {n_clusters}\\n")
    f.write(f"Samples: {X_pca.shape[0]}\\n")

