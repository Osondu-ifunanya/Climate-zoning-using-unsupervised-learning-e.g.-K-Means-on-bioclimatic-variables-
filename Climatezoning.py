Python 2.7.16 (v2.7.16:413a49145e, Mar  4 2019, 01:30:55) [MSC v.1500 32 bit (Intel)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ----------------------------
# 1. Generate Synthetic Bioclimatic Data
# ----------------------------
np.random.seed(42)
n_samples = 1000

# Simulated bioclimatic variables
temperature = np.random.normal(15, 10, n_samples)         # °C
precipitation = np.random.gamma(2, 50, n_samples)         # mm
humidity = np.random.uniform(30, 90, n_samples)           # %
solar_radiation = np.random.normal(200, 50, n_samples)    # W/m²
wind_speed = np.random.normal(3, 1, n_samples)            # m/s

# Combine into DataFrame
df = pd.DataFrame({
    'Temperature': temperature,
    'Precipitation': precipitation,
    'Humidity': humidity,
    'Solar_Radiation': solar_radiation,
    'Wind_Speed': wind_speed
})

# ----------------------------
# 2. Preprocess and Scale
# ----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# ----------------------------
# 3. Apply K-Means Clustering
# ----------------------------
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
df['Climate_Zone'] = kmeans.fit_predict(X_scaled)

# ----------------------------
# 4. Visualize using PCA
# ----------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8,6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['Climate_Zone'], cmap='Set2', s=20)
plt.title("Climate Zones (K-Means Clustering)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(scatter, label='Climate Zone')
plt.grid(True)
plt.tight_layout()
plt.show()

# ----------------------------
# 5. Export to Excel
# ----------------------------
output_path = "synthetic_climate_zoning.xlsx"
df.to_excel(output_path, index=False)
print(f"Excel file saved: {output_path}")
