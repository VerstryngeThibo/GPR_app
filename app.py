import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel

# Sample data
X = np.array([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
y = np.array([1.0, 1.2, 0.8, 0.9, 5.0, 8.0, 5.0, 8.0, 9.0, 0.8, 1.0])
X_test = np.linspace(0, 10, 500).reshape(-1, 1)

# Sidebar sliders
st.sidebar.title("GPR Parameters")
lengthscale = st.sidebar.slider("Length Scale ℓ", 0.1, 10.0, 1.5, 0.1)
variance = st.sidebar.slider("Signal Variance σ²_y", 0.1, 10.0, 1.0, 0.1)
nu = st.sidebar.selectbox("Smoothness ν", [0.5, 1.5, 2.5, "∞ (RBF)"])
noise_level = st.sidebar.slider("Noise Level", 1e-5, 1.0, 1e-2, 0.001, format="%.4f")

# Kernel selection
if nu == "∞ (RBF)":
    kernel = variance * RBF(length_scale=lengthscale) + WhiteKernel(noise_level=noise_level)
else:
    kernel = variance * Matern(length_scale=lengthscale, nu=float(nu)) + WhiteKernel(noise_level=noise_level)

# GP model
gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
gp.fit(X, y)
y_pred, sigma = gp.predict(X_test, return_std=True)

# Plot
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(X_test, y_pred, 'b-', label="GPR mean")
ax.fill_between(X_test.ravel(), y_pred - 2*sigma, y_pred + 2*sigma, color='blue', alpha=0.2, label='95% CI')
ax.scatter(X, y, c='red', label='Observed data')
ax.set_xlabel("Distance along profile line [m]")
ax.set_ylabel("$q_{c,avg,5m}$ [MPa]")
ax.set_title("GPR with Matérn or RBF Kernel")
ax.grid(True)
ax.legend()
st.pyplot(fig)
