import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C

# Sample data
X = np.array([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
y = np.array([1.0, 1.2, 0.8, 0.9, 5.0, 8.0, 5.0, 8.0, 9.0, 0.8, 1.0])
X_test = np.linspace(0, 10, 500).reshape(-1, 1)

# Sidebar sliders
st.sidebar.title("GPR Parameters")
lengthscale = st.sidebar.slider("Lengthscale", 0.1, 10.0, 1.5, 0.1)
variance = st.sidebar.slider("Signal Variance σ²_y", 0.1, 10.0, 1.0, 0.1)
nu = st.sidebar.slider("Smoothness ν", 0.5, 2.5, 1.5, 0.1)
noise_level = st.sidebar.slider("Noise Level", 0.00001, 1.0, 0.01, 0.001, format="%.4f")

# Kernel definition
kernel = (
    C(variance, constant_value_bounds="fixed") *
    Matern(length_scale=lengthscale, length_scale_bounds="fixed", nu=nu) +
    WhiteKernel(noise_level=noise_level, noise_level_bounds="fixed")
)

# GPR fitting and prediction
gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
gp.fit(X, y)
y_pred, sigma = gp.predict(X_test, return_std=True)

# Plotting
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(X_test, y_pred, 'b-', label=f"GPR mean (l={lengthscale:.2f}, σ²_y={variance:.2f}, ν={nu:.2f}, noise={noise_level:.4f})")
ax.fill_between(X_test.ravel(), y_pred - 2 * sigma, y_pred + 2 * sigma,
                alpha=0.2, color='blue', label="95% confidence interval")
ax.scatter(X, y, c='red', label='Observed data')
ax.set_title("GPR with Matern + White Noise Kernel")
ax.set_xlabel("Distance along profile line [m]")
ax.set_ylabel("$q_{c,avg,5m}$ [MPa]")
ax.legend()
ax.grid(True)
st.pyplot(fig)
