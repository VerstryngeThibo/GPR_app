{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "1a2f338f-e931-4f76-9e81-26f78f539761",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn.gaussian_process import GaussianProcessRegressor\n",
    "from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C\n",
    "from ipywidgets import interact, FloatSlider, Output\n",
    "from IPython.display import display, clear_output"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "f6ca97e0-f9e5-4a49-85a0-8ceaaeae158b",
   "metadata": {},
   "outputs": [],
   "source": [
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "630fdfac-3601-4796-a8e9-4d429ad0d047",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "b7184f51e85946e4bf9cc349342eed68",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "interactive(children=(FloatSlider(value=1.5, description='Lengthscale', max=10.0, min=0.1), FloatSlider(value=…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "0598e83a397b4efaab566d9049c1c700",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Output()"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Sample data\n",
    "X = np.array([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])\n",
    "y = np.array([1.0, 1.2, 0.8, 0.9, 5.0, 8.0, 5.0, 8.0, 9.0, 0.8, 1.0])\n",
    "X_test = np.linspace(0, 10, 500).reshape(-1, 1)\n",
    "\n",
    "out = Output()\n",
    "\n",
    "def plot_gpr(lengthscale, variance, nu, noise_level):\n",
    "    with out:\n",
    "        clear_output(wait=True)\n",
    "        \n",
    "        # Kernel setup:\n",
    "        # ConstantKernel for signal variance\n",
    "        kernel = (\n",
    "            C(variance, constant_value_bounds=\"fixed\") * \n",
    "            Matern(length_scale=lengthscale, length_scale_bounds=\"fixed\", nu=nu) + \n",
    "            WhiteKernel(noise_level=noise_level, noise_level_bounds=\"fixed\")\n",
    "        )\n",
    "        \n",
    "        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True)\n",
    "        gp.fit(X, y)\n",
    "        y_pred, sigma = gp.predict(X_test, return_std=True)\n",
    "        \n",
    "        plt.figure(figsize=(10, 5))\n",
    "        plt.plot(X_test, y_pred, 'b-', label=f\"GPR mean (l={lengthscale:.2f}, σ²_y={variance:.2f}, ν={nu:.2f}, noise={noise_level:.4f})\")\n",
    "        plt.fill_between(X_test.ravel(), y_pred - 2*sigma, y_pred + 2*sigma,\n",
    "                         alpha=0.2, color='blue', label=\"95% confidence interval\")\n",
    "        plt.scatter(X, y, c='red', label='Observed data')\n",
    "        \n",
    "        plt.title(\"GPR with Matern + White Noise Kernel\")\n",
    "        plt.xlabel(\"Distance along profile line [m]\")\n",
    "        plt.ylabel(\"$q_{c,avg,5m}$ [MPa]\")\n",
    "        plt.legend()\n",
    "        plt.grid(True)\n",
    "        plt.tight_layout()\n",
    "        plt.show()\n",
    "\n",
    "interact(\n",
    "    plot_gpr,\n",
    "    lengthscale=FloatSlider(value=1.5, min=0.1, max=10.0, step=0.1, description='Lengthscale'),\n",
    "    variance=FloatSlider(value=1.0, min=0.1, max=10.0, step=0.1, description='Signal Variance σ²_y'),\n",
    "    nu=FloatSlider(value=1.5, min=0.5, max=2.5, step=0.1, description='Smoothness ν'),\n",
    "    noise_level=FloatSlider(value=1e-2, min=1e-5, max=1.0, step=1e-3, readout_format='.4f', description='Noise Level')\n",
    ")\n",
    "display(out)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python (THESIS)",
   "language": "python",
   "name": "thesis"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
