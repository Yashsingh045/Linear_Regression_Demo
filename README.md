# Linear Regression Deep Dive - Interactive Streamlit App

An interactive, visual, and educational tool designed to teach the fundamentals of Linear Regression, Mean Squared Error (MSE), and Gradient Descent.

## 🚀 Features

- **Interactive Line Fitting**: Manually adjust the slope and intercept to see how they impact the fit.
- **Error Visualization**: See vertical residuals and "Squared Error" blocks to build an intuition for the MSE loss function.
- **Loss Landscape Explorer**: Visualize the 3D bowl-shaped loss surface and 2D contour maps.
- **Gradient Descent Animation**: Watch the model "learn" by descending the loss surface from your manual starting point.
- **Learning Rate Experiments**: Compare small, optimal, and "exploding" learning rates on a convergence plot.
- **Noise & Outliers Analysis**: Understand how Linear Regression is sensitive to extreme data points.

## 🛠️ Installation

1. **Clone the repository** (or copy the files).
2. **Set up a virtual environment** (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🏃 How to Run

Launch the application using Streamlit:

```bash
streamlit run app.py
```

## 📦 Project Structure

- `app.py`: The main Streamlit application containing all logic and visualizations.
- `requirements.txt`: List of Python dependencies.
- `README.md`: This project documentation.

## 🧪 Core Concepts Taught

- **Hypothesis Function**: $y = mx + b$
- **Loss Function (MSE)**: $J(m, b) = \frac{1}{n} \sum (y_i - (mx_i + b))^2$
- **Optimization (Gradient Descent)**: Iteratively updating parameters using gradients of the loss function.
- **Hyperparameters**: The impact of Learning Rate ($\alpha$) on convergence speed and stability.
- **Data Robustness**: The sensitivity of squared error to outliers.

---
