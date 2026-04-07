import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import time

# --- Page Configuration ---
st.set_page_config(page_title="Linear Regression Deep Dive", layout="wide")

# --- Core Logic Functions ---

@st.cache_data
def generate_data(n_points=50, noise=0.1, outliers=False, seed=42):
    """Generates synthetic linear data with optional noise and outliers."""
    np.random.seed(seed)
    X = np.linspace(0, 10, n_points)
    # Underlying relationship: y = 2x + 1
    y = 2 * X + 1 + np.random.normal(0, noise * 5, n_points)
    
    if outliers:
        # Add high-leverage outliers
        X = np.append(X, [1, 2, 8, 9])
        y = np.append(y, [15, 18, 2, 5])
        
    return X, y

def compute_mse(y_true, y_pred):
    """Computes Mean Squared Error."""
    return np.mean((y_true - y_pred)**2)

def compute_grid_loss(X, y, m_range, b_range):
    """Computes MSE over a grid of m and b values."""
    M, B = np.meshgrid(m_range, b_range)
    Z = np.zeros(M.shape)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            y_pred = M[i, j] * X + B[i, j]
            Z[i, j] = compute_mse(y, y_pred)
    return M, B, Z

def run_gradient_descent(X, y, m_init, b_init, lr, n_iters):
    """Performs manual gradient descent and returns history."""
    m, b = m_init, b_init
    history = []
    n = len(X)
    
    for i in range(n_iters):
        y_pred = m * X + b
        loss = compute_mse(y, y_pred)
        history.append((m, b, loss))
        
        # Gradients: 
        # dJ/dm = -2/n * sum(x * (y - (mx + b)))
        # dJ/db = -2/n * sum(y - (mx + b))
        dm = (-2/n) * np.sum(X * (y - y_pred))
        db = (-2/n) * np.sum(y - y_pred)
        
        m -= lr * dm
        b -= lr * db
        
    return history

# --- Sidebar ---
st.sidebar.title("🛠️ Global Controls")
st.sidebar.markdown("Configure the dataset and initial parameters.")

n_points = st.sidebar.slider("Number of Points", 20, 200, 50)
noise_level = st.sidebar.slider("Noise Intensity", 0.0, 2.0, 0.5)
add_outliers = st.sidebar.toggle("Inject Outliers")
random_seed = st.sidebar.number_input("Random Seed", 0, 100, 42)

st.sidebar.divider()
st.sidebar.subheader("Model Hyperparameters")
learning_rate = st.sidebar.slider("Learning Rate (α)", 0.001, 0.1, 0.01, format="%.3f")
iterations = st.sidebar.slider("Iterations", 1, 100, 20)

# Fetch Data
X, y = generate_data(n_points, noise_level, add_outliers, random_seed)

# Calculate Optimal Fit (Analytical Solution) for Global Use
m_opt, b_opt = np.polyfit(X, y, 1)

# Initialize Session State for Manual Controls with Optimal Values
if 'm_manual' not in st.session_state:
    st.session_state.m_manual = float(round(m_opt, 2))
if 'b_manual' not in st.session_state:
    st.session_state.b_manual = float(round(b_opt, 2))

# --- Main Page Header ---
st.title("📈 Linear Regression: Visual Deep Dive")
st.markdown("""
How does Linear Regression actually learn? This interactive tool helps you visualize how a model fits data, 
minimizes Mean Squared Error (MSE), and converges to an optimal solution using **Gradient Descent**.
""")

# --- Main App Tabs ---
tabs = st.tabs([
    "📍 Data & Fit", 
    "📊 Error Viz", 
    "⛰️ Loss Landscape", 
    "⚡ Gradient Descent", 
    "📈 LR Experiments", 
    "⚠️ Noise & Outliers"
])

# --- Tab 1: Data & Line Fit ---
with tabs[0]:
    st.header("1. Data & Line Fitting")
    st.markdown("""
    Linear Regression models the relationship between two variables by fitting a linear equation to observed data.
    The goal is to find the slope (**m**) and intercept (**b**) that describe the best-fitting line.
    """)
    
    col1, col2 = st.columns([1, 3])
    
    # Callback for the button to avoid state modification errors
    def snap_callback():
        st.session_state.m_manual = float(round(m_opt, 2))
        st.session_state.b_manual = float(round(b_opt, 2))

    with col1:
        st.subheader("Manual Layout")
        # Direct slider binding to session_state
        st.slider("Manual Slope (m)", -2.0, 6.0, key="m_manual")
        st.slider("Manual Intercept (b)", -10.0, 10.0, key="b_manual")
        
        # Access values for calculations
        m_manual = st.session_state.m_manual
        b_manual = st.session_state.b_manual

        st.button("🎯 Snap to Best Fit", on_click=snap_callback)

        y_pred_manual = m_manual * X + b_manual
        current_mse = compute_mse(y, y_pred_manual)
        st.metric("Current MSE", f"{current_mse:.4f}")
        
    with col2:
        fig = go.Figure()
        # Data points
        fig.add_trace(go.Scatter(x=X, y=y, mode='markers', name='Data Points', marker=dict(color='#3366CC')))
        
        line_x = np.array([min(X), max(X)])
        
        # Manual Line
        line_y = m_manual * line_x + b_manual
        fig.add_trace(go.Scatter(x=line_x, y=line_y, mode='lines', name='Your Manual Fit', line=dict(color='red', width=4)))
        
        # Reference Optimal Line
        opt_y = m_opt * line_x + b_opt
        fig.add_trace(go.Scatter(x=line_x, y=opt_y, mode='lines', name='Optimal Fit (Target)', line=dict(color='green', dash='dash', width=2)))
        
        fig.update_layout(title="Interactive Line Fitting vs. Target", xaxis_title="X", yaxis_title="y", height=500)
        st.plotly_chart(fig, width="stretch")

    st.info("**Key Takeaway:** Notice how changing the slope (tilt) and intercept (height) affects how well the line covers the data points.")

# --- Tab 2: Error Visualization ---
with tabs[1]:
    st.header("2. Visualizing the Error (MSE)")
    st.markdown("""
    The **Mean Squared Error (MSE)** measures average squared distance between observed and predicted values.
    Below, the **vertical lines** represent residuals, and the **squares** visualize the "penalty" added by squaring those residuals.
    """)
    
    y_pred = m_manual * X + b_manual
    
    fig_err = go.Figure()
    # Data points
    fig_err.add_trace(go.Scatter(x=X, y=y, mode='markers', name='Data', marker=dict(color='blue')))
    # Line
    line_x = np.array([min(X), max(X)])
    fig_err.add_trace(go.Scatter(x=line_x, y=m_manual * line_x + b_manual, mode='lines', name='Line', line=dict(color='red')))
    
    # Residuals & Squared Errors
    for xi, yi, ypi in zip(X, y, y_pred):
        # Vertical line (residual)
        fig_err.add_trace(go.Scatter(x=[xi, xi], y=[yi, ypi], mode='lines', line=dict(color='gray', dash='dot'), showlegend=False))
        # Square visualization
        err = abs(yi - ypi)
        if err > 0.1: # Only plot squares for visible error
            fig_err.add_shape(type="rect", 
                              x0=xi, x1=xi + err/10, y0=ypi, y1=ypi + (yi-ypi), 
                              fillcolor="rgba(255, 0, 0, 0.2)", line=dict(width=0))

    fig_err.update_layout(title="Residuals & Squared Penalties", xaxis_title="X", yaxis_title="y", height=500)
    st.plotly_chart(fig_err, width="stretch")

    st.success("**Key Takeaway:** Large errors are penalized much more heavily than small ones because they are squared. Try to minimize the 'total area' of the red squares!")

# --- Tab 3: Loss Landscape ---
with tabs[2]:
    st.header("3. The Loss Landscape")
    st.markdown("""
    The Loss Function $J(m, b)$ creates a 3D bowl-shaped surface. Linear Regression is the process of sliding down to the bottom of this bowl.
    """)
    
    # Generate Grid
    m_grid = np.linspace(-2, 6, 40)
    b_grid = np.linspace(-10, 10, 40)
    M, B, Z = compute_grid_loss(X, y, m_grid, b_grid)
    
    col_l, col_r = st.columns(2)
    
    with col_l:
        # Contour Plot
        fig_cont = go.Figure(data=go.Contour(z=Z, x=m_grid, y=b_grid, colorscale='Viridis'))
        fig_cont.add_trace(go.Scatter(x=[m_manual], y=[b_manual], mode='markers', marker=dict(color='white', size=15, symbol='x'), name='Current Pos'))
        fig_cont.update_layout(title="2D Contour Map (Top View)", xaxis_title="Slope (m)", yaxis_title="Intercept (b)")
        st.plotly_chart(fig_cont, width="stretch")
        
    with col_r:
        # 3D Surface
        fig_surf = go.Figure(data=[go.Surface(z=Z, x=m_grid, y=b_grid, colorscale='Viridis', opacity=0.8)])
        fig_surf.add_trace(go.Scatter3d(x=[m_manual], y=[b_manual], z=[current_mse], mode='markers', marker=dict(color='red', size=8), name='Current Pos'))
        fig_surf.update_layout(title="3D Loss Surface", scene=dict(xaxis_title='m', yaxis_title='b', zaxis_title='Loss'))
        st.plotly_chart(fig_surf, width="stretch")

    st.warning("**Key Takeaway:** The optimal solution is the lowest point in the landscape. Notice how the 'X' moves as you adjust the sliders in the first tab.")

# --- Tab 4: Gradient Descent ---
with tabs[3]:
    st.header("4. Gradient Descent Optimization")
    st.markdown("""
    Gradient Descent calculates the "slope" of the loss landscape at the current point and takes a step downhill.
    """)
    
    if st.button("🚀 Run Optimizer"):
        history = run_gradient_descent(X, y, m_manual, b_manual, learning_rate, iterations)
        
        # Plot Path on Contour
        hist_m = [h[0] for h in history]
        hist_b = [h[1] for h in history]
        hist_l = [h[2] for h in history]
        
        fig_path = go.Figure(data=go.Contour(z=Z, x=m_grid, y=b_grid, colorscale='Viridis'))
        fig_path.add_trace(go.Scatter(x=hist_m, y=hist_b, mode='lines+markers', line=dict(color='white'), marker=dict(color='red'), name='GD Path'))
        fig_path.update_layout(title="Learning Trajectory", xaxis_title="Slope (m)", yaxis_title="Intercept (b)")
        st.plotly_chart(fig_path, width="stretch")
        
        st.write("📈 **Final Parameters Found:**")
        st.code(f"m = {hist_m[-1]:.4f}, b = {hist_b[-1]:.4f}, Final Loss = {hist_l[-1]:.4f}")
    else:
        st.write("Click the button to watch the model learn from your current manual starting point.")

    st.info("**Key Takeaway:** The path shows how parameters move iteration by iteration. High learning rates take bigger jumps; low learning rates take tiny steps.")

# --- Tab 5: Learning Rate Experiments ---
with tabs[4]:
    st.header("5. Learning Rate & Convergence")
    st.markdown("""
    Choosing the right **Learning Rate (α)** is critical. Too small and it takes forever to learn; too large and it might explode (diverge).
    """)
    
    rates = [0.001, 0.01, 0.05, 0.15]
    colors = ['blue', 'green', 'orange', 'red']
    
    fig_lr = go.Figure()
    for lr_val, color in zip(rates, colors):
        hist = run_gradient_descent(X, y, 0.0, 0.0, lr_val, 50)
        losses = [h[2] for h in hist]
        fig_lr.add_trace(go.Scatter(y=losses, mode='lines', name=f'LR={lr_val}', line=dict(color=color)))
        
    fig_lr.update_layout(title="Loss vs. Iterations", xaxis_title="Iteration", yaxis_title="MSE Loss", yaxis_type="log")
    st.plotly_chart(fig_lr, width="stretch")
    
    st.error("**Key Takeaway:** Look at the Red line (Large LR) - if it shoots upward, the model is 'diverging' and failing to learn. The Blue line (Small LR) is stable but slow.")

# --- Tab 6: Noise & Robustness ---
with tabs[5]:
    st.header("6. The Impact of Noise & Outliers")
    colA, colB = st.columns(2)
    
    with colA:
        st.markdown("**Without Outliers**")
        X_clean, y_clean = generate_data(n_points, noise_level, False, random_seed)
        hist_clean = run_gradient_descent(X_clean, y_clean, 0, 0, 0.02, 100)
        m_c, b_c = hist_clean[-1][0], hist_clean[-1][1]
        
        fig_c = px.scatter(x=X_clean, y=y_clean, title="Standard Regression")
        fig_c.add_trace(go.Scatter(x=X_clean, y=m_c*X_clean + b_c, mode='lines', line=dict(color='green'), name='Fit'))
        st.plotly_chart(fig_c, width="stretch")
        
    with colB:
        st.markdown("**With Outliers**")
        X_out, y_out = generate_data(n_points, noise_level, True, random_seed)
        hist_out = run_gradient_descent(X_out, y_out, 0, 0, 0.02, 100)
        m_o, b_o = hist_out[-1][0], hist_out[-1][1]
        
        fig_o = px.scatter(x=X_out, y=y_out, title="Regression with Outliers")
        fig_o.add_trace(go.Scatter(x=X_out, y=m_o*X_out + b_o, mode='lines', line=dict(color='red'), name='Fit'))
        # Highlight outliers
        fig_o.add_trace(go.Scatter(x=X_out[-4:], y=y_out[-4:], mode='markers', marker=dict(color='red', size=12, symbol='star'), name='Outliers'))
        st.plotly_chart(fig_o, width="stretch")

    st.warning(f"""
    **Analysis:**
    - Clean Slope: `{m_c:.2f}`
    - Outlier Slope: `{m_o:.2f}`
    - Key Difference: `{abs(m_c - m_o):.2f}`
    
    **Key Takeaway:** Notice how the red line "tilts" towards the star-shaped outliers. Linear Regression is sensitive to outliers because it tries to minimize the *squared* error, and a distant point creates a massive square!
    """)
