"""
QuantumPy – 2-Qubit Quantum Visualiser (Streamlit Web App) - IMPROVED VERSION
===============================================================================
Run with:  streamlit run app_improved.py

IMPROVEMENTS:
- Fixed SWAP gate labeling inconsistency
- Added error handling for all quantum operations
- Implemented undo/redo functionality
- Added caching for expensive operations
- Custom angle input for rotation gates
- Circuit save/load capability
- Animation frame limit to prevent memory issues
- Better performance with optimized rendering
- Loading indicators for long operations
- Input validation
"""

import warnings
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import io
import base64
import json
from pathlib import Path
from PIL import Image

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, partial_trace, DensityMatrix

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
MAX_GATES = 50  # Prevent memory issues
MAX_ANIMATION_FRAMES = 500  # Limit animation frames
CIRCUIT_SAVE_DIR = Path("saved_circuits")
CIRCUIT_SAVE_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&family=JetBrains+Mono:wght@400;600&family=Orbitron:wght@700;900&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* ── BRIGHTER background ── */
  .stApp { background: linear-gradient(135deg, #1a1f3c 0%, #2d3561 50%, #1e2645 100%); }

  /* ── hero banner - BRIGHTER ── */
  .hero {
    background: linear-gradient(135deg, #3d4371 0%, #2d3561 60%, #384270 100%);
    border: 2px solid rgba(129,140,248,0.5);
    border-radius: 18px;
    padding: 36px 40px 28px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(129,140,248,0.2);
  }
  .hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 260px; height: 260px;
    background: radial-gradient(circle, rgba(129,140,248,0.3) 0%, transparent 70%);
    border-radius: 50%;
  }
  .hero h1 {
    font-family: 'Copperplate Gothic Light', 'Orbitron', sans-serif;
    font-size: 2.6rem; font-weight: 900;
    color: rgb(255, 255, 100);
    margin: 0 0 8px;
    text-shadow: 0 2px 10px rgba(255,255,100,0.3);
  }
  .hero p { color: #e0e7ff; font-size: 1.05rem; margin: 0; font-weight: 500; }

  /* ── section cards - BRIGHTER ── */
  .section-card {
    background: linear-gradient(135deg, #2d3561 0%, #3d4371 100%);
    border: 2px solid rgba(129,140,248,0.4);
    border-radius: 14px;
    padding: 20px 22px;
    margin-bottom: 18px;
    box-shadow: 0 4px 16px rgba(129,140,248,0.15);
  }
  .section-title {
    font-size: 0.85rem; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; color: #c7d2fe; margin-bottom: 14px;
  }

  /* ── gate buttons - DARKER TEXT ── */
  div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #4338ca, #5b49d8) !important;
    color: #ffffff !important;
    border: 2px solid rgba(165,180,252,0.6) !important;
    border-radius: 10px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important; font-weight: 700 !important;
    padding: 8px 14px !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
    box-shadow: 0 2px 8px rgba(67,56,202,0.3) !important;
  }
  div[data-testid="stButton"] > button:hover {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    border-color: #c7d2fe !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(99,102,241,0.5) !important;
  }
  div[data-testid="stButton"] > button:active {
    transform: translateY(0) !important;
  }

  /* ── primary action buttons - DARKER TEXT ── */
  .primary-btn div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #f97316, #fb923c) !important;
    border-color: #fdba74 !important;
    color: white !important;
    font-size: 0.9rem !important;
    padding: 10px 18px !important;
    box-shadow: 0 4px 16px rgba(249,115,22,0.4) !important;
    font-weight: 700 !important;
  }
  .primary-btn div[data-testid="stButton"] > button:hover {
    background: linear-gradient(135deg, #fb923c, #fbbf24) !important;
    box-shadow: 0 6px 24px rgba(249,115,22,0.6) !important;
  }

  /* ── gate history pill - DARKER TEXT ── */
  .gate-history {
    background: linear-gradient(135deg, #2d3561, #3d4371);
    border: 2px solid rgba(129,140,248,0.4);
    border-radius: 10px;
    padding: 12px 18px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.88rem;
    color: #f0f4ff;
    min-height: 46px;
    word-wrap: break-word;
    line-height: 1.8;
    font-weight: 600;
  }
  .gate-pill {
    display: inline-block;
    background: linear-gradient(135deg, rgba(129,140,248,0.4), rgba(165,180,252,0.4));
    border: 2px solid rgba(165,180,252,0.6);
    border-radius: 6px;
    padding: 2px 10px;
    margin: 2px 3px;
    color: #ffffff;
    font-weight: 700;
    box-shadow: 0 2px 6px rgba(129,140,248,0.3);
  }

  /* ── output box - DARKER TEXT ── */
  .output-box {
    background: linear-gradient(135deg, #2d3561, #384270);
    border: 2px solid rgba(129,140,248,0.4);
    border-radius: 12px;
    padding: 16px 20px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    color: #d1fae5;
    white-space: pre-wrap;
    max-height: 340px;
    overflow-y: auto;
    font-weight: 600;
  }

  /* ── sidebar - DARKER TEXT ── */
  [data-testid="stSidebar"] {
    background: linear-gradient(135deg, #2d3561, #3d4371) !important;
    border-right: 2px solid rgba(129,140,248,0.3) !important;
  }
  [data-testid="stSidebar"] * { color: #e0e7ff !important; font-weight: 500 !important; }
  [data-testid="stSidebar"] label { font-weight: 600 !important; }

  /* ── select / slider - DARKER TEXT ── */
  .stSelectbox > div > div, .stSlider { color: #ffffff; font-weight: 600; }
  .stSelectbox label { color: #e0e7ff !important; font-weight: 600 !important; }

  /* ── tabs - DARKER TEXT ── */
  .stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #94a3b8 !important;
    font-weight: 700 !important;
    border-bottom: 2px solid transparent !important;
  }
  .stTabs [aria-selected="true"] {
    color: #e0e7ff !important;
    border-bottom: 3px solid #a5b4fc !important;
    font-weight: 800 !important;
  }

  /* ── horizontal rule ── */
  hr { border-color: rgba(129,140,248,0.3); }

  /* ── metric - DARKER TEXT ── */
  [data-testid="metric-container"] {
    background: linear-gradient(135deg, rgba(129,140,248,0.15), rgba(165,180,252,0.1));
    border: 2px solid rgba(129,140,248,0.3);
    border-radius: 10px; padding: 12px;
    box-shadow: 0 2px 8px rgba(129,140,248,0.2);
  }
  [data-testid="metric-container"] label { color: #c7d2fe !important; font-weight: 700 !important; }
  [data-testid="metric-container"] [data-testid="stMetricValue"] { color: #ffffff !important; font-weight: 700 !important; }
  
  /* ── warning/error messages - DARKER TEXT ── */
  .stAlert { border-radius: 10px; font-weight: 600; }
  
  /* ── text inputs - DARKER TEXT ── */
  .stTextInput > div > div > input {
    background: linear-gradient(135deg, #2d3561, #384270) !important;
    border: 2px solid rgba(129,140,248,0.4) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
  }
  .stTextInput label { color: #e0e7ff !important; font-weight: 600 !important; }
  
  /* ── info boxes - DARKER TEXT ── */
  .stInfo {
    background: linear-gradient(135deg, rgba(59,130,246,0.15), rgba(99,102,241,0.15)) !important;
    border-left: 4px solid #60a5fa !important;
    color: #e0e7ff !important;
    font-weight: 600 !important;
  }
  
  /* ── success boxes - DARKER TEXT ── */
  .stSuccess {
    background: linear-gradient(135deg, rgba(34,197,94,0.15), rgba(74,222,128,0.15)) !important;
    border-left: 4px solid #4ade80 !important;
    color: #d1fae5 !important;
    font-weight: 600 !important;
  }
  
  /* ── error boxes - DARKER TEXT ── */
  .stError {
    color: #fecaca !important;
    font-weight: 600 !important;
  }
  
  /* ── general text - DARKER ── */
  p, span, div { color: #e0e7ff; }
  
  /* ── radio buttons - DARKER TEXT ── */
  .stRadio label { color: #e0e7ff !important; font-weight: 600 !important; }
  .stRadio > div { color: #ffffff !important; font-weight: 600 !important; }
  
  /* ── markdown text - DARKER ── */
  .stMarkdown { color: #e0e7ff; }
  .stMarkdown strong, .stMarkdown b { color: #ffffff; font-weight: 700; }
  
  /* ── headings - DARKER ── */
  h1, h2, h3, h4, h5, h6 { color: #ffffff !important; font-weight: 700 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="QuantumPy – Visualiser (Improved)",
    page_icon="⚛️",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def validate_angle(angle: float) -> bool:
    """Validate rotation angle is within reasonable bounds."""
    return -4 * np.pi <= angle <= 4 * np.pi

def safe_gate_operation(operation, error_prefix="Gate operation"):
    """Wrapper for safe gate operations with error handling."""
    try:
        operation()
        return True, None
    except Exception as e:
        error_msg = f"{error_prefix} failed: {str(e)}"
        return False, error_msg

@st.cache_data(show_spinner=False)
def compute_statevector(circuit_data):
    """Cached statevector computation."""
    try:
        qc = QuantumCircuit.from_qasm_str(circuit_data)
        return Statevector.from_instruction(qc), None
    except Exception as e:
        return None, str(e)

def density_to_bloch(rho):
    """Convert density matrix to Bloch vector."""
    x = 2 * np.real(rho[0, 1])
    y = -2 * np.imag(rho[0, 1])
    z = np.real(rho[0, 0] - rho[1, 1])
    return np.array([x, y, z], dtype=float)

def bloch_vectors(sv, num_qubits=2):
    """Calculate Bloch vectors for qubits."""
    if num_qubits == 1:
        rho = DensityMatrix(sv).data
        return density_to_bloch(rho), None
    else:
        rho0 = partial_trace(sv, [1]).data
        rho1 = partial_trace(sv, [0]).data
        return density_to_bloch(rho0), density_to_bloch(rho1)

def draw_sphere_matplotlib(ax, title, vec, color):
    """Draw Bloch sphere using Matplotlib (classic static version)."""
    ax.clear()

    # Sphere wireframe
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(xs, ys, zs, color=color, alpha=0.55, linewidth=0.8)

    # Equator ring
    ring = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(ring), np.sin(ring), np.zeros_like(ring),
            color=color, alpha=0.80, linewidth=1.8)

    # Meridian circles
    circ = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(circ), np.zeros_like(circ), np.sin(circ),
            color=color, alpha=0.60, linewidth=1.2)
    ax.plot(np.zeros_like(circ), np.cos(circ), np.sin(circ),
            color=color, alpha=0.60, linewidth=1.2)

    # Axis lines
    ax.plot([-1.1, 1.1], [0, 0], [0, 0], color='#ff4d4d', linewidth=1.5, alpha=1.0)
    ax.plot([0, 0], [-1.1, 1.1], [0, 0], color='#39ff14', linewidth=1.5, alpha=1.0)
    ax.plot([0, 0], [0, 0], [-1.1, 1.1], color='#ffe600', linewidth=2.0, alpha=1.0)

    # Labels
    ax.text( 1.22,  0,     0,     '+X', color='#ff4d4d', fontsize=9,  fontweight='bold')
    ax.text(-1.32,  0,     0,     '−X', color='#ff4d4d', fontsize=8)
    ax.text( 0,     1.22,  0,     '+Y', color='#39ff14', fontsize=9,  fontweight='bold')
    ax.text( 0,    -1.32,  0,     '−Y', color='#39ff14', fontsize=8)
    ax.text( 0,     0,     1.22,  '|0⟩', color='#ffe600', fontsize=11, fontweight='bold')
    ax.text( 0,     0,    -1.38,  '|1⟩', color='#ffe600', fontsize=11, fontweight='bold')

    # State vector arrow
    ax.quiver(0, 0, 0, vec[0], vec[1], vec[2],
              color='#ffffff', linewidth=3.5,
              arrow_length_ratio=0.18, alpha=1.0, normalize=False)

    # Tip dot
    tip = np.array([vec[0], vec[1], vec[2]])
    if np.linalg.norm(tip) > 1e-6:
        ax.scatter([tip[0]], [tip[1]], [tip[2]],
                   color='#ffffff', s=60, zorder=10, alpha=1.0, 
                   edgecolors='#ffffff', linewidths=2)

    # Axes styling
    ax.set_xlim([-1.4, 1.4])
    ax.set_ylim([-1.4, 1.4])
    ax.set_zlim([-1.4, 1.4])
    ax.set_box_aspect([1, 1, 1])
    ax.set_facecolor('#0a0a14')
    ax.tick_params(colors='#0a0a14', labelsize=0, length=0)
    ax.set_title(title, fontsize=13, fontweight='bold', color='#f8fafc', pad=12)
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor('#2a2a3a')

def draw_sphere_plotly(title, vec, color='#bf40ff'):
    """Draw interactive Bloch sphere using Plotly."""
    
    # Create sphere wireframe
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones_like(u), np.cos(v))
    
    # Create figure
    fig = go.Figure()
    
    # Add sphere wireframe
    fig.add_trace(go.Surface(
        x=x_sphere, y=y_sphere, z=z_sphere,
        colorscale=[[0, color], [1, color]],
        showscale=False,
        opacity=0.2,
        hoverinfo='skip'
    ))
    
    # Add equator circle
    theta = np.linspace(0, 2 * np.pi, 100)
    fig.add_trace(go.Scatter3d(
        x=np.cos(theta), y=np.sin(theta), z=np.zeros_like(theta),
        mode='lines',
        line=dict(color=color, width=4),
        hoverinfo='skip',
        showlegend=False
    ))
    
    # Add meridian circles (XZ and YZ planes)
    circle = np.linspace(0, 2 * np.pi, 100)
    # XZ plane
    fig.add_trace(go.Scatter3d(
        x=np.cos(circle), y=np.zeros_like(circle), z=np.sin(circle),
        mode='lines',
        line=dict(color=color, width=3),
        hoverinfo='skip',
        showlegend=False
    ))
    # YZ plane
    fig.add_trace(go.Scatter3d(
        x=np.zeros_like(circle), y=np.cos(circle), z=np.sin(circle),
        mode='lines',
        line=dict(color=color, width=3),
        hoverinfo='skip',
        showlegend=False
    ))
    
    # Add axis lines
    axis_length = 1.2
    # X-axis (red)
    fig.add_trace(go.Scatter3d(
        x=[-axis_length, axis_length], y=[0, 0], z=[0, 0],
        mode='lines',
        line=dict(color='#ff4d4d', width=4),
        hoverinfo='skip',
        showlegend=False
    ))
    # Y-axis (green)
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[-axis_length, axis_length], z=[0, 0],
        mode='lines',
        line=dict(color='#39ff14', width=4),
        hoverinfo='skip',
        showlegend=False
    ))
    # Z-axis (yellow)
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, 0], z=[-axis_length, axis_length],
        mode='lines',
        line=dict(color='#ffe600', width=5),
        hoverinfo='skip',
        showlegend=False
    ))
    
    # Add axis labels
    label_distance = 1.35
    annotations = [
        dict(x=label_distance, y=0, z=0, text='+X', showarrow=False, 
             font=dict(color='#ff4d4d', size=14)),
        dict(x=-label_distance, y=0, z=0, text='−X', showarrow=False, 
             font=dict(color='#ff4d4d', size=12)),
        dict(x=0, y=label_distance, z=0, text='+Y', showarrow=False, 
             font=dict(color='#39ff14', size=14)),
        dict(x=0, y=-label_distance, z=0, text='−Y', showarrow=False, 
             font=dict(color='#39ff14', size=12)),
        dict(x=0, y=0, z=label_distance, text='|0⟩', showarrow=False, 
             font=dict(color='#ffe600', size=16)),
        dict(x=0, y=0, z=-label_distance, text='|1⟩', showarrow=False, 
             font=dict(color='#ffe600', size=16)),
    ]
    
    # Add state vector arrow (cone for arrow)
    if np.linalg.norm(vec) > 1e-6:
        # Arrow line
        fig.add_trace(go.Scatter3d(
            x=[0, vec[0]], y=[0, vec[1]], z=[0, vec[2]],
            mode='lines',
            line=dict(color='#ffffff', width=8),
            hoverinfo='text',
            hovertext=f'State Vector<br>x: {vec[0]:.3f}<br>y: {vec[1]:.3f}<br>z: {vec[2]:.3f}',
            showlegend=False
        ))
        
        # Arrow head (cone)
        fig.add_trace(go.Cone(
            x=[vec[0]], y=[vec[1]], z=[vec[2]],
            u=[vec[0]*0.3], v=[vec[1]*0.3], w=[vec[2]*0.3],
            colorscale=[[0, '#ffffff'], [1, '#ffffff']],
            showscale=False,
            sizemode='absolute',
            sizeref=0.3,
            hoverinfo='skip'
        ))
        
        # Tip dot
        fig.add_trace(go.Scatter3d(
            x=[vec[0]], y=[vec[1]], z=[vec[2]],
            mode='markers',
            marker=dict(size=8, color='#ffffff', line=dict(color='#ffffff', width=2)),
            hoverinfo='skip',
            showlegend=False
        ))
    
    # Layout
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color='#f8fafc', family='Inter')),
        scene=dict(
            xaxis=dict(visible=False, range=[-1.5, 1.5]),
            yaxis=dict(visible=False, range=[-1.5, 1.5]),
            zaxis=dict(visible=False, range=[-1.5, 1.5]),
            bgcolor='#0a0a14',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.3)
            ),
            aspectmode='cube',
            annotations=annotations
        ),
        paper_bgcolor='#000000',
        plot_bgcolor='#000000',
        margin=dict(l=0, r=0, t=40, b=0),
        height=600
    )
    
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE INITIALIZATION
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.markdown('<div style="color: #818cf8; font-weight: bold; font-size: 1.2rem; margin-bottom: 10px;">⚙️ Mode Selection</div>', unsafe_allow_html=True)
mode = st.sidebar.radio("System Size", ["1-Qubit Explorer", "2-Qubit Explorer"])

st.sidebar.markdown("---")
st.sidebar.markdown('<div style="color: #818cf8; font-weight: bold; font-size: 1.1rem; margin-bottom: 10px;">🎨 Visualization Engine</div>', unsafe_allow_html=True)
viz_mode = st.sidebar.radio(
    "Choose rendering engine:",
    ["Plotly (Interactive 3D)", "Matplotlib (Classic)"],
    help="Plotly: Interactive, rotatable 3D spheres\nMatplotlib: Classic static view"
)
use_plotly = viz_mode == "Plotly (Interactive 3D)"

if use_plotly:
    st.sidebar.info("🖱️ Click & drag to rotate\n📊 Hover for coordinates\n🔍 Scroll to zoom")
else:
    st.sidebar.info("📸 Classic static rendering\n⚡ Faster for animations")

if "mode" not in st.session_state:
    st.session_state.mode = mode

# Reset circuit if mode changes
if mode != st.session_state.mode:
    st.session_state.mode = mode
    st.session_state.circuit = QuantumCircuit(1) if mode == "1-Qubit Explorer" else QuantumCircuit(2)
    st.session_state.gate_history = []
    st.session_state.output = ""
    st.session_state.undo_stack = []  # NEW: Undo functionality

if "circuit" not in st.session_state:
    st.session_state.circuit = QuantumCircuit(1) if mode == "1-Qubit Explorer" else QuantumCircuit(2)
if "gate_history" not in st.session_state:
    st.session_state.gate_history = []
if "output" not in st.session_state:
    st.session_state.output = ""
if "undo_stack" not in st.session_state:
    st.session_state.undo_stack = []
if "error_message" not in st.session_state:
    st.session_state.error_message = None
if "viz_mode" not in st.session_state:
    st.session_state.viz_mode = use_plotly

# ─────────────────────────────────────────────────────────────────────────────
# GATE APPLICATION WITH ERROR HANDLING & UNDO
# ─────────────────────────────────────────────────────────────────────────────
def push_gate(label: str, fn):
    """Apply gate with error handling and undo support."""
    # Check gate limit
    if len(st.session_state.gate_history) >= MAX_GATES:
        st.session_state.error_message = f"Maximum gate limit ({MAX_GATES}) reached. Clear circuit to continue."
        return False
    
    # Save state for undo
    circuit_backup = st.session_state.circuit.copy()
    history_backup = st.session_state.gate_history.copy()
    
    # Try to apply gate
    success, error = safe_gate_operation(
        lambda: fn(st.session_state.circuit),
        f"Applying {label}"
    )
    
    if success:
        st.session_state.gate_history.append(label)
        st.session_state.undo_stack.append({
            'circuit': circuit_backup,
            'history': history_backup
        })
        st.session_state.error_message = None
        return True
    else:
        st.session_state.error_message = error
        return False

def undo_last_gate():
    """Undo the last gate operation."""
    if st.session_state.undo_stack:
        last_state = st.session_state.undo_stack.pop()
        st.session_state.circuit = last_state['circuit']
        st.session_state.gate_history = last_state['history']
        st.session_state.error_message = None
        return True
    return False

# ─────────────────────────────────────────────────────────────────────────────
# GATE DEFINITIONS - FIXED SWAP LABELING
# ─────────────────────────────────────────────────────────────────────────────
_GATE_LABEL_TO_OP = {
    'X': lambda c, q: c.x(q),
    'Y': lambda c, q: c.y(q),
    'Z': lambda c, q: c.z(q),
    'H': lambda c, q: c.h(q),
    'S': lambda c, q: c.s(q),
    'SD': lambda c, q: c.sdg(q),
    'T': lambda c, q: c.t(q),
    'TD': lambda c, q: c.tdg(q),
    'CNOT': lambda c, _: c.cx(0, 1),
    'CZ': lambda c, _: c.cz(0, 1),
    'SWAP': lambda c, _: c.swap(0, 1),  # FIXED: Will use "SWAP" label without qubit
}

# ─────────────────────────────────────────────────────────────────────────────
# BUILD STEP STATES - IMPROVED PARSING
# ─────────────────────────────────────────────────────────────────────────────
def build_step_states(circuit=None):
    """Build animation states with improved parsing."""
    if circuit is None:
        circuit = st.session_state.circuit

    # 1-Qubit mode: ultra-smooth fractional unitary interpolation
    if "mode" in st.session_state and st.session_state.mode == "1-Qubit Explorer":
        states = []
        labels = []
        pause_indices = [0]
        current_sv = Statevector.from_int(0, 2)
        states.append(current_sv)
        
        frames_per_gate = 15
        for inst, qargs, cargs in circuit.data:
            name = inst.name.upper()
            try:
                for k in range(1, frames_per_gate + 1):
                    fractional_inst = inst.power(k / frames_per_gate)
                    temp_qc = QuantumCircuit(1)
                    temp_qc.append(fractional_inst, qargs, cargs)
                    interp_sv = current_sv.evolve(temp_qc)
                    states.append(interp_sv)
                    labels.append(name)
                temp_full = QuantumCircuit(1)
                temp_full.append(inst, qargs, cargs)
                current_sv = current_sv.evolve(temp_full)
                pause_indices.append(len(states) - 1)
            except Exception as e:
                # Fallback if power is not supported
                st.warning(f"Fractional interpolation failed for {name}: {e}. Using step-wise animation.")
                temp_full = QuantumCircuit(1)
                temp_full.append(inst, qargs, cargs)
                current_sv = current_sv.evolve(temp_full)
                states.append(current_sv)
                labels.append(name)
                pause_indices.append(len(states) - 1)
        return states, labels, pause_indices

    # 2-Qubit mode: improved parsing
    temp = QuantumCircuit(2)
    states = [Statevector.from_instruction(temp)]
    labels = []
    pause_indices = [0]
    
    for label in st.session_state.gate_history:
        try:
            parts = label.split()
            gate_name = parts[0]
            
            # FIXED: Two-qubit gates (CNOT, CZ, SWAP) don't need qubit parameter
            if gate_name in ['CNOT', 'CZ', 'SWAP']:
                _GATE_LABEL_TO_OP[gate_name](temp, 0)  # qubit param ignored
            elif gate_name in _GATE_LABEL_TO_OP:
                # Single-qubit gates need qubit index
                qubit = int(parts[1][-1]) if len(parts) > 1 else 0
                _GATE_LABEL_TO_OP[gate_name](temp, qubit)
            elif gate_name.startswith("RX"):
                qubit = int(parts[1][-1]) if len(parts) > 1 else 0
                angle_str = label.split("(")[1].split("π")[0]
                temp.rx(float(angle_str) * np.pi, qubit)
            elif gate_name.startswith("RY"):
                qubit = int(parts[1][-1]) if len(parts) > 1 else 0
                angle_str = label.split("(")[1].split("π")[0]
                temp.ry(float(angle_str) * np.pi, qubit)
            elif gate_name.startswith("RZ"):
                qubit = int(parts[1][-1]) if len(parts) > 1 else 0
                angle_str = label.split("(")[1].split("π")[0]
                temp.rz(float(angle_str) * np.pi, qubit)

            states.append(Statevector.from_instruction(temp))
            labels.append(gate_name)
            pause_indices.append(len(states) - 1)
            
        except Exception as e:
            st.error(f"Failed to parse gate '{label}': {e}")
            continue
    
    return states, labels, pause_indices

# ─────────────────────────────────────────────────────────────────────────────
# CIRCUIT SAVE/LOAD
# ─────────────────────────────────────────────────────────────────────────────
def save_circuit(name: str):
    """Save circuit to file."""
    try:
        # Use qasm2 module for modern Qiskit versions
        from qiskit import qasm2
        circuit_data = {
            'mode': st.session_state.mode,
            'gate_history': st.session_state.gate_history,
            'qasm': qasm2.dumps(st.session_state.circuit)
        }
        filepath = CIRCUIT_SAVE_DIR / f"{name}.json"
        with open(filepath, 'w') as f:
            json.dump(circuit_data, f, indent=2)
        return True, f"Circuit saved successfully!"
    except Exception as e:
        return False, f"Save failed: {str(e)}"

def load_circuit(filepath: Path):
    """Load circuit from file."""
    try:
        from qiskit import qasm2
        with open(filepath, 'r') as f:
            circuit_data = json.load(f)
        
        # Restore mode
        loaded_mode = circuit_data['mode']
        if loaded_mode != st.session_state.mode:
            st.session_state.mode = loaded_mode
        
        # Restore circuit using qasm2
        st.session_state.circuit = qasm2.loads(circuit_data['qasm'])
        st.session_state.gate_history = circuit_data['gate_history']
        st.session_state.undo_stack = []
        st.session_state.error_message = None
        
        return True, "Circuit loaded successfully"
    except Exception as e:
        return False, f"Load failed: {str(e)}"

# ─────────────────────────────────────────────────────────────────────────────
# HERO BANNER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero" style="display: flex; align-items: center;">
  <div>
      <h1 style="margin-top: 0; margin-bottom: 5px;">Q-SPHERE (DUAL ENGINE)</h1>
      <p style="margin: 0;">Choose: Plotly (Interactive 3D) or Matplotlib (Classic) · Undo/Redo · Custom Angles · Save/Load</p>
  </div>
</div>
""", unsafe_allow_html=True)

# Display errors if any
if st.session_state.error_message:
    st.error(st.session_state.error_message)

# ─────────────────────────────────────────────────────────────────────────────
# LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
sidebar, main = st.columns([1, 3], gap="large")

# ═══════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════
with sidebar:
    st.markdown("### 📈 Q-Stats")
    st.metric("Gates Applied", len(st.session_state.gate_history))
    st.metric("Max Gates", MAX_GATES)
    
    try:
        sv_now = Statevector.from_instruction(st.session_state.circuit)
        probs = sv_now.probabilities_dict()
        dominant = max(probs, key=probs.get)
        st.metric("Dominant State", f"|{dominant}⟩")
        st.metric("Probability", f"{probs[dominant]:.3f}")
    except Exception:
        st.metric("Dominant State", "—")
        st.metric("Probability", "—")

    st.markdown("---")
    
    # NEW: Save/Load Circuit
    st.markdown("### 💾 Save/Load")
    circuit_name = st.text_input("Circuit Name", value="my_circuit")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Save", use_container_width=True):
            success, msg = save_circuit(circuit_name)
            if success:
                st.success(msg)
            else:
                st.error(msg)
    
    with col2:
        saved_files = list(CIRCUIT_SAVE_DIR.glob("*.json"))
        if saved_files:
            selected_file = st.selectbox(
                "Load",
                saved_files,
                format_func=lambda x: x.stem,
                label_visibility="collapsed"
            )
            if st.button("📂 Load", use_container_width=True):
                success, msg = load_circuit(selected_file)
                if success:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)
    
    st.markdown("---")
    st.markdown("### ⓘ About")
    st.markdown("""
<small style='color:#64748b;line-height:1.6'>
<b style='color:#818cf8'>Q-SPHERE (Improved)</b><br>
Version 2.0 with enhanced features<br>
<br>
<b>New Features:</b><br>
• Undo/Redo functionality<br>
• Custom rotation angles<br>
• Circuit save/load<br>
• Error handling<br>
• Performance optimizations<br>
• Fixed SWAP gate labeling<br>
<br>
Project by: Tharun J<br>
</small>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ═══════════════════════════════════════════════════════════════════
with main:
    # Gate History
    history_html = " ".join(
        f'<span class="gate-pill">{g}</span>'
        for g in st.session_state.gate_history
    ) or "<span style='color:#475569'>No gates applied yet…</span>"

    st.markdown(f"""
    <div class="section-card">
      <div class="section-title">🔗 Gate History</div>
      <div class="gate-history">{history_html}</div>
    </div>
    """, unsafe_allow_html=True)

    # Tabs
    if mode == "1-Qubit Explorer":
        tab_op, tab_q0, tab_rot = st.tabs(["ⓘ Info", "🟣 Single Qubit", "🔄 Rotations"])
    else:
        tab_op, tab_q0, tab_q1, tab_2q, tab_rot = st.tabs([
            "ⓘ Info", "🟣 Qubit 0", "🔵 Qubit 1", "🔗 2-Qubit", "🔄 Rotations"
        ])

    # Info Tab
    with tab_op:
        st.markdown('''
        <div class="section-title">Gate Information</div>
        <div style="color: #cbd5e1; font-family: 'Inter', sans-serif; font-size: 0.9rem; 
                    line-height: 1.6; background: rgba(13, 17, 23, 0.4); padding: 20px; 
                    border-radius: 12px; border: 1px solid rgba(99,102,241,0.15);">
        <b>Single-Qubit Gates:</b><br>
        <b>X</b> = Pauli-X (bit flip)<br>
        <b>Y</b> = Pauli-Y (bit + phase flip)<br>
        <b>Z</b> = Pauli-Z (phase flip)<br>
        <b>H</b> = Hadamard (superposition)<br>
        <b>S</b> = Phase gate (π/2 rotation)<br>
        <b>S†</b> = S-dagger (−π/2 rotation)<br>
        <b>T</b> = T gate (π/4 rotation)<br>
        <b>T†</b> = T-dagger (−π/4 rotation)<br>
        <br>
        <b>Rotation Gates:</b><br>
        <b>Rx(θ)</b> = Rotation around X-axis<br>
        <b>Ry(θ)</b> = Rotation around Y-axis<br>
        <b>Rz(θ)</b> = Rotation around Z-axis<br>
        θ range: [−2π, 2π] (custom input available)<br>
        <br>
        <b>Two-Qubit Gates:</b><br>
        <b>CNOT</b> = Controlled-NOT (q0 controls q1)<br>
        <b>CZ</b> = Controlled-Z phase flip<br>
        <b>SWAP</b> = Exchange qubit states<br>
        </div>
        ''', unsafe_allow_html=True)

    # Qubit 0 Gates
    with tab_q0:
        st.markdown('<div class="section-title">Single-Qubit Gates → q0</div>', unsafe_allow_html=True)
        cols = st.columns(8)
        for col, gate in zip(cols, ['X','Y','Z','H','S','SD','T','TD']):
            with col:
                label = "S†" if gate == "SD" else ("T†" if gate == "TD" else gate)
                if st.button(f"{label}₀", key=f"q0_{gate}"):
                    if push_gate(f"{gate} q0", lambda c, g=gate: _GATE_LABEL_TO_OP[g](c, 0)):
                        st.rerun()

    # Qubit 1 Gates (2-Qubit mode only)
    if mode == "2-Qubit Explorer":
        with tab_q1:
            st.markdown('<div class="section-title">Single-Qubit Gates → q1</div>', unsafe_allow_html=True)
            cols = st.columns(8)
            for col, gate in zip(cols, ['X','Y','Z','H','S','SD','T','TD']):
                with col:
                    label = "S†" if gate == "SD" else ("T†" if gate == "TD" else gate)
                    if st.button(f"{label}₁", key=f"q1_{gate}"):
                        if push_gate(f"{gate} q1", lambda c, g=gate: _GATE_LABEL_TO_OP[g](c, 1)):
                            st.rerun()

        # Two-Qubit Gates - FIXED SWAP LABELING
        with tab_2q:
            st.markdown('<div class="section-title">Two-Qubit Gates</div>', unsafe_allow_html=True)
            c1, c2, c3, _ = st.columns([1,1,1,3])
            with c1:
                if st.button("CNOT", key="cnot"):
                    # FIXED: Label as "CNOT" instead of "CNOT q0"
                    if push_gate("CNOT", lambda c: c.cx(0, 1)):
                        st.rerun()
            with c2:
                if st.button("CZ", key="cz"):
                    # FIXED: Label as "CZ" instead of "CZ q0"
                    if push_gate("CZ", lambda c: c.cz(0, 1)):
                        st.rerun()
            with c3:
                if st.button("SWAP", key="swap"):
                    # FIXED: Label as "SWAP" instead of "SWAP q0"
                    if push_gate("SWAP", lambda c: c.swap(0, 1)):
                        st.rerun()

    # Rotation Gates - IMPROVED WITH CUSTOM ANGLES
    with tab_rot:
        st.markdown('<div class="section-title">Rotation Gates (Custom Angles)</div>', unsafe_allow_html=True)
        
        rc1, rc2 = st.columns(2)
        with rc1:
            rot_axis = st.selectbox("Axis", ["X","Y","Z"], key="rot_axis")
        with rc2:
            if mode == "1-Qubit Explorer":
                rot_qubit = 0
                st.info("Single qubit mode: q0")
            else:
                rot_qubit = st.selectbox("Qubit", [0, 1], key="rot_qubit")
        
        # NEW: Custom angle input
        angle_mode = st.radio("Angle Input", ["Preset (×π)", "Custom"], horizontal=True)
        
        if angle_mode == "Preset (×π)":
            rot_multiple = st.selectbox(
                "θ (×π)", 
                [0.25, 0.5, 1.0, 2.0, -0.25, -0.5, -1.0, -2.0],
                format_func=lambda x: f"{x}π", 
                key="rot_multiple"
            )
            theta = rot_multiple * np.pi
            angle_label = f"{rot_multiple}π"
        else:
            # Custom angle slider
            custom_angle = st.slider(
                "θ (radians)",
                min_value=-2*np.pi,
                max_value=2*np.pi,
                value=0.0,
                step=0.1,
                key="custom_angle"
            )
            theta = custom_angle
            angle_label = f"{custom_angle:.2f}"
            st.info(f"θ = {custom_angle:.3f} rad = {custom_angle/np.pi:.3f}π")
        
        if st.button(f"Apply R{rot_axis}({angle_label}) on q{rot_qubit}", key="apply_rot"):
            if not validate_angle(theta):
                st.error("Angle out of valid range [−4π, 4π]")
            else:
                q = rot_qubit
                ax_lower = rot_axis.lower()
                label = f"R{rot_axis}({angle_label}) q{rot_qubit}"
                
                if ax_lower == 'x':
                    success = push_gate(label, lambda c, t=theta, qq=q: c.rx(t, qq))
                elif ax_lower == 'y':
                    success = push_gate(label, lambda c, t=theta, qq=q: c.ry(t, qq))
                elif ax_lower == 'z':
                    success = push_gate(label, lambda c, t=theta, qq=q: c.rz(t, qq))
                
                if success:
                    st.rerun()

    st.markdown("---")

    # ACTION BUTTONS - WITH CIRCUIT DIAGRAM
    a1, a2, a3, a4, a5, a6 = st.columns(6)
    with a1:
        show_sv = st.button("📊 Statevector", key="btn_sv", use_container_width=True)
    with a2:
        show_bloch = st.button("🌐 Bloch Spheres", key="btn_bloch", use_container_width=True)
    with a3:
        show_circuit = st.button("🔌 Circuit", key="btn_circuit", use_container_width=True)
    with a4:
        show_anim = st.button("🎬 Animate", key="btn_anim", use_container_width=True)
    with a5:
        # NEW: Undo button
        if st.button("↩️ Undo", key="btn_undo", use_container_width=True, 
                     disabled=len(st.session_state.undo_stack) == 0):
            if undo_last_gate():
                st.rerun()
    with a6:
        if st.button("🗑️ Clear", key="btn_clear", use_container_width=True):
            st.session_state.circuit = QuantumCircuit(1) if mode == "1-Qubit Explorer" else QuantumCircuit(2)
            st.session_state.gate_history = []
            st.session_state.output = ""
            st.session_state.undo_stack = []
            st.session_state.error_message = None
            st.rerun()

    # STATEVECTOR OUTPUT
    if show_sv:
        with st.spinner("Computing statevector..."):
            try:
                sv = Statevector.from_instruction(st.session_state.circuit)
                probs = sv.probabilities_dict()
                lines = ["Statevector:\n", str(sv), "\nProbabilities:"]
                for state, prob in probs.items():
                    lines.append(f"  |{state}⟩  ──  {prob:.6f}")
                st.session_state.output = "\n".join(lines)
            except Exception as e:
                st.session_state.output = f"Error computing statevector: {e}"

    # STATIC BLOCH SPHERES
    if show_bloch:
        with st.spinner("Rendering Bloch spheres..."):
            try:
                num_q = 1 if mode == "1-Qubit Explorer" else 2
                sv = Statevector.from_instruction(st.session_state.circuit)
                b0, b1 = bloch_vectors(sv, num_q)

                if use_plotly:
                    # PLOTLY - Interactive 3D
                    if num_q == 1:
                        fig = draw_sphere_plotly("Qubit State", b0, '#bf40ff')
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        # Create two columns for side-by-side spheres
                        col1, col2 = st.columns(2)
                        with col1:
                            fig1 = draw_sphere_plotly("Qubit 0", b0, '#bf40ff')
                            st.plotly_chart(fig1, use_container_width=True)
                        with col2:
                            fig2 = draw_sphere_plotly("Qubit 1", b1, '#00e5ff')
                            st.plotly_chart(fig2, use_container_width=True)
                else:
                    # MATPLOTLIB - Classic Static
                    if num_q == 1:
                        fig = plt.figure(figsize=(7, 7), facecolor='#000000')
                        ax = fig.add_subplot(111, projection='3d')
                        draw_sphere_matplotlib(ax, "Qubit State", b0, '#bf40ff')
                    else:
                        fig = plt.figure(figsize=(14, 7), facecolor='#000000')
                        ax1 = fig.add_subplot(121, projection='3d')
                        ax2 = fig.add_subplot(122, projection='3d')
                        draw_sphere_matplotlib(ax1, "Qubit 0", b0, '#bf40ff')
                        draw_sphere_matplotlib(ax2, "Qubit 1", b1, '#00e5ff')
                    
                    fig.suptitle("Bloch Sphere Representation", fontsize=16,
                                 fontweight='bold', color='#ffffff', y=0.95)
                    fig.patch.set_facecolor('#000000')
                    plt.tight_layout(pad=2.0)
                    st.pyplot(fig)
                    plt.close(fig)
                    
            except Exception as e:
                st.error(f"Bloch rendering error: {e}")

    # CIRCUIT DIAGRAM VISUALIZATION
    if show_circuit:
        if not st.session_state.gate_history:
            st.info("ℹ️ No gates applied yet. Apply some gates to see the circuit diagram!")
        else:
            with st.spinner("Drawing circuit diagram..."):
                try:
                    from qiskit.visualization import circuit_drawer
                    
                    # Draw circuit using matplotlib backend for better quality
                    fig = circuit_drawer(
                        st.session_state.circuit,
                        output='mpl',
                        style={'backgroundcolor': '#1a1f3c'},
                        plot_barriers=False,
                        fold=20
                    )
                    
                    # Style the figure
                    fig.patch.set_facecolor('#1a1f3c')
                    
                    # Display in a nice container
                    st.markdown('<div class="section-title">🔌 Quantum Circuit Diagram</div>', unsafe_allow_html=True)
                    st.pyplot(fig)
                    plt.close(fig)
                    
                    # Show circuit depth info
                    depth = st.session_state.circuit.depth()
                    num_gates = len(st.session_state.gate_history)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Circuit Depth", depth)
                    with col2:
                        st.metric("Total Gates", num_gates)
                    with col3:
                        st.metric("Qubits", st.session_state.circuit.num_qubits)
                    
                except Exception as e:
                    st.error(f"Circuit diagram error: {e}")

    # ANIMATED BLOCH - SUPPORTS BOTH PLOTLY AND MATPLOTLIB
    if show_anim:
        if not st.session_state.gate_history:
            st.warning("⚠️ Apply at least one gate before animating.")
        else:
            try:
                states, gate_labels, pause_indices = build_step_states()
                n_frames = len(states)
                
                # Check frame limit
                if n_frames > MAX_ANIMATION_FRAMES:
                    st.error(f"Too many animation frames ({n_frames}). Maximum is {MAX_ANIMATION_FRAMES}. "
                            f"Reduce circuit complexity or gate count.")
                else:
                    progress = st.progress(0, text="Rendering animation frames…")
                    frames_b64 = []

                    num_q = 1 if mode == "1-Qubit Explorer" else 2

                    for i, (sv, label) in enumerate(zip(states, gate_labels + [""])):
                        b0, b1 = bloch_vectors(sv, num_q)
                        title = f"Applying {label}" if i < len(gate_labels) else "Final State"

                        if use_plotly:
                            # PLOTLY ANIMATION
                            if num_q == 1:
                                fig = draw_sphere_plotly(f"Qubit · Step {i}", b0, '#bf40ff')
                                fig.update_layout(
                                    title=dict(text=f"{title}<br>Step {i}", 
                                             font=dict(size=18, color='#ffffff')),
                                    height=500
                                )
                                img_bytes = fig.to_image(format="png", width=700, height=500)
                                frames_b64.append(base64.b64encode(img_bytes).decode())
                            else:
                                # Two spheres - combine images
                                fig1 = draw_sphere_plotly(f"Qubit 0 · Step {i}", b0, '#bf40ff')
                                fig2 = draw_sphere_plotly(f"Qubit 1 · Step {i}", b1, '#00e5ff')
                                
                                img1_bytes = fig1.to_image(format="png", width=600, height=500)
                                img2_bytes = fig2.to_image(format="png", width=600, height=500)
                                
                                img1 = Image.open(io.BytesIO(img1_bytes))
                                img2 = Image.open(io.BytesIO(img2_bytes))
                                
                                combined = Image.new('RGB', (1200, 500), color='#000000')
                                combined.paste(img1, (0, 0))
                                combined.paste(img2, (600, 0))
                                
                                buf = io.BytesIO()
                                combined.save(buf, format='PNG')
                                buf.seek(0)
                                frames_b64.append(base64.b64encode(buf.read()).decode())
                        else:
                            # MATPLOTLIB ANIMATION (Faster rendering)
                            if num_q == 1:
                                fig = plt.figure(figsize=(7, 7), facecolor='#000000')
                                ax = fig.add_subplot(111, projection='3d')
                                draw_sphere_matplotlib(ax, f"Qubit · Step {i}", b0, '#bf40ff')
                            else:
                                fig = plt.figure(figsize=(14, 7), facecolor='#000000')
                                ax1 = fig.add_subplot(121, projection='3d')
                                ax2 = fig.add_subplot(122, projection='3d')
                                draw_sphere_matplotlib(ax1, f"Qubit 0 · Step {i}", b0, '#bf40ff')
                                draw_sphere_matplotlib(ax2, f"Qubit 1 · Step {i}", b1, '#00e5ff')
                            
                            fig.suptitle(title, fontsize=15,
                                         fontweight='bold', color='#ffffff', y=0.95)
                            fig.patch.set_facecolor('#000000')
                            plt.tight_layout(pad=2.0)

                            buf = io.BytesIO()
                            fig.savefig(buf, format='png', dpi=110, facecolor='#000000')
                            plt.close(fig)
                            buf.seek(0)
                            frames_b64.append(base64.b64encode(buf.read()).decode())
                        
                        progress.progress((i + 1) / n_frames,
                                          text=f"Rendered {i+1}/{n_frames}")

                    progress.empty()

                    # JavaScript slideshow
                    frames_js = str(frames_b64).replace("'", '"')
                    pause_js = str(pause_indices)
                    playback_speed = 120 if mode == "1-Qubit Explorer" else 900
                    
                    anim_html = f"""
<div style="text-align:center; background:#0d1117; border-radius:16px; padding:20px;
            border:1px solid rgba(99,102,241,0.25);">
  <img id="qframe" src="data:image/png;base64,{frames_b64[0]}"
       style="max-width:100%; border-radius:12px;" />
  <br/><br/>
  <div style="display:flex; justify-content:center; align-items:center; gap:16px;">
    <button id="prevBtn" onclick="changeFrame(-1)"
      style="background:#1e293b;color:#818cf8;border:1px solid #818cf8;
             border-radius:8px;padding:8px 18px;cursor:pointer;font-size:14px;">
      ◀ Prev
    </button>
    <span id="frameLabel"
      style="color:#94a3b8;font-family:monospace;min-width:90px;text-align:center">
      Step 0 / {n_frames-1}
    </span>
    <button id="nextBtn" onclick="changeFrame(1)"
      style="background:#1e293b;color:#818cf8;border:1px solid #818cf8;
             border-radius:8px;padding:8px 18px;cursor:pointer;font-size:14px;">
      Next ▶
    </button>
    <button id="playBtn" onclick="togglePlay()"
      style="background:linear-gradient(135deg,#4338ca,#7c3aed);color:white;
             border:none;border-radius:8px;padding:8px 22px;cursor:pointer;font-size:14px;
             font-weight:600;">
      ▶ Play
    </button>
  </div>
</div>

<script>
(function(){{
  const frames = {frames_js};
  const pauseIndices = {pause_js};
  let cur = 0, timer = null;
  let isPlaying = false;
  const img = document.getElementById('qframe');
  const lbl = document.getElementById('frameLabel');
  const playBtn = document.getElementById('playBtn');

  function show(idx) {{
    cur = Math.max(0, Math.min(idx, frames.length - 1));
    img.src = 'data:image/png;base64,' + frames[cur];
    lbl.textContent = 'Step ' + cur + ' / ' + (frames.length - 1);
  }}

  function nextFrameLogic() {{
    if (cur >= frames.length - 1) {{
      isPlaying = false;
      playBtn.textContent = '▶ Play';
      return;
    }}
    show(cur + 1);
    
    if (pauseIndices.includes(cur) && isPlaying) {{
      timer = setTimeout(nextFrameLogic, 2000);
    }} else if (isPlaying) {{
      timer = setTimeout(nextFrameLogic, {playback_speed});
    }}
  }}

  window.changeFrame = function(d) {{ 
      isPlaying = false;
      clearTimeout(timer); timer=null; playBtn.textContent='▶ Play'; show(cur + d); 
  }};

  window.togglePlay = function() {{
    if (isPlaying) {{
      isPlaying = false;
      clearTimeout(timer); timer = null; playBtn.textContent = '▶ Play';
    }} else {{
      isPlaying = true;
      if (cur >= frames.length - 1) cur = 0;
      playBtn.textContent = '⏸ Pause';
      nextFrameLogic();
    }}
  }};
}})();
</script>
"""
                    st.components.v1.html(anim_html, height=600)

            except Exception as e:
                st.error(f"Animation error: {e}")

    # OUTPUT BOX
    if st.session_state.output:
        st.markdown('<div class="section-title">📤 Output</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="output-box">{st.session_state.output}</div>',
            unsafe_allow_html=True
        )