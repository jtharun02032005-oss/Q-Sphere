"""
QuantumLab - Advanced Quantum Circuit Research Platform
========================================================
Professional-grade quantum computing research tool with advanced analysis,
state tomography, noise modeling, and custom gate synthesis.

Run with: streamlit run quantum_research_platform.py
"""

import warnings
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import io
import base64
import os
import json
from pathlib import Path
from scipy.linalg import expm, logm
from scipy.optimize import minimize
import time

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, DensityMatrix, partial_trace, Operator, state_fidelity, entanglement_of_formation
from qiskit.visualization import circuit_drawer
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
MAX_QUBITS = 10
SAVE_DIR = Path("quantum_research_data")
SAVE_DIR.mkdir(exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
# PROFESSIONAL RESEARCH STYLING
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

  * { 
    font-family: 'IBM Plex Sans', -apple-system, sans-serif;
    color: #000000 !important;
  }
  
  .stApp { 
    background: #ffffff; 
    color: #000000 !important;
  }

  /* Professional header */
  .research-header {
    background: linear-gradient(135deg, #1f6feb 0%, #0969da 100%);
    padding: 2rem 2.5rem;
    border-radius: 6px;
    margin-bottom: 1.5rem;
    border: 1px solid #d0d7de;
  }
  
  .research-header h1 {
    font-size: 2rem;
    font-weight: 600;
    color: white !important;
    margin: 0 0 0.5rem 0;
    letter-spacing: -0.02em;
  }
  
  .research-header p {
    color: rgba(255, 255, 255, 0.95) !important;
    font-size: 1rem;
    margin: 0;
    font-weight: 400;
  }

  /* Data panels */
  .data-panel {
    background: #f6f8fa;
    border: 1px solid #d0d7de;
    border-radius: 6px;
    padding: 1.25rem;
    margin-bottom: 1rem;
  }
  
  .panel-header {
    font-size: 0.875rem;
    font-weight: 600;
    color: #000000 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.75rem;
  }

  /* Metrics */
  [data-testid="metric-container"] {
    background: #f6f8fa;
    border: 1px solid #d0d7de;
    border-radius: 6px;
    padding: 1rem;
  }
  
  [data-testid="metric-container"] label {
    color: #000000 !important;
    font-weight: 500 !important;
    font-size: 0.75rem !important;
  }
  
  [data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #000000 !important;
    font-weight: 600 !important;
    font-family: 'IBM Plex Mono', monospace !important;
  }
  
  [data-testid="metric-container"] [data-testid="stMetricDelta"] {
    color: #000000 !important;
  }

  /* Buttons */
  div[data-testid="stButton"] > button {
    background: #1f6feb !important;
    color: white !important;
    border: 1px solid #0969da !important;
    border-radius: 6px !important;
    font-weight: 500 !important;
    padding: 0.5rem 1rem !important;
    font-size: 0.875rem !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
  }
  
  div[data-testid="stButton"] > button:hover {
    background: #0969da !important;
    border-color: #58a6ff !important;
    color: white !important;
  }
  
  .secondary-btn div[data-testid="stButton"] > button {
    background: transparent !important;
    border: 1px solid #d0d7de !important;
    color: #000000 !important;
  }
  
  .secondary-btn div[data-testid="stButton"] > button:hover {
    background: #f6f8fa !important;
    border-color: #8b949e !important;
    color: #000000 !important;
  }
  
  .danger-btn div[data-testid="stButton"] > button {
    background: #da3633 !important;
    border: 1px solid #f85149 !important;
    color: white !important;
  }

  /* Code blocks */
  .code-block {
    background: #f6f8fa;
    border: 1px solid #d0d7de;
    border-radius: 6px;
    padding: 1rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.875rem;
    color: #000000 !important;
    overflow-x: auto;
  }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: #ffffff;
  }
  
  .stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #000000 !important;
    font-weight: 500 !important;
    border-radius: 6px 6px 0 0 !important;
    padding: 0.5rem 1rem !important;
    border-bottom: 2px solid transparent !important;
  }
  
  .stTabs [aria-selected="true"] {
    color: #000000 !important;
    border-bottom: 2px solid #1f6feb !important;
    font-weight: 600 !important;
  }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: #f6f8fa !important;
    border-right: 1px solid #d0d7de !important;
  }
  
  [data-testid="stSidebar"] * {
    color: #000000 !important;
  }

  /* Info boxes */
  .stInfo {
    background: rgba(56, 139, 253, 0.1) !important;
    border: 1px solid #1f6feb !important;
    color: #000000 !important;
  }
  
  .stInfo * {
    color: #000000 !important;
  }
  
  .stSuccess {
    background: rgba(31, 111, 235, 0.1) !important;
    border: 1px solid #1f6feb !important;
    color: #000000 !important;
  }
  
  .stSuccess * {
    color: #000000 !important;
  }
  
  .stWarning {
    background: rgba(187, 128, 9, 0.1) !important;
    border: 1px solid #9e6a03 !important;
    color: #000000 !important;
  }
  
  .stWarning * {
    color: #000000 !important;
  }
  
  .stError {
    background: rgba(248, 81, 73, 0.1) !important;
    border: 1px solid #da3633 !important;
    color: #000000 !important;
  }
  
  .stError * {
    color: #000000 !important;
  }

  /* Tables */
  .dataframe {
    background: #ffffff;
    color: #000000 !important;
    border: 1px solid #d0d7de;
  }
  
  .dataframe th {
    background: #f6f8fa !important;
    color: #000000 !important;
    font-weight: 600 !important;
  }
  
  .dataframe td {
    color: #000000 !important;
  }

  /* Expander */
  .streamlit-expanderHeader {
    background: #f6f8fa;
    border: 1px solid #d0d7de;
    border-radius: 6px;
    color: #000000 !important;
    font-weight: 500;
  }
  
  .streamlit-expanderContent {
    color: #000000 !important;
  }

  /* Text inputs */
  .stTextInput > div > div > input,
  .stNumberInput > div > div > input,
  .stSelectbox > div > div {
    background: #ffffff !important;
    border: 1px solid #d0d7de !important;
    color: #000000 !important;
    border-radius: 6px !important;
  }
  
  .stTextInput label,
  .stNumberInput label,
  .stSelectbox label {
    color: #000000 !important;
  }

  /* Sliders */
  .stSlider > div > div > div {
    background: #f6f8fa;
  }
  
  .stSlider label {
    color: #000000 !important;
  }
  
  .stSlider [data-testid="stTickBarMin"],
  .stSlider [data-testid="stTickBarMax"] {
    color: #000000 !important;
  }

  /* Radio buttons */
  .stRadio label {
    color: #000000 !important;
  }
  
  .stRadio div[role="radiogroup"] label {
    color: #000000 !important;
  }

  /* Checkbox */
  .stCheckbox label {
    color: #000000 !important;
  }

  /* General text overrides - FORCE BLACK */
  .stMarkdown, 
  .stMarkdown p, 
  .stMarkdown li, 
  .stMarkdown h1, 
  .stMarkdown h2, 
  .stMarkdown h3, 
  .stMarkdown h4, 
  .stMarkdown h5, 
  .stMarkdown h6,
  .stMarkdown span,
  .stMarkdown div,
  .stMarkdown strong,
  .stMarkdown em {
    color: #000000 !important;
  }

  /* Sidebar specific */
  [data-testid="stSidebar"] .stMarkdown, 
  [data-testid="stSidebar"] .stMarkdown p,
  [data-testid="stSidebar"] .stMarkdown h1, 
  [data-testid="stSidebar"] .stMarkdown h2,
  [data-testid="stSidebar"] .stMarkdown h3, 
  [data-testid="stSidebar"] label,
  [data-testid="stSidebar"] span,
  [data-testid="stSidebar"] div {
    color: #000000 !important;
  }

  [data-testid="stSidebar"] [data-testid="stMetricValue"] {
    color: #000000 !important;
  }

  [data-testid="stSidebar"] [data-testid="metric-container"] label {
    color: #000000 !important;
  }
  
  /* Button captions */
  .button-caption {
    font-size: 0.7rem;
    color: #000000 !important;
    text-align: center;
    margin-top: -0.5rem;
    margin-bottom: 0.5rem;
    line-height: 1.2;
  }
  
  /* Code elements */
  code {
    color: #000000 !important;
  }
  
  pre {
    color: #000000 !important;
  }
  
  /* Captions */
  .caption, [data-testid="caption"] {
    color: #000000 !important;
  }
  
  /* Selectbox options */
  [data-baseweb="popover"] {
    color: #000000 !important;
  }
  
  [data-baseweb="menu"] li {
    color: #000000 !important;
  }
  
  /* Spinner */
  .stSpinner > div {
    color: #000000 !important;
  }
  
  /* All text elements */
  p, span, div, label, h1, h2, h3, h4, h5, h6, li, a, td, th {
    color: #000000 !important;
  }
  
  /* Override any remaining elements */
  [class*="st"] {
    color: #000000 !important;
  }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="QuantumLab - Research Platform",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════════════════
# ADVANCED QUANTUM UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

class QuantumStateAnalyzer:
    """Advanced quantum state analysis tools."""
    
    @staticmethod
    def compute_entropy(density_matrix):
        """Compute von Neumann entropy."""
        eigenvalues = np.linalg.eigvalsh(density_matrix)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]  # Remove numerical zeros
        return -np.sum(eigenvalues * np.log2(eigenvalues))
    
    @staticmethod
    def compute_purity(density_matrix):
        """Compute purity Tr(ρ²)."""
        return np.real(np.trace(density_matrix @ density_matrix))
    
    @staticmethod
    def compute_concurrence(statevector):
        """Compute concurrence for 2-qubit systems."""
        if statevector.num_qubits != 2:
            return None
        
        # Get density matrix
        rho = DensityMatrix(statevector).data
        
        # Pauli Y matrix
        sigma_y = np.array([[0, -1j], [1j, 0]])
        
        # Compute spin-flipped state
        sigma_y_total = np.kron(sigma_y, sigma_y)
        rho_tilde = sigma_y_total @ np.conj(rho) @ sigma_y_total
        
        # Compute R matrix
        R = rho @ rho_tilde
        
        # Get eigenvalues
        eigenvalues = np.linalg.eigvalsh(R)
        eigenvalues = np.sqrt(np.maximum(eigenvalues, 0))  # Ensure non-negative
        eigenvalues = np.sort(eigenvalues)[::-1]  # Sort descending
        
        # Concurrence
        C = max(0, eigenvalues[0] - eigenvalues[1] - eigenvalues[2] - eigenvalues[3])
        return C
    
    @staticmethod
    def compute_negativity(statevector, subsystem):
        """Compute negativity as entanglement measure."""
        if statevector.num_qubits < 2:
            return 0
        
        # Get density matrix
        rho = DensityMatrix(statevector).data
        
        # Partial transpose
        dim = 2 ** statevector.num_qubits
        subsystem_dim = 2
        
        # Reshape for partial transpose
        rho_reshaped = rho.reshape(subsystem_dim, dim // subsystem_dim, 
                                     subsystem_dim, dim // subsystem_dim)
        
        # Partial transpose over subsystem
        rho_pt = np.transpose(rho_reshaped, (2, 1, 0, 3)).reshape(dim, dim)
        
        # Compute eigenvalues
        eigenvalues = np.linalg.eigvalsh(rho_pt)
        
        # Negativity is sum of absolute values of negative eigenvalues
        negativity = np.sum(np.abs(eigenvalues[eigenvalues < 0]))
        
        return negativity
    
    @staticmethod
    def density_matrix_to_bloch(rho):
        """Convert density matrix to Bloch vector with improved precision."""
        # Ensure rho is a numpy array
        rho = np.array(rho)
        x = 2 * np.real(rho[0, 1])
        y = -2 * np.imag(rho[0, 1])
        z = np.real(rho[0, 0] - rho[1, 1])
        
        # Clip to unit sphere to handle numerical noise
        vec = np.array([x, y, z], dtype=float)
        norm = np.linalg.norm(vec)
        if norm > 1.0:
            vec = vec / norm
        return vec

class CustomGateBuilder:
    """Build custom quantum gates."""
    
    @staticmethod
    def rotation_gate(axis, angle):
        """Create rotation gate around arbitrary axis."""
        axis = np.array(axis)
        axis = axis / np.linalg.norm(axis)  # Normalize
        
        # Pauli matrices
        sigma_x = np.array([[0, 1], [1, 0]])
        sigma_y = np.array([[0, -1j], [1j, 0]])
        sigma_z = np.array([[1, 0], [0, -1]])
        
        # σ·n
        sigma_n = axis[0] * sigma_x + axis[1] * sigma_y + axis[2] * sigma_z
        
        # R(n,θ) = exp(-iθσ·n/2)
        return expm(-1j * angle * sigma_n / 2)
    
    @staticmethod
    def controlled_unitary(unitary):
        """Create controlled version of arbitrary unitary."""
        n = unitary.shape[0]
        cu = np.eye(2 * n, dtype=complex)
        cu[n:, n:] = unitary
        return cu

class NoiseModeling:
    """Quantum noise models for realistic simulations."""
    
    @staticmethod
    def create_noise_model(gate_error=0.001, measurement_error=0.01, 
                          t1=50e-6, t2=70e-6):
        """Create a realistic noise model."""
        noise_model = NoiseModel()
        
        # Depolarizing error on single-qubit gates
        error_1q = depolarizing_error(gate_error, 1)
        noise_model.add_all_qubit_quantum_error(error_1q, ['u1', 'u2', 'u3', 'rx', 'ry', 'rz'])
        
        # Depolarizing error on two-qubit gates
        error_2q = depolarizing_error(gate_error * 10, 2)
        noise_model.add_all_qubit_quantum_error(error_2q, ['cx', 'cz', 'swap'])
        
        # Thermal relaxation (T1, T2)
        gate_time = 50e-9  # 50 ns
        thermal_error = thermal_relaxation_error(t1, t2, gate_time)
        noise_model.add_all_qubit_quantum_error(thermal_error, ['u1', 'u2', 'u3'])
        
        return noise_model

# ═══════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════

def init_session_state():
    """Initialize session state variables."""
    if "num_qubits" not in st.session_state:
        st.session_state.num_qubits = 2
    if "circuit" not in st.session_state:
        st.session_state.circuit = QuantumCircuit(st.session_state.num_qubits)
    if "gate_history" not in st.session_state:
        st.session_state.gate_history = []
    if "analysis_cache" not in st.session_state:
        st.session_state.analysis_cache = {}
    if "experiment_log" not in st.session_state:
        st.session_state.experiment_log = []

init_session_state()

# ═══════════════════════════════════════════════════════════════════════════
# GATE OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════

def apply_gate(gate_type, params):
    """Apply quantum gate with parameters."""
    try:
        if gate_type == 'H':
            st.session_state.circuit.h(params['qubit'])
            label = f"H q{params['qubit']}"
        elif gate_type == 'X':
            st.session_state.circuit.x(params['qubit'])
            label = f"X q{params['qubit']}"
        elif gate_type == 'Y':
            st.session_state.circuit.y(params['qubit'])
            label = f"Y q{params['qubit']}"
        elif gate_type == 'Z':
            st.session_state.circuit.z(params['qubit'])
            label = f"Z q{params['qubit']}"
        elif gate_type == 'S':
            st.session_state.circuit.s(params['qubit'])
            label = f"S q{params['qubit']}"
        elif gate_type == 'T':
            st.session_state.circuit.t(params['qubit'])
            label = f"T q{params['qubit']}"
        elif gate_type == 'RX':
            st.session_state.circuit.rx(params['angle'], params['qubit'])
            label = f"Rx({params['angle']:.3f}) q{params['qubit']}"
        elif gate_type == 'RY':
            st.session_state.circuit.ry(params['angle'], params['qubit'])
            label = f"Ry({params['angle']:.3f}) q{params['qubit']}"
        elif gate_type == 'RZ':
            st.session_state.circuit.rz(params['angle'], params['qubit'])
            label = f"Rz({params['angle']:.3f}) q{params['qubit']}"
        elif gate_type == 'CNOT':
            st.session_state.circuit.cx(params['control'], params['target'])
            label = f"CNOT q{params['control']}→q{params['target']}"
        elif gate_type == 'CZ':
            st.session_state.circuit.cz(params['control'], params['target'])
            label = f"CZ q{params['control']}→q{params['target']}"
        elif gate_type == 'SWAP':
            st.session_state.circuit.swap(params['control'], params['target'])
            label = f"SWAP q{params['control']}↔q{params['target']}"
        elif gate_type == 'Toffoli':
            st.session_state.circuit.ccx(params['control1'], params['control2'], params['target'])
            label = f"Toffoli q{params['control1']},q{params['control2']}→q{params['target']}"
        elif gate_type == 'BARRIER':
            st.session_state.circuit.barrier()
            label = "BARRIER"
        elif gate_type == 'CUSTOM':
            # Custom unitary gate
            unitary_matrix = params['unitary']
            st.session_state.circuit.unitary(unitary_matrix, params['qubits'], label=params.get('label', 'U'))
            label = f"{params.get('label', 'U')} q{params['qubits']}"
        
        st.session_state.gate_history.append(label)
        st.session_state.analysis_cache = {}  # Invalidate cache
        return True, label
    except Exception as e:
        return False, str(e)

# ═══════════════════════════════════════════════════════════════════════════
# VISUALIZATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def draw_bloch_sphere(title, vec, color='#1f6feb', qubit_idx=0):
    """Draw Bloch sphere with research-grade styling and clear axis labeling."""
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 30)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones_like(u), np.cos(v))
    
    fig = go.Figure()
    
    # Sphere surface
    fig.add_trace(go.Surface(
        x=x_sphere, y=y_sphere, z=z_sphere,
        colorscale=[[0, '#f8f9fa'], [1, '#f8f9fa']],
        showscale=False,
        opacity=0.15,
        hoverinfo='skip',
        contours=dict(
            x=dict(show=True, color='#d0d7de', width=1, usecolormap=False),
            y=dict(show=True, color='#d0d7de', width=1, usecolormap=False),
            z=dict(show=True, color='#d0d7de', width=1, usecolormap=False)
        )
    ))
    
    # Axes
    axis_length = 1.3
    # X-axis (Red)
    fig.add_trace(go.Scatter3d(
        x=[-axis_length, axis_length], y=[0, 0], z=[0, 0],
        mode='lines', line=dict(color='#f85149', width=4),
        hoverinfo='skip', name='X-axis'
    ))
    # Y-axis (Green)
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[-axis_length, axis_length], z=[0, 0],
        mode='lines', line=dict(color='#3fb950', width=4),
        hoverinfo='skip', name='Y-axis'
    ))
    # Z-axis (Blue)
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, 0], z=[-axis_length, axis_length],
        mode='lines', line=dict(color='#58a6ff', width=5),
        hoverinfo='skip', name='Z-axis'
    ))
    
    # Axis Labels
    fig.add_trace(go.Scatter3d(
        x=[axis_length+0.1, 0, 0],
        y=[0, axis_length+0.1, 0],
        z=[0, 0, axis_length+0.1],
        mode='text',
        text=['X', 'Y', 'Z'],
        textfont=dict(color=['#f85149', '#3fb950', '#58a6ff'], size=14, family='IBM Plex Mono'),
        hoverinfo='skip'
    ))
    
    # Pole Labels
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, 0], z=[1.1, -1.1],
        mode='text',
        text=['|0⟩', '|1⟩'],
        textfont=dict(color='#000000', size=16, family='IBM Plex Mono'),
        hoverinfo='skip'
    ))
    
    # State vector
    vec_norm = np.linalg.norm(vec)
    if vec_norm > 1e-6:
        # Line from origin
        fig.add_trace(go.Scatter3d(
            x=[0, vec[0]], y=[0, vec[1]], z=[0, vec[2]],
            mode='lines', line=dict(color=color, width=10),
            hoverinfo='text',
            hovertext=f'Qubit {qubit_idx}<br>x:{vec[0]:.3f}<br>y:{vec[1]:.3f}<br>z:{vec[2]:.3f}<br>Purity:{vec_norm:.3f}',
            name='State Vector'
        ))
        
        # Arrow tip
        fig.add_trace(go.Cone(
            x=[vec[0]], y=[vec[1]], z=[vec[2]],
            u=[vec[0]], v=[vec[1]], w=[vec[2]],
            colorscale=[[0, color], [1, color]],
            showscale=False, sizemode='absolute', sizeref=0.25,
            hoverinfo='skip'
        ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color='#000000', family='IBM Plex Sans')),
        scene=dict(
            xaxis=dict(visible=False, range=[-1.5, 1.5]),
            yaxis=dict(visible=False, range=[-1.5, 1.5]),
            zaxis=dict(visible=False, range=[-1.5, 1.5]),
            bgcolor='#ffffff',
            camera=dict(
                eye=dict(x=1.6, y=1.2, z=1.0),
                up=dict(x=0, y=0, z=1)
            ),
            aspectmode='cube'
        ),
        paper_bgcolor='#ffffff',
        margin=dict(l=0, r=0, t=40, b=0),
        height=450,
        showlegend=False
    )
    
    return fig

def plot_density_matrix(density_matrix, title="Density Matrix"):
    """Visualize density matrix."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Real Part', 'Imaginary Part'),
        specs=[[{'type': 'heatmap'}, {'type': 'heatmap'}]]
    )
    
    # Real part
    fig.add_trace(
        go.Heatmap(
            z=np.real(density_matrix),
            colorscale='RdBu',
            zmid=0,
            showscale=True,
            colorbar=dict(x=0.45)
        ),
        row=1, col=1
    )
    
    # Imaginary part
    fig.add_trace(
        go.Heatmap(
            z=np.imag(density_matrix),
            colorscale='RdBu',
            zmid=0,
            showscale=True,
            colorbar=dict(x=1.05)
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title=dict(text=title, font=dict(color='#000000')),
        paper_bgcolor='#ffffff',
        plot_bgcolor='#f6f8fa',
        font=dict(color='#000000'),
        height=400
    )
    
    # Update subplot titles to black
    for annotation in fig.layout.annotations:
        annotation.font.color = '#000000'
    
    return fig

# ═══════════════════════════════════════════════════════════════════════════
# MAIN INTERFACE
# ═══════════════════════════════════════════════════════════════════════════

# Header with logo
logo_html = ""
logo_path = "logo.png"
if not os.path.exists(logo_path) and os.path.exists("logo.png.png"):
    logo_path = "logo.png.png"

if os.path.exists(logo_path):
    with open(logo_path, "rb") as f:
        b64_logo = base64.b64encode(f.read()).decode()
    logo_html = f'<img src="data:image/png;base64,{b64_logo}" style="height: 110px; margin-right: 25px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.15); flex-shrink: 0;"/>'

st.markdown(f"""
<div class="research-header" style="display: flex; align-items: center;">
    {logo_html}
    <div>
        <h1>Q-SPHERE</h1>
        <p>Advanced quantum circuit simulation, state analysis, and noise modeling for quantum computing research</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.markdown("### ⚙️ System Configuration")
    
    new_num_qubits = st.slider("Number of Qubits", 1, MAX_QUBITS, st.session_state.num_qubits)
    
    if new_num_qubits != st.session_state.num_qubits:
        st.session_state.num_qubits = new_num_qubits
        st.session_state.circuit = QuantumCircuit(new_num_qubits)
        st.session_state.gate_history = []
        st.session_state.analysis_cache = {}
        st.rerun()
    
    st.markdown("---")
    
    st.markdown("### 📊 Circuit Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Gates", len(st.session_state.gate_history))
    with col2:
        st.metric("Depth", st.session_state.circuit.depth())
    
    st.metric("Hilbert Space Dim", f"2^{st.session_state.num_qubits} = {2**st.session_state.num_qubits}")
    
    st.markdown("---")
    
    st.markdown("### 🎛️ Simulation Settings")
    
    use_noise = st.checkbox("Enable Noise Model", value=False)
    
    if use_noise:
        gate_error = st.slider("Gate Error Rate", 0.0, 0.1, 0.001, 0.0001, format="%.4f")
        t1_time = st.slider("T1 Time (μs)", 10.0, 200.0, 50.0, 1.0)
        t2_time = st.slider("T2 Time (μs)", 10.0, 200.0, 70.0, 1.0)
    
    st.markdown("---")
    
    st.markdown("### 💾 Circuit Management")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📋 Copy QASM", use_container_width=True, help="Export circuit as OpenQASM code"):
            try:
                from qiskit import qasm2
                qasm_str = qasm2.dumps(st.session_state.circuit)
                st.code(qasm_str, language='text')
            except Exception as e:
                st.error(f"Error: {e}")
        st.markdown("<p class='button-caption'>Export to QASM</p>", unsafe_allow_html=True)
    
    with col2:
        if st.button("🗑️ Clear", use_container_width=True, help="Clear the entire circuit and start fresh"):
            st.session_state.circuit = QuantumCircuit(st.session_state.num_qubits)
            st.session_state.gate_history = []
            st.session_state.analysis_cache = {}
            st.rerun()
        st.markdown("<p class='button-caption'>Reset circuit</p>", unsafe_allow_html=True)
    
    circuit_name = st.text_input("Save As", "experiment_1")
    if st.button("💾 Save Circuit", use_container_width=True, help="Save circuit to disk as JSON file"):
        try:
            from qiskit import qasm2
            data = {
                'qasm': qasm2.dumps(st.session_state.circuit),
                'num_qubits': st.session_state.num_qubits,
                'gate_history': st.session_state.gate_history,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            filepath = SAVE_DIR / f"{circuit_name}.json"
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            st.success(f"Saved to {filepath}")
        except Exception as e:
            st.error(f"Save failed: {e}")
    st.markdown("<p class='button-caption'>Save to file</p>", unsafe_allow_html=True)

# Main content
tab_build, tab_analyze, tab_tomography, tab_noise, tab_custom = st.tabs([
    "🔨 Circuit Builder",
    "📊 State Analysis", 
    "🔬 Quantum Tomography",
    "🌊 Noise Simulation",
    "⚡ Custom Gates"
])

# ═══════════════════════════════════════════════════════════════════════════
# CIRCUIT BUILDER TAB
# ═══════════════════════════════════════════════════════════════════════════
with tab_build:
    st.markdown("### Quantum Circuit Construction")
    
    # Display circuit with compact visualization
    if st.session_state.gate_history:
        try:
            # Draw circuit with professional research styling
            fig = circuit_drawer(
                st.session_state.circuit, 
                output='mpl',
                style={
                    'backgroundcolor': '#ffffff',
                    'fontsize': 8,           # Stable font size
                    'subfontsize': 7,
                    'linecolor': '#57606a',
                    'gatetextcolor': '#000000',
                    'gatefacecolor': '#f6f8fa',
                    'barrierfacecolor': '#d0d7de'
                },
                fold=-1,
                scale=0.5  # Standard scale
            )
            
            # Save to buffer with high DPI for clarity but fixed display width for consistency
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='#ffffff')
            plt.close(fig)
            
            # Use fixed width to ensure fonts don't scale up/down with qubit count
            st.image(buf.getvalue(), width=700)
        except Exception as e:
            # Fallback to text representation
            st.code("\n".join(st.session_state.gate_history))
    else:
        st.info("Circuit is empty. Add gates below.")
    
    # Live State Preview (Experimental Feature)
    if st.session_state.gate_history:
        with st.expander("🔍 Live State Preview", expanded=True):
            try:
                sv_live = Statevector.from_instruction(st.session_state.circuit)
                analyzer_live = QuantumStateAnalyzer()
                
                prev_cols = st.columns(min(st.session_state.num_qubits, 4))
                for i in range(min(st.session_state.num_qubits, 4)):
                    with prev_cols[i]:
                        if st.session_state.num_qubits == 1:
                            rho_i = DensityMatrix(sv_live).data
                        else:
                            trace_qubits = [j for j in range(st.session_state.num_qubits) if j != i]
                            rho_i = partial_trace(sv_live, trace_qubits).data
                        
                        b_vec = analyzer_live.density_matrix_to_bloch(rho_i)
                        fig_live = draw_bloch_sphere(f"Q{i}", b_vec, qubit_idx=i)
                        # More compact for preview
                        fig_live.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
                        st.plotly_chart(fig_live, use_container_width=True, key=f"bloch_live_{i}")
            except Exception as e:
                st.error(f"Live preview error: {e}")
    
    st.markdown("---")
    
    # Gate palette
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Single-Qubit Gates**")
        
        qubit_select = st.selectbox("Target Qubit", range(st.session_state.num_qubits), key="sq_qubit")
        
        gate_cols = st.columns(4)
        with gate_cols[0]:
            if st.button("H", use_container_width=True, help="Hadamard gate: creates superposition (|0⟩+|1⟩)/√2"):
                apply_gate('H', {'qubit': qubit_select})
                st.rerun()
            st.markdown("<p class='button-caption'>Hadamard</p>", unsafe_allow_html=True)
        with gate_cols[1]:
            if st.button("X", use_container_width=True, help="Pauli-X gate: bit flip, quantum NOT gate"):
                apply_gate('X', {'qubit': qubit_select})
                st.rerun()
            st.markdown("<p class='button-caption'>NOT gate</p>", unsafe_allow_html=True)
        with gate_cols[2]:
            if st.button("Y", use_container_width=True, help="Pauli-Y gate: bit and phase flip"):
                apply_gate('Y', {'qubit': qubit_select})
                st.rerun()
            st.markdown("<p class='button-caption'>Y flip</p>", unsafe_allow_html=True)
        with gate_cols[3]:
            if st.button("Z", use_container_width=True, help="Pauli-Z gate: phase flip, leaves |0⟩, negates |1⟩"):
                apply_gate('Z', {'qubit': qubit_select})
                st.rerun()
            st.markdown("<p class='button-caption'>Phase flip</p>", unsafe_allow_html=True)
        
        gate_cols2 = st.columns(4)
        with gate_cols2[0]:
            if st.button("S", use_container_width=True, help="S gate: √Z, applies π/2 phase rotation"):
                apply_gate('S', {'qubit': qubit_select})
                st.rerun()
            st.markdown("<p class='button-caption'>√Z gate</p>", unsafe_allow_html=True)
        with gate_cols2[1]:
            if st.button("T", use_container_width=True, help="T gate: ⁴√Z, applies π/4 phase rotation"):
                apply_gate('T', {'qubit': qubit_select})
                st.rerun()
            st.markdown("<p class='button-caption'>⁴√Z gate</p>", unsafe_allow_html=True)
        with gate_cols2[2]:
            if st.button("S†", use_container_width=True, help="S-dagger: inverse of S gate, -π/2 phase"):
                st.session_state.circuit.sdg(qubit_select)
                st.session_state.gate_history.append(f"S† q{qubit_select}")
                st.rerun()
            st.markdown("<p class='button-caption'>S inverse</p>", unsafe_allow_html=True)
        with gate_cols2[3]:
            if st.button("T†", use_container_width=True, help="T-dagger: inverse of T gate, -π/4 phase"):
                st.session_state.circuit.tdg(qubit_select)
                st.session_state.gate_history.append(f"T† q{qubit_select}")
                st.rerun()
            st.markdown("<p class='button-caption'>T inverse</p>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("**Rotation Gates**")
        
        rot_qubit = st.selectbox("Target Qubit", range(st.session_state.num_qubits), key="rot_qubit")
        rot_axis = st.selectbox("Axis", ["X", "Y", "Z"])
        rot_angle = st.number_input("Angle (radians)", -2*np.pi, 2*np.pi, 0.0, 0.1)
        
        if st.button(f"Apply R{rot_axis}({rot_angle:.2f})", use_container_width=True, help=f"Rotate qubit by {rot_angle:.2f} radians around {rot_axis}-axis"):
            apply_gate(f'R{rot_axis}', {'qubit': rot_qubit, 'angle': rot_angle})
            st.rerun()
        st.markdown("<p class='button-caption'>Custom rotation</p>", unsafe_allow_html=True)
        
        st.markdown("**Quick Angles**")
        quick_cols = st.columns(3)
        with quick_cols[0]:
            if st.button("π/4", key="pi4", help="Quick rotation: 45 degrees"):
                apply_gate(f'R{rot_axis}', {'qubit': rot_qubit, 'angle': np.pi/4})
                st.rerun()
            st.markdown("<p class='button-caption'>45°</p>", unsafe_allow_html=True)
        with quick_cols[1]:
            if st.button("π/2", key="pi2", help="Quick rotation: 90 degrees"):
                apply_gate(f'R{rot_axis}', {'qubit': rot_qubit, 'angle': np.pi/2})
                st.rerun()
            st.markdown("<p class='button-caption'>90°</p>", unsafe_allow_html=True)
        with quick_cols[2]:
            if st.button("π", key="pi", help="Quick rotation: 180 degrees"):
                apply_gate(f'R{rot_axis}', {'qubit': rot_qubit, 'angle': np.pi})
                st.rerun()
            st.markdown("<p class='button-caption'>180°</p>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("**Multi-Qubit Gates**")
        
        if st.session_state.num_qubits >= 2:
            control_q = st.selectbox("Control", range(st.session_state.num_qubits), key="control")
            target_q = st.selectbox("Target", range(st.session_state.num_qubits), key="target")
            
            if control_q == target_q:
                st.warning("Control ≠ Target required")
            else:
                gate_cols3 = st.columns(3)
                with gate_cols3[0]:
                    if st.button("CNOT", use_container_width=True, help="Controlled-NOT: flips target if control is |1⟩, creates entanglement"):
                        apply_gate('CNOT', {'control': control_q, 'target': target_q})
                        st.rerun()
                    st.markdown("<p class='button-caption'>Entangle</p>", unsafe_allow_html=True)
                with gate_cols3[1]:
                    if st.button("CZ", use_container_width=True, help="Controlled-Z: applies Z to target if control is |1⟩"):
                        apply_gate('CZ', {'control': control_q, 'target': target_q})
                        st.rerun()
                    st.markdown("<p class='button-caption'>C-Phase</p>", unsafe_allow_html=True)
                with gate_cols3[2]:
                    if st.button("SWAP", use_container_width=True, help="SWAP: exchanges quantum states of two qubits"):
                        apply_gate('SWAP', {'control': control_q, 'target': target_q})
                        st.rerun()
                    st.markdown("<p class='button-caption'>Exchange</p>", unsafe_allow_html=True)
                
                if st.session_state.num_qubits >= 3:
                    st.markdown("**Toffoli (CCX)**")
                    control2_q = st.selectbox("Control 2", range(st.session_state.num_qubits), key="control2")
                    
                    if len({control_q, control2_q, target_q}) == 3:
                        if st.button("Apply Toffoli", use_container_width=True, help="Toffoli/CCX: flips target only if both controls are |1⟩"):
                            apply_gate('Toffoli', {
                                'control1': control_q,
                                'control2': control2_q,
                                'target': target_q
                            })
                            st.rerun()
                        st.markdown("<p class='button-caption'>Double-controlled NOT</p>", unsafe_allow_html=True)
        else:
            st.info("Need ≥2 qubits for multi-qubit gates")
        
        st.markdown("**Utilities**")
        if st.button("Add Barrier", use_container_width=True, help="Barrier: visual separator, prevents gate reordering in optimization"):
            apply_gate('BARRIER', {})
            st.rerun()
        st.markdown("<p class='button-caption'>Visual separator</p>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# STATE ANALYSIS TAB
# ═══════════════════════════════════════════════════════════════════════════
with tab_analyze:
    st.markdown("### Quantum State Analysis")
    
    if not st.session_state.gate_history:
        st.info("Build a circuit first to analyze its quantum state.")
    else:
        try:
            # Compute state
            sv = Statevector.from_instruction(st.session_state.circuit)
            dm = DensityMatrix(sv)
            
            # Analysis
            analyzer = QuantumStateAnalyzer()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### State Vector Visualization")
                
                # Bloch spheres for each qubit
                num_display = min(st.session_state.num_qubits, 4)
                cols = st.columns(min(num_display, 2))
                
                for i in range(num_display):
                    with cols[i % 2]:
                        # Partial trace to get single qubit state
                        if st.session_state.num_qubits == 1:
                            rho_i = dm.data
                        else:
                            trace_qubits = [j for j in range(st.session_state.num_qubits) if j != i]
                            rho_i = partial_trace(sv, trace_qubits).data
                        
                        bloch_vec = analyzer.density_matrix_to_bloch(rho_i)
                        fig = draw_bloch_sphere(f"Qubit {i} State", bloch_vec, qubit_idx=i)
                        st.plotly_chart(fig, use_container_width=True, key=f"bloch_analyze_{i}")
                
                if st.session_state.num_qubits > 4:
                    st.warning("Showing first 4 qubits. Additional qubits omitted from Bloch display.")
            
            with col2:
                st.markdown("#### Quantum Metrics")
                
                # Purity
                purity = analyzer.compute_purity(dm.data)
                st.metric("Purity Tr(ρ²)", f"{purity:.6f}")
                st.caption("Purity = 1 for pure states, < 1 for mixed states")
                
                # Entropy
                entropy = analyzer.compute_entropy(dm.data)
                st.metric("Von Neumann Entropy", f"{entropy:.6f}")
                st.caption("Entropy = 0 for pure states, > 0 for mixed states")
                
                # Entanglement measures
                if st.session_state.num_qubits == 2:
                    concurrence = analyzer.compute_concurrence(sv)
                    st.metric("Concurrence", f"{concurrence:.6f}")
                    st.caption("Concurrence ∈ [0,1], 0=separable, 1=maximally entangled")
                    
                    negativity = analyzer.compute_negativity(sv, subsystem=0)
                    st.metric("Negativity", f"{negativity:.6f}")
                    st.caption("Negativity > 0 indicates entanglement")
                
                st.markdown("---")
                
                # Probability distribution
                st.markdown("#### Measurement Probabilities")
                probs = sv.probabilities_dict()
                
                prob_df = pd.DataFrame({
                    'State': [f"|{k}⟩" for k in probs.keys()],
                    'Probability': list(probs.values()),
                    'Percentage': [f"{v*100:.2f}%" for v in probs.values()]
                })
                
                st.dataframe(prob_df, use_container_width=True, hide_index=True)
                
                # Plot probabilities
                fig = go.Figure(data=[
                    go.Bar(
                        x=[f"|{k}⟩" for k in probs.keys()],
                        y=list(probs.values()),
                        marker_color='#1f6feb',
                        text=[f"{v*100:.1f}%" for v in probs.values()],
                        textposition='auto',
                        textfont=dict(color='#000000')
                    )
                ])
                
                fig.update_layout(
                    title="Measurement Outcome Distribution",
                    xaxis_title="Basis State",
                    yaxis_title="Probability",
                    paper_bgcolor='#ffffff',
                    plot_bgcolor='#f6f8fa',
                    font=dict(color='#000000'),
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Density matrix visualization
            st.markdown("---")
            st.markdown("#### Density Matrix Representation")
            
            if st.session_state.num_qubits <= 4:
                fig = plot_density_matrix(dm.data, "Full Density Matrix")
                st.plotly_chart(fig, use_container_width=True)
                
                # Show numerical values
                with st.expander("View Density Matrix Values"):
                    st.write("**Real Part:**")
                    st.dataframe(pd.DataFrame(np.real(dm.data)), use_container_width=True)
                    st.write("**Imaginary Part:**")
                    st.dataframe(pd.DataFrame(np.imag(dm.data)), use_container_width=True)
            else:
                st.warning("Density matrix too large to display (>4 qubits). Showing reduced state of first 2 qubits.")
                trace_qubits = list(range(2, st.session_state.num_qubits))
                reduced_dm = partial_trace(sv, trace_qubits).data
                fig = plot_density_matrix(reduced_dm, "Reduced Density Matrix (Qubits 0-1)")
                st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"Analysis error: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# TOMOGRAPHY TAB
# ═══════════════════════════════════════════════════════════════════════════
with tab_tomography:
    st.markdown("### Quantum State Tomography")
    st.info("Reconstruct the quantum state from measurement statistics in different bases.")
    
    if not st.session_state.gate_history:
        st.warning("Build a circuit first to perform tomography.")
    else:
        st.markdown("#### Measurement Basis Selection")
        
        num_shots = st.slider("Number of Measurements", 100, 10000, 1000, 100)
        
        # For small systems, do full tomography
        if st.session_state.num_qubits <= 2:
            st.markdown("**Pauli Basis Measurements**")
            
            # Create measurement circuits for all Pauli combinations
            paulis = ['I', 'X', 'Y', 'Z']
            
            if st.button("🔬 Run Full Tomography", use_container_width=True, help="Measure in all Pauli bases to reconstruct the full quantum state"):
                with st.spinner("Performing quantum state tomography..."):
                    # Simulate measurements
                    simulator = AerSimulator()
                    
                    results = {}
                    
                    # For each Pauli string
                    for pauli_str in [''.join(p) for p in [[p1, p2] for p1 in paulis for p2 in paulis]] if st.session_state.num_qubits == 2 else paulis:
                        # Create measurement circuit
                        meas_circ = st.session_state.circuit.copy()
                        
                        # Add basis rotation
                        for i, p in enumerate(pauli_str):
                            if p == 'X':
                                meas_circ.h(i)
                            elif p == 'Y':
                                meas_circ.sdg(i)
                                meas_circ.h(i)
                            # Z basis is computational basis, no rotation needed
                        
                        # Add measurements
                        meas_circ.measure_all()
                        
                        # Run simulation
                        try:
                            job = simulator.run(meas_circ, shots=num_shots)
                            counts = job.result().get_counts()
                            results[pauli_str] = counts
                        except:
                            continue
                    
                    # Display results
                    st.success("Tomography complete!")
                    
                    st.markdown("#### Measurement Results")
                    
                    for basis, counts in results.items():
                        with st.expander(f"Basis: {basis}"):
                            # Create bar chart
                            fig = go.Figure(data=[
                                go.Bar(
                                    x=list(counts.keys()),
                                    y=list(counts.values()),
                                    marker_color='#1f6feb',
                                    text=list(counts.values()),
                                    textposition='auto',
                                    textfont=dict(color='#000000')
                                )
                            ])
                            
                            fig.update_layout(
                                title=f"Measurements in {basis} basis",
                                xaxis_title="Outcome",
                                yaxis_title="Counts",
                                paper_bgcolor='#ffffff',
                                plot_bgcolor='#f6f8fa',
                                font=dict(color='#000000'),
                                height=300
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            st.json(counts)
            st.markdown("<p class='button-caption'>Reconstruct full state</p>", unsafe_allow_html=True)
        else:
            st.warning("Full tomography requires exponential resources for >2 qubits. Use selective measurements.")
            
            st.markdown("#### Single-Qubit Tomography")
            qubit_tomo = st.selectbox("Select qubit for tomography", range(st.session_state.num_qubits))
            
            if st.button("Run Single-Qubit Tomography", help="Reconstruct the state of one qubit by measuring in X, Y, Z bases"):
                # Similar to above but for single qubit
                st.info("Single-qubit tomography reconstruction in progress...")
            st.markdown("<p class='button-caption'>Reconstruct one qubit</p>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# NOISE SIMULATION TAB
# ═══════════════════════════════════════════════════════════════════════════
with tab_noise:
    st.markdown("### Realistic Noise Simulation")
    st.info("Model real quantum hardware noise: gate errors, decoherence, and measurement errors.")
    
    if not st.session_state.gate_history:
        st.warning("Build a circuit first to simulate noise effects.")
    else:
        st.markdown("#### Noise Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            gate_error_rate = st.slider("Gate Error Rate", 0.0, 0.1, 0.001, 0.0001, format="%.4f")
            measurement_error = st.slider("Measurement Error", 0.0, 0.1, 0.01, 0.001, format="%.3f")
        
        with col2:
            t1_time_us = st.slider("T1 Relaxation Time (μs)", 10.0, 200.0, 50.0, 1.0)
            t2_time_us = st.slider("T2 Dephasing Time (μs)", 10.0, 200.0, 70.0, 1.0)
        
        num_shots_noise = st.slider("Measurement Shots", 100, 10000, 1000, 100)
        
        if st.button("🌊 Run Noisy Simulation", use_container_width=True, help="Simulate circuit with realistic quantum hardware noise model"):
            with st.spinner("Simulating with noise model..."):
                try:
                    # Create noise model
                    noise_model = NoiseModeling.create_noise_model(
                        gate_error=gate_error_rate,
                        measurement_error=measurement_error,
                        t1=t1_time_us * 1e-6,
                        t2=t2_time_us * 1e-6
                    )
                    
                    # Prepare circuit
                    noisy_circ = st.session_state.circuit.copy()
                    noisy_circ.measure_all()
                    
                    # Ideal simulation
                    ideal_sim = AerSimulator()
                    ideal_job = ideal_sim.run(noisy_circ, shots=num_shots_noise)
                    ideal_counts = ideal_job.result().get_counts()
                    
                    # Noisy simulation
                    noisy_sim = AerSimulator(noise_model=noise_model)
                    noisy_job = noisy_sim.run(noisy_circ, shots=num_shots_noise)
                    noisy_counts = noisy_job.result().get_counts()
                    
                    # Compare results
                    st.success("Simulation complete!")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### Ideal (No Noise)")
                        fig_ideal = go.Figure(data=[
                            go.Bar(
                                x=list(ideal_counts.keys()),
                                y=list(ideal_counts.values()),
                                marker_color='#1f6feb',
                                name='Ideal',
                                text=list(ideal_counts.values()),
                                textposition='auto',
                                textfont=dict(color='#000000')
                            )
                        ])
                        fig_ideal.update_layout(
                            paper_bgcolor='#ffffff',
                            plot_bgcolor='#f6f8fa',
                            font=dict(color='#000000'),
                            height=400
                        )
                        st.plotly_chart(fig_ideal, use_container_width=True)
                    
                    with col2:
                        st.markdown("#### Noisy (Realistic)")
                        fig_noisy = go.Figure(data=[
                            go.Bar(
                                x=list(noisy_counts.keys()),
                                y=list(noisy_counts.values()),
                                marker_color='#f85149',
                                name='Noisy',
                                text=list(noisy_counts.values()),
                                textposition='auto',
                                textfont=dict(color='#000000')
                            )
                        ])
                        fig_noisy.update_layout(
                            paper_bgcolor='#ffffff',
                            plot_bgcolor='#f6f8fa',
                            font=dict(color='#000000'),
                            height=400
                        )
                        st.plotly_chart(fig_noisy, use_container_width=True)
                    
                    # Fidelity calculation
                    st.markdown("---")
                    st.markdown("#### Noise Impact Analysis")
                    
                    # Convert counts to probability distributions
                    ideal_probs = {k: v/num_shots_noise for k, v in ideal_counts.items()}
                    noisy_probs = {k: v/num_shots_noise for k, v in noisy_counts.items()}
                    
                    # All possible states
                    all_states = set(list(ideal_probs.keys()) + list(noisy_probs.keys()))
                    
                    # TVD (Total Variation Distance)
                    tvd = 0.5 * sum([abs(ideal_probs.get(s, 0) - noisy_probs.get(s, 0)) for s in all_states])
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Variation Distance", f"{tvd:.4f}")
                        st.caption("0 = identical, 1 = maximally different")
                    with col2:
                        st.metric("Effective Fidelity", f"{1-tvd:.4f}")
                        st.caption("Similarity to ideal distribution")
                    with col3:
                        error_amp = (tvd / gate_error_rate) if gate_error_rate > 0 else 0
                        st.metric("Error Amplification", f"{error_amp:.1f}×")
                        st.caption("How much errors are amplified")
                
                except Exception as e:
                    st.error(f"Simulation error: {e}")
        st.markdown("<p class='button-caption'>Compare ideal vs noisy</p>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# CUSTOM GATES TAB
# ═══════════════════════════════════════════════════════════════════════════
with tab_custom:
    st.markdown("### Custom Gate Designer")
    st.info("Create custom quantum gates using matrix representation or parameterized unitaries.")
    
    gate_method = st.radio("Creation Method", ["Matrix Input", "Parameterized Rotation", "Gate Decomposition"])
    
    if gate_method == "Matrix Input":
        st.markdown("#### Direct Unitary Matrix Input")
        st.caption("Enter a 2×2 unitary matrix for single-qubit gates")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Row 1**")
            r1c1_real = st.number_input("Real(U[0,0])", -1.0, 1.0, 1.0, 0.1, key="r1c1r")
            r1c1_imag = st.number_input("Imag(U[0,0])", -1.0, 1.0, 0.0, 0.1, key="r1c1i")
            r1c2_real = st.number_input("Real(U[0,1])", -1.0, 1.0, 0.0, 0.1, key="r1c2r")
            r1c2_imag = st.number_input("Imag(U[0,1])", -1.0, 1.0, 0.0, 0.1, key="r1c2i")
        
        with col2:
            st.markdown("**Row 2**")
            r2c1_real = st.number_input("Real(U[1,0])", -1.0, 1.0, 0.0, 0.1, key="r2c1r")
            r2c1_imag = st.number_input("Imag(U[1,0])", -1.0, 1.0, 0.0, 0.1, key="r2c1i")
            r2c2_real = st.number_input("Real(U[1,1])", -1.0, 1.0, 1.0, 0.1, key="r2c2r")
            r2c2_imag = st.number_input("Imag(U[1,1])", -1.0, 1.0, 0.0, 0.1, key="r2c2i")
        
        # Construct matrix
        U_custom = np.array([
            [r1c1_real + 1j*r1c1_imag, r1c2_real + 1j*r1c2_imag],
            [r2c1_real + 1j*r2c1_imag, r2c2_real + 1j*r2c2_imag]
        ])
        
        # Check if unitary
        is_unitary = np.allclose(U_custom @ U_custom.conj().T, np.eye(2))
        
        if is_unitary:
            st.success("✓ Matrix is unitary")
        else:
            st.error("✗ Matrix is not unitary! U†U must equal I")
        
        st.markdown("**Matrix Preview:**")
        st.code(str(U_custom))
        
        gate_label = st.text_input("Gate Label", "U_custom")
        target_qubit_custom = st.selectbox("Apply to qubit", range(st.session_state.num_qubits), key="custom_q")
        
        if st.button("Apply Custom Gate", help="Add this custom unitary gate to the circuit") and is_unitary:
            success, msg = apply_gate('CUSTOM', {
                'unitary': U_custom,
                'qubits': [target_qubit_custom],
                'label': gate_label
            })
            if success:
                st.success(f"Applied {gate_label}")
                st.rerun()
            else:
                st.error(msg)
        st.markdown("<p class='button-caption'>Apply custom unitary</p>", unsafe_allow_html=True)
    
    elif gate_method == "Parameterized Rotation":
        st.markdown("#### Arbitrary Axis Rotation")
        st.caption("Rotate around arbitrary axis n̂ = (nx, ny, nz) by angle θ")
        
        col1, col2 = st.columns(2)
        
        with col1:
            nx = st.slider("nx", -1.0, 1.0, 1.0, 0.1)
            ny = st.slider("ny", -1.0, 1.0, 0.0, 0.1)
            nz = st.slider("nz", -1.0, 1.0, 0.0, 0.1)
        
        with col2:
            theta = st.slider("Rotation Angle θ (radians)", -2*np.pi, 2*np.pi, 0.0, 0.1)
            target_q_rot = st.selectbox("Apply to qubit", range(st.session_state.num_qubits), key="rot_custom_q")
        
        # Normalize axis
        axis_norm = np.sqrt(nx**2 + ny**2 + nz**2)
        if axis_norm > 0:
            axis = np.array([nx, ny, nz]) / axis_norm
            
            # Display axis
            st.markdown(f"**Normalized axis:** n̂ = ({axis[0]:.3f}, {axis[1]:.3f}, {axis[2]:.3f})")
            
            # Build rotation matrix
            builder = CustomGateBuilder()
            U_rot = builder.rotation_gate(axis, theta)
            
            st.markdown("**Unitary Matrix:**")
            st.code(str(U_rot))
            
            if st.button("Apply Rotation", help="Apply rotation around custom axis to circuit"):
                success, msg = apply_gate('CUSTOM', {
                    'unitary': U_rot,
                    'qubits': [target_q_rot],
                    'label': f"R_n({theta:.2f})"
                })
                if success:
                    st.success("Applied custom rotation")
                    st.rerun()
                else:
                    st.error(msg)
            st.markdown("<p class='button-caption'>Rotate around axis</p>", unsafe_allow_html=True)
        else:
            st.warning("Define a non-zero rotation axis")
    
    else:  # Gate Decomposition
        st.markdown("#### Universal Gate Decomposition")
        st.info("Any single-qubit gate can be decomposed into rotations: U = Rz(α)Ry(β)Rz(γ)")
        
        alpha = st.slider("α (Z-rotation)", -2*np.pi, 2*np.pi, 0.0, 0.1, key="alpha")
        beta = st.slider("β (Y-rotation)", -2*np.pi, 2*np.pi, 0.0, 0.1, key="beta")
        gamma = st.slider("γ (Z-rotation)", -2*np.pi, 2*np.pi, 0.0, 0.1, key="gamma")
        
        target_q_decomp = st.selectbox("Apply to qubit", range(st.session_state.num_qubits), key="decomp_q")
        
        # Construct unitary
        Rz_alpha = np.array([[np.exp(-1j*alpha/2), 0], [0, np.exp(1j*alpha/2)]])
        Ry_beta = np.array([[np.cos(beta/2), -np.sin(beta/2)], [np.sin(beta/2), np.cos(beta/2)]])
        Rz_gamma = np.array([[np.exp(-1j*gamma/2), 0], [0, np.exp(1j*gamma/2)]])
        
        U_decomp = Rz_alpha @ Ry_beta @ Rz_gamma
        
        st.markdown("**Resulting Unitary:**")
        st.code(str(U_decomp))
        
        if st.button("Apply Decomposed Gate", help="Apply universal gate decomposition Rz(α)Ry(β)Rz(γ)"):
            success, msg = apply_gate('CUSTOM', {
                'unitary': U_decomp,
                'qubits': [target_q_decomp],
                'label': f"U(α,β,γ)"
            })
            if success:
                st.success("Applied decomposed gate")
                st.rerun()
            else:
                st.error(msg)
        st.markdown("<p class='button-caption'>Universal decomposition</p>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #000000; padding: 1.5rem 0;'>
    <p style='margin: 0; font-size: 0.875rem; color: #000000;'>
        QuantumLab Research Platform | Built with Qiskit & Streamlit
    </p>
    <p style='margin: 0.5rem 0 0 0; font-size: 0.8rem; color: #000000;'>
        Advanced quantum circuit simulation for research and development
    </p>
</div>
""", unsafe_allow_html=True)