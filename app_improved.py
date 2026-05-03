"""
QuantumLab - Advanced Quantum Circuit Research Platform (Enhanced)
========================================================
Professional-grade quantum computing research tool with advanced analysis,
state tomography, noise modeling, custom gate synthesis, and STATE ANIMATION.

Run with: streamlit run quantum_research_platform_enhanced.py
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

  * { font-family: 'IBM Plex Sans', -apple-system, sans-serif; }
  
  .stApp { background: #ffffff; color: #1a1a2e; }

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
    color: white;
    margin: 0 0 0.5rem 0;
    letter-spacing: -0.02em;
  }
  
  .research-header p {
    color: rgba(255, 255, 255, 0.95);
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
    color: #1a1a2e;
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
    color: #57606a !important;
    font-weight: 500 !important;
    font-size: 0.75rem !important;
  }
  
  [data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #1a1a2e !important;
    font-weight: 600 !important;
    font-family: 'IBM Plex Mono', monospace !important;
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
  }
  
  .secondary-btn div[data-testid="stButton"] > button {
    background: transparent !important;
    border: 1px solid #d0d7de !important;
    color: #1a1a2e !important;
  }
  
  .secondary-btn div[data-testid="stButton"] > button:hover {
    background: #f6f8fa !important;
    border-color: #8b949e !important;
  }
  
  .danger-btn div[data-testid="stButton"] > button {
    background: #da3633 !important;
    border: 1px solid #f85149 !important;
  }

  /* Code blocks */
  .code-block {
    background: #f6f8fa;
    border: 1px solid #d0d7de;
    border-radius: 6px;
    padding: 1rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.875rem;
    color: #1a1a2e;
    overflow-x: auto;
  }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: #ffffff;
  }
  
  .stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #57606a !important;
    font-weight: 500 !important;
    border-radius: 6px 6px 0 0 !important;
    padding: 0.5rem 1rem !important;
    border-bottom: 2px solid transparent !important;
  }
  
  .stTabs [aria-selected="true"] {
    color: #1a1a2e !important;
    border-bottom: 2px solid #1f6feb !important;
  }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: #f6f8fa !important;
    border-right: 1px solid #d0d7de !important;
  }

  /* Info boxes */
  .stInfo {
    background: rgba(56, 139, 253, 0.1) !important;
    border: 1px solid #1f6feb !important;
    color: #0550ae !important;
  }
  
  .stSuccess {
    background: rgba(31, 111, 235, 0.1) !important;
    border: 1px solid #1f6feb !important;
    color: #0969da !important;
  }
  
  .stWarning {
    background: rgba(187, 128, 9, 0.1) !important;
    border: 1px solid #9e6a03 !important;
    color: #7a4706 !important;
  }
  
  .stError {
    background: rgba(248, 81, 73, 0.1) !important;
    border: 1px solid #da3633 !important;
    color: #a40e26 !important;
  }

  /* Tables */
  .dataframe {
    background: #ffffff;
    color: #1a1a2e;
    border: 1px solid #d0d7de;
  }
  
  .dataframe th {
    background: #f6f8fa !important;
    color: #1a1a2e !important;
    font-weight: 600 !important;
  }

  /* Expander */
  .streamlit-expanderHeader {
    background: #f6f8fa;
    border: 1px solid #d0d7de;
    border-radius: 6px;
    color: #1a1a2e !important;
    font-weight: 500;
  }

  /* Text inputs */
  .stTextInput > div > div > input,
  .stNumberInput > div > div > input,
  .stSelectbox > div > div {
    background: #ffffff !important;
    border: 1px solid #d0d7de !important;
    color: #1a1a2e !important;
    border-radius: 6px !important;
  }

  /* Sliders */
  .stSlider > div > div > div {
    background: #f6f8fa;
  }

  /* General text overrides */
  .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
    color: #1a1a2e !important;
  }

  [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] .stMarkdown p,
  [data-testid="stSidebar"] .stMarkdown h1, [data-testid="stSidebar"] .stMarkdown h2,
  [data-testid="stSidebar"] .stMarkdown h3, [data-testid="stSidebar"] label {
    color: #1a1a2e !important;
  }

  [data-testid="stSidebar"] [data-testid="stMetricValue"] {
    color: #1a1a2e !important;
  }

  [data-testid="stSidebar"] [data-testid="metric-container"] label {
    color: #57606a !important;
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
        """Convert density matrix to Bloch vector."""
        x = 2 * np.real(rho[0, 1])
        y = -2 * np.imag(rho[0, 1])
        z = np.real(rho[0, 0] - rho[1, 1])
        return np.array([x, y, z], dtype=float)

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
    if "state_history" not in st.session_state:
        st.session_state.state_history = []
    if "record_states" not in st.session_state:
        st.session_state.record_states = False

init_session_state()

# ═══════════════════════════════════════════════════════════════════════════
# GATE OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════

def record_state_snapshot():
    """Record current quantum state for animation."""
    if st.session_state.record_states and st.session_state.gate_history:
        try:
            sv = Statevector.from_instruction(st.session_state.circuit)
            dm = DensityMatrix(sv)
            
            snapshot = {
                'step': len(st.session_state.state_history),
                'gate': st.session_state.gate_history[-1] if st.session_state.gate_history else "Initial",
                'statevector': sv.data.copy(),
                'density_matrix': dm.data.copy(),
                'probabilities': sv.probabilities_dict()
            }
            
            # For small systems, store Bloch vectors
            if st.session_state.num_qubits <= 3:
                analyzer = QuantumStateAnalyzer()
                bloch_vecs = []
                for i in range(st.session_state.num_qubits):
                    if st.session_state.num_qubits == 1:
                        rho_i = dm.data
                    else:
                        trace_qubits = [j for j in range(st.session_state.num_qubits) if j != i]
                        rho_i = partial_trace(sv, trace_qubits).data
                    bloch_vecs.append(analyzer.density_matrix_to_bloch(rho_i))
                snapshot['bloch_vectors'] = bloch_vecs
            
            st.session_state.state_history.append(snapshot)
        except:
            pass

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
        
        # Record state if recording is enabled
        record_state_snapshot()
        
        return True, label
    except Exception as e:
        return False, str(e)

# ═══════════════════════════════════════════════════════════════════════════
# VISUALIZATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def draw_bloch_sphere(title, vec, color='#1f6feb'):
    """Draw Bloch sphere with research-grade styling."""
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones_like(u), np.cos(v))
    
    fig = go.Figure()
    
    # Sphere surface
    fig.add_trace(go.Surface(
        x=x_sphere, y=y_sphere, z=z_sphere,
        colorscale=[[0, color], [1, color]],
        showscale=False,
        opacity=0.1,
        hoverinfo='skip'
    ))
    
    # Axes
    axis_length = 1.2
    fig.add_trace(go.Scatter3d(
        x=[-axis_length, axis_length], y=[0, 0], z=[0, 0],
        mode='lines', line=dict(color='#f85149', width=3),
        hoverinfo='skip', showlegend=False
    ))
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[-axis_length, axis_length], z=[0, 0],
        mode='lines', line=dict(color='#3fb950', width=3),
        hoverinfo='skip', showlegend=False
    ))
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, 0], z=[-axis_length, axis_length],
        mode='lines', line=dict(color='#58a6ff', width=4),
        hoverinfo='skip', showlegend=False
    ))
    
    # State vector
    if np.linalg.norm(vec) > 1e-6:
        fig.add_trace(go.Scatter3d(
            x=[0, vec[0]], y=[0, vec[1]], z=[0, vec[2]],
            mode='lines', line=dict(color=color, width=8),
            hoverinfo='text',
            hovertext=f'State<br>x:{vec[0]:.3f}<br>y:{vec[1]:.3f}<br>z:{vec[2]:.3f}',
            showlegend=False
        ))
        
        fig.add_trace(go.Cone(
            x=[vec[0]], y=[vec[1]], z=[vec[2]],
            u=[vec[0]*0.3], v=[vec[1]*0.3], w=[vec[2]*0.3],
            colorscale=[[0, color], [1, color]],
            showscale=False, sizemode='absolute', sizeref=0.3,
            hoverinfo='skip'
        ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color='#1a1a2e')),
        scene=dict(
            xaxis=dict(visible=False, range=[-1.5, 1.5]),
            yaxis=dict(visible=False, range=[-1.5, 1.5]),
            zaxis=dict(visible=False, range=[-1.5, 1.5]),
            bgcolor='#ffffff',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3)),
            aspectmode='cube'
        ),
        paper_bgcolor='#ffffff',
        margin=dict(l=0, r=0, t=40, b=0),
        height=500
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
        title=title,
        paper_bgcolor='#ffffff',
        plot_bgcolor='#f6f8fa',
        font=dict(color='#1a1a2e'),
        height=400
    )
    
    return fig

# ═══════════════════════════════════════════════════════════════════════════
# MAIN INTERFACE
# ═══════════════════════════════════════════════════════════════════════════

st.markdown(f"""
<div class="research-header" style="display: flex; align-items: center;">
    <div>
        <h1>⚛️ QuantumLab Enhanced</h1>
        <p>Advanced quantum circuit simulation with state evolution animation and comprehensive multi-qubit operations</p>
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
        st.session_state.state_history = []
        st.rerun()
    
    st.markdown("---")
    
    # State recording toggle
    st.markdown("### 🎬 Animation Control")
    record_toggle = st.toggle("Record State Evolution", value=st.session_state.record_states)
    
    if record_toggle != st.session_state.record_states:
        st.session_state.record_states = record_toggle
        if record_toggle:
            st.session_state.state_history = []
            st.success("Recording started!")
        else:
            st.info(f"Recorded {len(st.session_state.state_history)} states")
    
    if st.session_state.state_history:
        st.metric("Recorded Steps", len(st.session_state.state_history))
        if st.button("🗑️ Clear Recording"):
            st.session_state.state_history = []
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
    
    st.markdown("### 💾 Circuit Management")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📋 Copy QASM", use_container_width=True):
            try:
                from qiskit import qasm2
                qasm_str = qasm2.dumps(st.session_state.circuit)
                st.code(qasm_str, language='text')
            except Exception as e:
                st.error(f"Error: {e}")
    
    with col2:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.circuit = QuantumCircuit(st.session_state.num_qubits)
            st.session_state.gate_history = []
            st.session_state.analysis_cache = {}
            st.session_state.state_history = []
            st.rerun()

# Main content
tab_build, tab_animate, tab_analyze, tab_multi = st.tabs([
    "🔨 Circuit Builder",
    "🎬 State Animation",
    "📊 State Analysis",
    "🔗 Multi-Qubit Gates"
])

# ═══════════════════════════════════════════════════════════════════════════
# CIRCUIT BUILDER TAB
# ═══════════════════════════════════════════════════════════════════════════
with tab_build:
    st.markdown("### Quantum Circuit Construction")
    
    # Display circuit
    if st.session_state.gate_history:
        try:
            fig = circuit_drawer(st.session_state.circuit, output='mpl', 
                               style={'backgroundcolor': '#ffffff'})
            fig.patch.set_facecolor('#ffffff')
            st.pyplot(fig)
            plt.close(fig)
        except:
            st.code("\n".join(st.session_state.gate_history))
    else:
        st.info("Circuit is empty. Add gates below.")
    
    st.markdown("---")
    
    # Gate palette
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Single-Qubit Gates**")
        
        qubit_select = st.selectbox("Target Qubit", range(st.session_state.num_qubits), key="sq_qubit")
        
        gate_cols = st.columns(4)
        with gate_cols[0]:
            if st.button("H", use_container_width=True):
                apply_gate('H', {'qubit': qubit_select})
                st.rerun()
        with gate_cols[1]:
            if st.button("X", use_container_width=True):
                apply_gate('X', {'qubit': qubit_select})
                st.rerun()
        with gate_cols[2]:
            if st.button("Y", use_container_width=True):
                apply_gate('Y', {'qubit': qubit_select})
                st.rerun()
        with gate_cols[3]:
            if st.button("Z", use_container_width=True):
                apply_gate('Z', {'qubit': qubit_select})
                st.rerun()
        
        gate_cols2 = st.columns(4)
        with gate_cols2[0]:
            if st.button("S", use_container_width=True):
                apply_gate('S', {'qubit': qubit_select})
                st.rerun()
        with gate_cols2[1]:
            if st.button("T", use_container_width=True):
                apply_gate('T', {'qubit': qubit_select})
                st.rerun()
        with gate_cols2[2]:
            if st.button("S†", use_container_width=True):
                st.session_state.circuit.sdg(qubit_select)
                st.session_state.gate_history.append(f"S† q{qubit_select}")
                record_state_snapshot()
                st.rerun()
        with gate_cols2[3]:
            if st.button("T†", use_container_width=True):
                st.session_state.circuit.tdg(qubit_select)
                st.session_state.gate_history.append(f"T† q{qubit_select}")
                record_state_snapshot()
                st.rerun()
    
    with col2:
        st.markdown("**Rotation Gates**")
        
        rot_qubit = st.selectbox("Target Qubit", range(st.session_state.num_qubits), key="rot_qubit")
        rot_axis = st.selectbox("Axis", ["X", "Y", "Z"])
        rot_angle = st.number_input("Angle (radians)", -2*np.pi, 2*np.pi, 0.0, 0.1)
        
        if st.button(f"Apply R{rot_axis}({rot_angle:.2f})", use_container_width=True):
            apply_gate(f'R{rot_axis}', {'qubit': rot_qubit, 'angle': rot_angle})
            st.rerun()
        
        st.markdown("**Quick Angles**")
        quick_cols = st.columns(3)
        with quick_cols[0]:
            if st.button("π/4", key="pi4"):
                apply_gate(f'R{rot_axis}', {'qubit': rot_qubit, 'angle': np.pi/4})
                st.rerun()
        with quick_cols[1]:
            if st.button("π/2", key="pi2"):
                apply_gate(f'R{rot_axis}', {'qubit': rot_qubit, 'angle': np.pi/2})
                st.rerun()
        with quick_cols[2]:
            if st.button("π", key="pi"):
                apply_gate(f'R{rot_axis}', {'qubit': rot_qubit, 'angle': np.pi})
                st.rerun()
    
    with col3:
        st.markdown("**Two-Qubit Gates**")
        
        if st.session_state.num_qubits >= 2:
            control_q = st.selectbox("Control", range(st.session_state.num_qubits), key="control")
            target_q = st.selectbox("Target", range(st.session_state.num_qubits), key="target")
            
            if control_q == target_q:
                st.warning("Control ≠ Target required")
            else:
                gate_cols3 = st.columns(3)
                with gate_cols3[0]:
                    if st.button("CNOT", use_container_width=True):
                        apply_gate('CNOT', {'control': control_q, 'target': target_q})
                        st.rerun()
                with gate_cols3[1]:
                    if st.button("CZ", use_container_width=True):
                        apply_gate('CZ', {'control': control_q, 'target': target_q})
                        st.rerun()
                with gate_cols3[2]:
                    if st.button("SWAP", use_container_width=True):
                        apply_gate('SWAP', {'control': control_q, 'target': target_q})
                        st.rerun()
        else:
            st.info("Need ≥2 qubits for two-qubit gates")

# ═══════════════════════════════════════════════════════════════════════════
# STATE ANIMATION TAB
# ═══════════════════════════════════════════════════════════════════════════
with tab_animate:
    st.markdown("### 🎬 Quantum State Evolution Animation")
    
    if not st.session_state.state_history:
        st.info("Enable 'Record State Evolution' in the sidebar and build a circuit to see animation.")
    else:
        st.success(f"✓ {len(st.session_state.state_history)} state snapshots recorded")
        
        # Animation controls
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            step_idx = st.slider(
                "Animation Step",
                0,
                len(st.session_state.state_history) - 1,
                0,
                help="Slide to see state evolution"
            )
        
        with col2:
            auto_play = st.button("▶️ Auto Play")
        
        with col3:
            speed = st.select_slider("Speed", options=[0.5, 1, 2, 3], value=1)
        
        # Get current state
        current_state = st.session_state.state_history[step_idx]
        
        st.markdown(f"### Step {step_idx}: {current_state['gate']}")
        
        # Auto-play functionality
        if auto_play:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i in range(len(st.session_state.state_history)):
                state = st.session_state.state_history[i]
                progress_bar.progress(i / (len(st.session_state.state_history) - 1))
                status_text.text(f"Step {i}: {state['gate']}")
                time.sleep(0.5 / speed)
            
            progress_bar.empty()
            status_text.empty()
            st.success("Animation complete!")
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Probability Distribution")
            
            probs = current_state['probabilities']
            
            fig = go.Figure(data=[
                go.Bar(
                    x=[f"|{k}⟩" for k in probs.keys()],
                    y=list(probs.values()),
                    marker_color='#1f6feb',
                    text=[f"{v*100:.1f}%" for v in probs.values()],
                    textposition='auto'
                )
            ])
            
            fig.update_layout(
                xaxis_title="Basis State",
                yaxis_title="Probability",
                paper_bgcolor='#ffffff',
                plot_bgcolor='#f6f8fa',
                font=dict(color='#1a1a2e'),
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### State Vector Amplitudes")
            
            sv_data = current_state['statevector']
            amplitudes = np.abs(sv_data)
            phases = np.angle(sv_data)
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                name='Magnitude',
                x=[f"|{i:0{st.session_state.num_qubits}b}⟩" for i in range(len(sv_data))],
                y=amplitudes,
                marker_color='#1f6feb'
            ))
            
            fig.update_layout(
                xaxis_title="Basis State",
                yaxis_title="|Amplitude|",
                paper_bgcolor='#ffffff',
                plot_bgcolor='#f6f8fa',
                font=dict(color='#1a1a2e'),
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Bloch sphere animation for small systems
        if 'bloch_vectors' in current_state and st.session_state.num_qubits <= 3:
            st.markdown("---")
            st.markdown("#### Bloch Sphere Representation")
            
            bloch_cols = st.columns(min(st.session_state.num_qubits, 3))
            
            for i, col in enumerate(bloch_cols):
                with col:
                    vec = current_state['bloch_vectors'][i]
                    fig = draw_bloch_sphere(f"Qubit {i}", vec)
                    st.plotly_chart(fig, use_container_width=True)
        
        # Evolution trajectory
        if len(st.session_state.state_history) > 1 and st.session_state.num_qubits <= 2:
            st.markdown("---")
            st.markdown("#### Evolution Trajectory")
            
            # Track probability evolution
            all_states = set()
            for state in st.session_state.state_history:
                all_states.update(state['probabilities'].keys())
            
            all_states = sorted(list(all_states))
            
            fig = go.Figure()
            
            for basis_state in all_states:
                probs_over_time = []
                for state in st.session_state.state_history:
                    probs_over_time.append(state['probabilities'].get(basis_state, 0))
                
                fig.add_trace(go.Scatter(
                    x=list(range(len(st.session_state.state_history))),
                    y=probs_over_time,
                    mode='lines+markers',
                    name=f"|{basis_state}⟩",
                    line=dict(width=2),
                    marker=dict(size=6)
                ))
            
            # Highlight current step
            fig.add_vline(x=step_idx, line_dash="dash", line_color="red", 
                         annotation_text="Current Step")
            
            fig.update_layout(
                title="Probability Evolution Over Time",
                xaxis_title="Step",
                yaxis_title="Probability",
                paper_bgcolor='#ffffff',
                plot_bgcolor='#f6f8fa',
                font=dict(color='#1a1a2e'),
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)

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
                if st.session_state.num_qubits <= 3:
                    for i in range(st.session_state.num_qubits):
                        # Partial trace to get single qubit state
                        if st.session_state.num_qubits == 1:
                            rho_i = dm.data
                        else:
                            trace_qubits = [j for j in range(st.session_state.num_qubits) if j != i]
                            rho_i = partial_trace(sv, trace_qubits).data
                        
                        bloch_vec = analyzer.density_matrix_to_bloch(rho_i)
                        fig = draw_bloch_sphere(f"Qubit {i}", bloch_vec)
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Too many qubits for full Bloch sphere display. Showing first 3.")
                    for i in range(3):
                        trace_qubits = [j for j in range(st.session_state.num_qubits) if j != i]
                        rho_i = partial_trace(sv, trace_qubits).data
                        bloch_vec = analyzer.density_matrix_to_bloch(rho_i)
                        fig = draw_bloch_sphere(f"Qubit {i}", bloch_vec)
                        st.plotly_chart(fig, use_container_width=True)
            
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
                        textposition='auto'
                    )
                ])
                
                fig.update_layout(
                    title="Measurement Outcome Distribution",
                    xaxis_title="Basis State",
                    yaxis_title="Probability",
                    paper_bgcolor='#ffffff',
                    plot_bgcolor='#f6f8fa',
                    font=dict(color='#1a1a2e'),
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"Analysis error: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# MULTI-QUBIT GATES TAB
# ═══════════════════════════════════════════════════════════════════════════
with tab_multi:
    st.markdown("### 🔗 Advanced Multi-Qubit Gates")
    
    if st.session_state.num_qubits < 2:
        st.warning("Need at least 2 qubits for multi-qubit operations. Adjust in sidebar.")
    else:
        st.markdown("#### Two-Qubit Gate Operations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**CNOT (Controlled-X)**")
            st.caption("Flips target if control is |1⟩")
            
            cnot_ctrl = st.selectbox("Control qubit", range(st.session_state.num_qubits), key="cnot_ctrl")
            cnot_tgt = st.selectbox("Target qubit", range(st.session_state.num_qubits), key="cnot_tgt")
            
            if cnot_ctrl != cnot_tgt:
                if st.button("Apply CNOT", use_container_width=True):
                    apply_gate('CNOT', {'control': cnot_ctrl, 'target': cnot_tgt})
                    st.rerun()
            else:
                st.error("Control and target must be different")
            
            st.markdown("---")
            
            st.markdown("**CZ (Controlled-Z)**")
            st.caption("Applies Z to target if control is |1⟩")
            
            cz_ctrl = st.selectbox("Control qubit", range(st.session_state.num_qubits), key="cz_ctrl")
            cz_tgt = st.selectbox("Target qubit", range(st.session_state.num_qubits), key="cz_tgt")
            
            if cz_ctrl != cz_tgt:
                if st.button("Apply CZ", use_container_width=True):
                    apply_gate('CZ', {'control': cz_ctrl, 'target': cz_tgt})
                    st.rerun()
            else:
                st.error("Control and target must be different")
        
        with col2:
            st.markdown("**SWAP**")
            st.caption("Exchanges quantum states of two qubits")
            
            swap_q1 = st.selectbox("Qubit 1", range(st.session_state.num_qubits), key="swap_q1")
            swap_q2 = st.selectbox("Qubit 2", range(st.session_state.num_qubits), key="swap_q2")
            
            if swap_q1 != swap_q2:
                if st.button("Apply SWAP", use_container_width=True):
                    apply_gate('SWAP', {'control': swap_q1, 'target': swap_q2})
                    st.rerun()
            else:
                st.error("Must select different qubits")
            
            st.markdown("---")
            
            st.markdown("**Controlled Rotations**")
            
            crot_ctrl = st.selectbox("Control", range(st.session_state.num_qubits), key="crot_ctrl")
            crot_tgt = st.selectbox("Target", range(st.session_state.num_qubits), key="crot_tgt")
            crot_axis = st.selectbox("Axis", ["X", "Y", "Z"], key="crot_axis")
            crot_angle = st.number_input("Angle (rad)", -np.pi, np.pi, 0.0, 0.1, key="crot_angle")
            
            if crot_ctrl != crot_tgt:
                if st.button(f"Apply CR{crot_axis}", use_container_width=True):
                    if crot_axis == "X":
                        st.session_state.circuit.crx(crot_angle, crot_ctrl, crot_tgt)
                    elif crot_axis == "Y":
                        st.session_state.circuit.cry(crot_angle, crot_ctrl, crot_tgt)
                    else:
                        st.session_state.circuit.crz(crot_angle, crot_ctrl, crot_tgt)
                    
                    st.session_state.gate_history.append(
                        f"CR{crot_axis}({crot_angle:.3f}) q{crot_ctrl}→q{crot_tgt}"
                    )
                    record_state_snapshot()
                    st.rerun()
        
        # Three-qubit gates
        if st.session_state.num_qubits >= 3:
            st.markdown("---")
            st.markdown("#### Three-Qubit Gates")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Toffoli (CCX)**")
                st.caption("Controlled-controlled-X: flips target if both controls are |1⟩")
                
                tof_c1 = st.selectbox("Control 1", range(st.session_state.num_qubits), key="tof_c1")
                tof_c2 = st.selectbox("Control 2", range(st.session_state.num_qubits), key="tof_c2")
                tof_tgt = st.selectbox("Target", range(st.session_state.num_qubits), key="tof_tgt")
                
                if len({tof_c1, tof_c2, tof_tgt}) == 3:
                    if st.button("Apply Toffoli", use_container_width=True):
                        apply_gate('Toffoli', {
                            'control1': tof_c1,
                            'control2': tof_c2,
                            'target': tof_tgt
                        })
                        st.rerun()
                else:
                    st.error("All qubits must be different")
            
            with col2:
                st.markdown("**Fredkin (CSWAP)**")
                st.caption("Controlled-SWAP: swaps targets if control is |1⟩")
                
                frd_ctrl = st.selectbox("Control", range(st.session_state.num_qubits), key="frd_ctrl")
                frd_t1 = st.selectbox("Target 1", range(st.session_state.num_qubits), key="frd_t1")
                frd_t2 = st.selectbox("Target 2", range(st.session_state.num_qubits), key="frd_t2")
                
                if len({frd_ctrl, frd_t1, frd_t2}) == 3:
                    if st.button("Apply Fredkin", use_container_width=True):
                        st.session_state.circuit.cswap(frd_ctrl, frd_t1, frd_t2)
                        st.session_state.gate_history.append(
                            f"CSWAP q{frd_ctrl}: q{frd_t1}↔q{frd_t2}"
                        )
                        record_state_snapshot()
                        st.rerun()
                else:
                    st.error("All qubits must be different")
        
        # Quick entanglement circuits
        st.markdown("---")
        st.markdown("#### Quick Entanglement Circuits")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Bell State |Φ+⟩", use_container_width=True):
                if st.session_state.num_qubits >= 2:
                    st.session_state.circuit.h(0)
                    st.session_state.circuit.cx(0, 1)
                    st.session_state.gate_history.extend(["H q0", "CNOT q0→q1"])
                    record_state_snapshot()
                    st.rerun()
        
        with col2:
            if st.button("GHZ State", use_container_width=True):
                if st.session_state.num_qubits >= 3:
                    st.session_state.circuit.h(0)
                    for i in range(1, min(st.session_state.num_qubits, 3)):
                        st.session_state.circuit.cx(0, i)
                    st.session_state.gate_history.append("GHZ state preparation")
                    record_state_snapshot()
                    st.rerun()
        
        with col3:
            if st.button("W State", use_container_width=True):
                if st.session_state.num_qubits >= 3:
                    # W state for 3 qubits
                    theta1 = np.arccos(np.sqrt(1/3))
                    theta2 = np.arccos(np.sqrt(1/2))
                    
                    st.session_state.circuit.ry(theta1, 0)
                    st.session_state.circuit.ch(0, 1)
                    st.session_state.circuit.x(0)
                    st.session_state.circuit.ch(0, 2)
                    st.session_state.circuit.x(0)
                    
                    st.session_state.gate_history.append("W state preparation")
                    record_state_snapshot()
                    st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #57606a; padding: 1.5rem 0;'>
    <p style='margin: 0; font-size: 0.875rem;'>
        QuantumLab Enhanced | Built with Qiskit & Streamlit
    </p>
    <p style='margin: 0.5rem 0 0 0; font-size: 0.8rem;'>
        Advanced quantum circuit simulation with state evolution animation
    </p>
</div>
""", unsafe_allow_html=True)