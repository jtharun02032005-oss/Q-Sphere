"""
QuantumPy – 2-Qubit Quantum Visualiser (Streamlit Web App)
===========================================================
Run with:  streamlit run app.py
"""

import warnings
import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")           # non-interactive backend – required for Streamlit
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import io, base64

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, partial_trace
from qiskit.visualization import plot_bloch_multivector
import base64
import os

warnings.filterwarnings("ignore")



# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&family=JetBrains+Mono:wght@400;600&family=Orbitron:wght@700;900&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* ── background ── */
  .stApp { background: #0a0e1a; }

  /* ── hero banner ── */
  .hero {
    background: linear-gradient(135deg, #1a1f3c 0%, #0d1117 60%, #111827 100%);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 18px;
    padding: 36px 40px 28px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
  }
  .hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 260px; height: 260px;
    background: radial-gradient(circle, rgba(99,102,241,0.18) 0%, transparent 70%);
    border-radius: 50%;
  }
  .hero h1 {
    font-family: 'Copperplate Gothic Light', 'Orbitron', sans-serif;
    font-size: 2.6rem; font-weight: 900;
    color: rgb(255, 255, 0);
    margin: 0 0 8px;
  }
  .hero p { color: #94a3b8; font-size: 1.05rem; margin: 0; }

  /* ── section cards ── */
  .section-card {
    background: linear-gradient(135deg, #111827 0%, #1a1f3c 100%);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 14px;
    padding: 20px 22px;
    margin-bottom: 18px;
  }
  .section-title {
    font-size: 0.85rem; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; color: #818cf8; margin-bottom: 14px;
  }

  /* ── gate buttons ── */
  div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #1e293b, #0f172a) !important;
    color: #e2e8f0 !important;
    border: 1px solid rgba(99,102,241,0.35) !important;
    border-radius: 10px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important; font-weight: 600 !important;
    padding: 8px 14px !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
  }
  div[data-testid="stButton"] > button:hover {
    background: linear-gradient(135deg, #4338ca, #7c3aed) !important;
    border-color: #818cf8 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(99,102,241,0.35) !important;
  }
  div[data-testid="stButton"] > button:active {
    transform: translateY(0) !important;
  }

  /* ── primary action buttons ── */
  .primary-btn div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #f97316, #ef4444) !important;
    border-color: #f97316 !important;
    color: white !important;
    font-size: 0.9rem !important;
    padding: 10px 18px !important;
  }
  .primary-btn div[data-testid="stButton"] > button:hover {
    background: linear-gradient(135deg, #fb923c, #f87171) !important;
    box-shadow: 0 6px 24px rgba(249,115,22,0.42) !important;
  }

  /* ── gate history pill ── */
  .gate-history {
    background: #0d1117;
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 10px;
    padding: 12px 18px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.88rem;
    color: #a5f3fc;
    min-height: 46px;
    word-wrap: break-word;
    line-height: 1.8;
  }
  .gate-pill {
    display: inline-block;
    background: linear-gradient(135deg, rgba(99,102,241,0.25), rgba(124,58,237,0.25));
    border: 1px solid rgba(99,102,241,0.4);
    border-radius: 6px;
    padding: 2px 10px;
    margin: 2px 3px;
    color: #c4b5fd;
    font-weight: 600;
  }

  /* ── output box ── */
  .output-box {
    background: #0d1117;
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 12px;
    padding: 16px 20px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    color: #86efac;
    white-space: pre-wrap;
    max-height: 340px;
    overflow-y: auto;
  }

  /* ── sidebar ── */
  [data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid rgba(99,102,241,0.18) !important;
  }
  [data-testid="stSidebar"] * { color: #94a3b8 !important; }

  /* ── select / slider ── */
  .stSelectbox > div > div, .stSlider { color: #e2e8f0; }

  /* ── tabs ── */
  .stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #64748b !important;
    font-weight: 600 !important;
    border-bottom: 2px solid transparent !important;
  }
  .stTabs [aria-selected="true"] {
    color: #818cf8 !important;
    border-bottom: 2px solid #818cf8 !important;
  }

  /* ── horizontal rule ── */
  hr { border-color: rgba(99,102,241,0.15); }

  /* ── metric ── */
  [data-testid="metric-container"] {
    background: rgba(99,102,241,0.07);
    border: 1px solid rgba(99,102,241,0.18);
    border-radius: 10px; padding: 12px;
  }
  [data-testid="metric-container"] label { color: #818cf8 !important; }
  [data-testid="metric-container"] [data-testid="stMetricValue"] { color: #f1f5f9 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG & SIDEBAR MODE
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="QuantumPy – Visualiser",
    page_icon="⚛️",
    layout="wide",
)

st.sidebar.markdown('<div style="color: #818cf8; font-weight: bold; font-size: 1.2rem; margin-bottom: 10px;">⚙️ Mode Selection</div>', unsafe_allow_html=True)
mode = st.sidebar.radio("System Size", ["1-Qubit Explorer", "2-Qubit Explorer"])

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE – persistent circuit between reruns
# ─────────────────────────────────────────────────────────────────────────────
if "mode" not in st.session_state:
    st.session_state.mode = mode

# Reset circuit if mode changes
if mode != st.session_state.mode:
    st.session_state.mode = mode
    st.session_state.circuit = QuantumCircuit(1) if mode == "1-Qubit Explorer" else QuantumCircuit(2)
    st.session_state.gate_history = []
    st.session_state.output = ""

if "circuit" not in st.session_state:
    st.session_state.circuit = QuantumCircuit(1) if mode == "1-Qubit Explorer" else QuantumCircuit(2)
if "gate_history" not in st.session_state:
    st.session_state.gate_history = []   # list of dicts
if "output" not in st.session_state:
    st.session_state.output = ""

def push_gate(label: str, fn):
    """Apply callable *fn* to circuit, record history."""
    fn(st.session_state.circuit)
    st.session_state.gate_history.append(label)

# ─────────────────────────────────────────────────────────────────────────────
# BLOCH HELPERS
# ─────────────────────────────────────────────────────────────────────────────
from qiskit.quantum_info import DensityMatrix

def density_to_bloch(rho):
    x = 2 * np.real(rho[0, 1])
    y = -2 * np.imag(rho[0, 1])
    z = np.real(rho[0, 0] - rho[1, 1])
    return np.array([x, y, z], dtype=float)

def bloch_vectors(sv, num_qubits=2):
    if num_qubits == 1:
        rho = DensityMatrix(sv).data
        return density_to_bloch(rho), None
    else:
        rho0 = partial_trace(sv, [1]).data
        rho1 = partial_trace(sv, [0]).data
        return density_to_bloch(rho0), density_to_bloch(rho1)

def draw_sphere(ax, title, vec, color):
    """Clean, bright Bloch sphere — no glow, just vivid crisp lines."""
    ax.clear()

    # ── Sphere wireframe: single bright layer ──
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(xs, ys, zs, color=color, alpha=0.20, linewidth=0.5)

    # ── Equator ring (dimmed) ──
    ring = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(ring), np.sin(ring), np.zeros_like(ring),
            color=color, alpha=0.40, linewidth=1.2)

    # ── Meridian circles (XZ and YZ) — dim ──
    circ = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(circ), np.zeros_like(circ), np.sin(circ),
            color=color, alpha=0.18, linewidth=0.8)
    ax.plot(np.zeros_like(circ), np.cos(circ), np.sin(circ),
            color=color, alpha=0.18, linewidth=0.8)

    # ── Axis lines (bright, single pass) ──
    ax.plot([-1.1, 1.1], [0, 0], [0, 0], color='#ff4d4d', linewidth=1.5, alpha=1.0)
    ax.plot([0, 0], [-1.1, 1.1], [0, 0], color='#39ff14', linewidth=1.5, alpha=1.0)
    ax.plot([0, 0], [0, 0], [-1.1, 1.1], color='#ffe600', linewidth=2.0, alpha=1.0)

    # ── Axis & pole labels ──
    ax.text( 1.22,  0,     0,     '+X', color='#ff4d4d', fontsize=9,  fontweight='bold')
    ax.text(-1.32,  0,     0,     '−X', color='#ff4d4d', fontsize=8)
    ax.text( 0,     1.22,  0,     '+Y', color='#39ff14', fontsize=9,  fontweight='bold')
    ax.text( 0,    -1.32,  0,     '−Y', color='#39ff14', fontsize=8)
    ax.text( 0,     0,     1.22,  '|0⟩', color='#ffe600', fontsize=11, fontweight='bold')
    ax.text( 0,     0,    -1.38,  '|1⟩', color='#ffe600', fontsize=11, fontweight='bold')

    # ── State vector: single solid bright arrow ──
    ax.quiver(0, 0, 0, vec[0], vec[1], vec[2],
              color=color, linewidth=3.5,
              arrow_length_ratio=0.18, alpha=1.0, normalize=False)

    # ── Bright dot at tip ──
    tip = np.array([vec[0], vec[1], vec[2]])
    if np.linalg.norm(tip) > 1e-6:
        ax.scatter([tip[0]], [tip[1]], [tip[2]],
                   color='#ffffff', s=60, zorder=10, alpha=1.0, edgecolors=color, linewidths=2)

    # ── Clean axes styling (no tinted panes) ──
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

# ─────────────────────────────────────────────────────────────────────────────
# BUILD STEP STATES (for animation / step view)
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
    'SWAP': lambda c, _: c.swap(0, 1),
}

def build_step_states(circuit=None):
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
            except Exception:
                # Fallback if power is not supported
                temp_full = QuantumCircuit(1)
                temp_full.append(inst, qargs, cargs)
                current_sv = current_sv.evolve(temp_full)
                states.append(current_sv)
                labels.append(name)
                pause_indices.append(len(states) - 1)
        return states, labels, pause_indices

    # 2-Qubit mode: standard step-by-step
    temp = QuantumCircuit(2)
    states = [Statevector.from_instruction(temp)]
    labels = []
    pause_indices = [0]
    for label in st.session_state.gate_history:
        # label examples: "H q0", "RX(0.50π) q1", "CNOT"
        parts = label.split()
        gate_name = parts[0]
        qubit = int(parts[1][-1]) if len(parts) > 1 else 0

        if gate_name in _GATE_LABEL_TO_OP:
            _GATE_LABEL_TO_OP[gate_name](temp, qubit)
        elif gate_name.startswith("RX"):
            angle_str = label.split("(")[1].split("π")[0]
            temp.rx(float(angle_str) * np.pi, qubit)
        elif gate_name.startswith("RY"):
            angle_str = label.split("(")[1].split("π")[0]
            temp.ry(float(angle_str) * np.pi, qubit)
        elif gate_name.startswith("RZ"):
            angle_str = label.split("(")[1].split("π")[0]
            temp.rz(float(angle_str) * np.pi, qubit)

        states.append(Statevector.from_instruction(temp))
        labels.append(gate_name)
        pause_indices.append(len(states) - 1)
    return states, labels, pause_indices

# ─────────────────────────────────────────────────────────────────────────────
# HERO BANNER
# ─────────────────────────────────────────────────────────────────────────────
logo_html = ""
logo_path = "logo.png"
if not os.path.exists(logo_path) and os.path.exists("logo.png.png"):
    logo_path = "logo.png.png"

if os.path.exists(logo_path):
    with open(logo_path, "rb") as f:
        b64_logo = base64.b64encode(f.read()).decode()
    logo_html = f'<img src="data:image/png;base64,{b64_logo}" style="height: 110px; margin-right: 25px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.5); flex-shrink: 0;"/>'

st.markdown(f"""
<div class="hero" style="display: flex; align-items: center;">
  {logo_html}
  <div>
      <h1 style="margin-top: 0; margin-bottom: 5px;">Q-SPHERE </h1>
      <p style="margin: 0;">Interactive Bloch Sphere Animation &nbsp;·&nbsp; 1-Qubit Quantum Circuit Visualiser &nbsp;·&nbsp; 2-Qubit Quantum Circuit Visualiser</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# LAYOUT: sidebar = info / metrics   |   main = gates + output
# ─────────────────────────────────────────────────────────────────────────────
sidebar, main = st.columns([1, 3], gap="large")

# ═══════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════
with sidebar:
    st.markdown("### 📊 Circuit Info")
    st.metric("Gates Applied", len(st.session_state.gate_history))
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
    st.markdown("### ⓘ About")
    st.markdown("""
<small style='color:#64748b;line-height:1.6'>
<b style='color:#818cf8'>Q-SPHERE</b> visualises a 2-qubit 
quantum circuit in real time.<br><br>

• Single-qubit gates: X Y Z H S S† T T†<br>
• Rotation gates: Rₓ Rᵧ R_z<br>
• Two-qubit gates: CNOT CZ SWAP<br>
• Animated Bloch sphere step-by-step<br>
• Statevector + probability readout<br>
• Text circuit diagram<br><br>

<i></i> 

</small>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════
# MAIN CONTENT
# ═══════════════════════════════════════════════════
with main:

    # ── Gate History ─────────────────────────────
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

    # ── Tabs for gate categories ──────────────────
    if mode == "1-Qubit Explorer":
        tab_op, tab_q0, tab_rot = st.tabs(["ⓘ Operators", "🟣 Single Qubit Gates", " Rotation Gates"])
    else:
        tab_op, tab_q0, tab_q1, tab_2q, tab_rot = st.tabs(["ⓘ Operators", "🟣 Qubit 0 Gates", "🔵 Qubit 1 Gates", " 2-Qubit Gates", "Rotation Gates"])

    # ────── Operators (Legend) ────────────────────
    with tab_op:
        st.markdown('''
        <div class="section-title">Info about the gate buttons</div>
        <div style="color: #cbd5e1; font-family: 'Inter', sans-serif; font-size: 0.9rem; line-height: 1.6; background: rgba(13, 17, 23, 0.4); padding: 20px; border-radius: 12px; border: 1px solid rgba(99,102,241,0.15);">
        <b>X</b> = flips the state of qubit<br>
        <b>Y</b> = rotates the state vector about Y-axis<br>
        <b>Z</b> = flips the phase by π radians<br>
        <b>Rx</b> = parameterized rotation about the X axis<br>
        <b>Ry</b> = parameterized rotation about the Y axis<br>
        <b>Rz</b> = parameterized rotation about the Z axis<br>
        <b>S</b> = rotates the state vector about Z axis by π/2 radians<br>
        <b>T</b> = rotates the state vector about Z axis by π/4 radians<br>
        <b>Sd (S†)</b> = rotates the state vector about Z axis by −π/2 radians<br>
        <b>Td (T†)</b> = rotates the state vector about Z axis by −π/4 radians<br>
        <b>H</b> = creates the state of superposition<br>
        <b>CNOT</b> — Flips the target qubit only when the control qubit is |1⟩.<br>
        <b>CZ</b> — Applies a phase flip when both qubits are |1⟩.<br>
        <b>SWAP</b> — Exchanges the quantum states of two qubits.<br><br>
        <span style="color: #818cf8; font-weight: bold;">For Rx, Ry and Rz:</span><br>
        θ (rotation angle) allowed range in the app is [-2π, 2π]
        </div>
        ''', unsafe_allow_html=True)

    # ────── Qubit 0 ──────────────────────────────
    with tab_q0:
        st.markdown('<div class="section-title">Single-Qubit Gates → q0</div>', unsafe_allow_html=True)
        cols = st.columns(8)
        for col, gate in zip(cols, ['X','Y','Z','H','S','SD','T','TD']):
            with col:
                label = "S†" if gate == "SD" else ("T†" if gate == "TD" else gate)
                if st.button(f"{label}₀", key=f"q0_{gate}"):
                    push_gate(f"{gate} q0", lambda c, g=gate: _GATE_LABEL_TO_OP[g](c, 0))
                    st.rerun()

    # ────── Qubit 1 ──────────────────────────────
    if mode == "2-Qubit Explorer":
        with tab_q1:
            st.markdown('<div class="section-title">Single-Qubit Gates → q1</div>', unsafe_allow_html=True)
            cols = st.columns(8)
            for col, gate in zip(cols, ['X','Y','Z','H','S','SD','T','TD']):
                with col:
                    label = "S†" if gate == "SD" else ("T†" if gate == "TD" else gate)
                    if st.button(f"{label}₁", key=f"q1_{gate}"):
                        push_gate(f"{gate} q1", lambda c, g=gate: _GATE_LABEL_TO_OP[g](c, 1))
                        st.rerun()

        # ────── 2-Qubit Gates ─────────────────────────
        with tab_2q:
            st.markdown('<div class="section-title">Entangling & Two-Qubit Gates</div>', unsafe_allow_html=True)
            c1, c2, c3, _ = st.columns([1,1,1,3])
            with c1:
                if st.button("CNOT q0→q1", key="cnot"):
                    push_gate("CNOT q0", lambda c: c.cx(0, 1))
                    st.rerun()
            with c2:
                if st.button("CZ", key="cz"):
                    push_gate("CZ q0", lambda c: c.cz(0, 1))
                    st.rerun()
            with c3:
                if st.button("SWAP", key="swap"):
                    push_gate("SWAP q0", lambda c: c.swap(0, 1))
                    st.rerun()

    # ────── Rotation Gates ───────────────────────
    with tab_rot:
        st.markdown('<div class="section-title">Rotation Gates (Rₓ Rᵧ R_z)</div>', unsafe_allow_html=True)
        rc1, rc2, rc3, rc4 = st.columns([1,1,1,2])
        with rc1:
            rot_axis = st.selectbox("Axis", ["X","Y","Z"], key="rot_axis")
        with rc2:
            if mode == "1-Qubit Explorer":
                rot_qubit = 0
            else:
                rot_qubit = st.selectbox("Qubit", [0, 1], key="rot_qubit")
        with rc3:
            rot_multiple = st.selectbox("θ (×π)", [0.25, 0.5, 1.0, 2.0, -0.25, -0.5, -1.0, -2.0],
                                        format_func=lambda x: f"{x}π", key="rot_multiple")
        with rc4:
            if st.button(f"Apply R{rot_axis}({rot_multiple}π) on q{rot_qubit}", key="apply_rot"):
                theta = rot_multiple * np.pi
                q = rot_qubit
                ax_lower = rot_axis.lower()
                angle_label = f"{rot_multiple}π"
                label = f"R{rot_axis}({angle_label}) q{rot_qubit}"
                if ax_lower == 'x':
                    push_gate(label, lambda c, t=theta, qq=q: c.rx(t, qq))
                elif ax_lower == 'y':
                    push_gate(label, lambda c, t=theta, qq=q: c.ry(t, qq))
                elif ax_lower == 'z':
                    push_gate(label, lambda c, t=theta, qq=q: c.rz(t, qq))
                st.rerun()

    st.markdown("---")

    # ─────────────────────────────────────────────
    # ACTION BUTTONS
    # ─────────────────────────────────────────────
    a1, a2, a3, a4 = st.columns(4)
    with a1:
        show_sv    = st.button(" Statevector", key="btn_sv",    use_container_width=True)
    with a2:
        show_bloch = st.button("Bloch Spheres",key="btn_bloch", use_container_width=True)
    with a3:
        show_anim  = st.button("Animate",       key="btn_anim",  use_container_width=True)
    with a4:
        clear_btn  = st.button("Clear",          key="btn_clear", use_container_width=True)

    if clear_btn:
        st.session_state.circuit = QuantumCircuit(1) if mode == "1-Qubit Explorer" else QuantumCircuit(2)
        st.session_state.gate_history = []
        st.session_state.output = ""
        st.rerun()

    # ─────────────────────────────────────────────
    # STATEVECTOR OUTPUT
    # ─────────────────────────────────────────────
    if show_sv:
        try:
            sv = Statevector.from_instruction(st.session_state.circuit)
            probs = sv.probabilities_dict()
            lines = ["Statevector:\n", str(sv), "\nProbabilities:"]
            for state, prob in probs.items():
                lines.append(f"  |{state}⟩  ──  {prob:.6f}")
            st.session_state.output = "\n".join(lines)
        except Exception as e:
            st.session_state.output = f"Error: {e}"

    # ─────────────────────────────────────────────
    # STATIC BLOCH SPHERES
    # ─────────────────────────────────────────────
    if show_bloch:
        try:
            num_q = 1 if mode == "1-Qubit Explorer" else 2
            sv = Statevector.from_instruction(st.session_state.circuit)
            b0, b1 = bloch_vectors(sv, num_q)

            if num_q == 1:
                fig = plt.figure(figsize=(7, 7), facecolor='#000000')
                ax = fig.add_subplot(111, projection='3d')
                draw_sphere(ax, "Qubit State", b0, '#bf40ff')
            else:
                fig = plt.figure(figsize=(14, 7), facecolor='#000000')
                ax1 = fig.add_subplot(121, projection='3d')
                ax2 = fig.add_subplot(122, projection='3d')
                draw_sphere(ax1, "Qubit 0", b0, '#bf40ff')
                draw_sphere(ax2, "Qubit 1", b1, '#00e5ff')
            
            fig.suptitle("Bloch Sphere Representation", fontsize=16,
                         fontweight='bold', color='#ffffff', y=0.95)
            fig.patch.set_facecolor('#000000')
            plt.tight_layout(pad=2.0)
            st.pyplot(fig)
            plt.close(fig)
        except Exception as e:
            st.error(f"Bloch Error: {e}")

    # ─────────────────────────────────────────────
    # ANIMATED BLOCH STEP-BY-STEP
    # ─────────────────────────────────────────────
    if show_anim:
        if not st.session_state.gate_history:
            st.warning("Apply at least one gate before animating.")
        else:
            try:
                states, gate_labels, pause_indices = build_step_states()
                n_frames = len(states)

                progress = st.progress(0, text="Rendering animation frames…")
                frames_b64 = []

                num_q = 1 if mode == "1-Qubit Explorer" else 2

                for i, (sv, label) in enumerate(zip(states, gate_labels + [""])):
                    b0, b1 = bloch_vectors(sv, num_q)
                    title = f" Applying  {label}" if i < len(gate_labels) else "Final State"

                    if num_q == 1:
                        fig = plt.figure(figsize=(7, 7), facecolor='#000000')
                        ax = fig.add_subplot(111, projection='3d')
                        draw_sphere(ax, f"Qubit  ·  Step {i}", b0, '#bf40ff')
                    else:
                        fig = plt.figure(figsize=(14, 7), facecolor='#000000')
                        ax1 = fig.add_subplot(121, projection='3d')
                        ax2 = fig.add_subplot(122, projection='3d')
                        draw_sphere(ax1, f"Qubit 0  ·  Step {i}", b0, '#bf40ff')
                        draw_sphere(ax2, f"Qubit 1  ·  Step {i}", b1, '#00e5ff')
                    fig.suptitle(title, fontsize=15,
                                 fontweight='bold', color='#ffffff', y=0.95)
                    fig.patch.set_facecolor('#000000')
                    plt.tight_layout(pad=2.0)

                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', dpi=110,
                                facecolor='#000000')
                    plt.close(fig)
                    buf.seek(0)
                    frames_b64.append(base64.b64encode(buf.read()).decode())
                    progress.progress((i + 1) / n_frames,
                                      text=f"Rendered frame {i+1}/{n_frames}")

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
                st.components.v1.html(anim_html, height=800)

            except Exception as e:
                st.error(f"Animation Error: {e}")

    # ─────────────────────────────────────────────
    # OUTPUT BOX
    # ─────────────────────────────────────────────
    if st.session_state.output:
        st.markdown('<div class="section-title">📤 Output</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="output-box">{st.session_state.output}</div>',
            unsafe_allow_html=True
        )
