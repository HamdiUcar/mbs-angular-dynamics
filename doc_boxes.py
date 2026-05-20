#print("Loading doc_boxes...")
from   typing import Callable, Optional, Literal, Tuple, Dict, Any
from   IPython.display import display, clear_output, Latex, Math,  Markdown, HTML, Javascript
from   ipywidgets import  Label, Button, ToggleButtons, Text, Checkbox, Dropdown, Select, HBox, VBox, GridBox, Accordion, HTML as iHTML, HTMLMath, Layout, Output

import ipywidgets as W
import numpy as np
import sympy as sp
import math
from   mpmath import findroot,mpc
from   sympy import Eq, latex, simplify, linear_eq_to_matrix
from   sympy import Add, Mul, sin as sin, cos as cos, tan as tan, asin as asin
import re
from   eom_diag import EOMDiag, equations_in_latex, dotify_latex
import traceback

# —————————————————————————————————————————————————————————————————————————————————
pat_trig_remove_parantheses = re.compile(
#    r'\\(sin|cos|tan|csc|sec|cot|arcsin|arccos|arctan|sinh|cosh|tanh)'
    r'\\(sin|cos|tan|sin\^2|sin\^3|sin\^4|sin\^5|sin\^6|cos\^2|cos\^3|cos\^4|cos\^5|cos\^6)'
    r'\s*(?:\{\s*\\left\(\s*([^\+\-\*/\^,=()]+?)\s*\\right\)\s*\}'
    r'|\{\s*([^+\-\*/\^,=()]+?)\s*\}'
    r'|\(\s*([^\+\-\*/\^,=()]+?)\s*\))')
#def repl(m):
#    name = m.group(1)
#    inner = m.group(2) or m.group(3) or m.group(4) or ''
#    return f'\\{name}{inner}'

def repl(m):
    name = m.group(1)
    inner = m.group(2) or m.group(3) or m.group(4) or ''
    s_full = m.string
    k = m.end()
    # advance past whitespace
    while k < len(s_full) and s_full[k].isspace():
        k += 1

    add = ''
    # add '\,' only when there is a next token, it's not '\right', and not a punctuation/operator
    if k < len(s_full) and not s_full.startswith(r'\right', k) and s_full[k] not in ')]},.;+-*/^=':
        add = r'\;'

    return f'\\{name}{inner}{add}'

# remove paranteses from trigonometric terms
def rem_par(s:str) -> str:
    #return s
    return pat_trig_remove_parantheses.sub(repl,s)

# —————————————————————————————————————————————————————————————————————————————————
def reordered_vector_latex(name, em, vec):
    """
    Display `name = vec` as a pmatrix, reordering each entry
    so (ω*t) factors come before γ-factors, preserving that order in LaTeX,
    and omitting the leading `1*` when the coefficient is +1.
    """
    rows = []
    for i in range(vec.rows):
        e = vec[i, 0]
        # 1) extract coeff and factors
        coeff, factors = e.as_coeff_mul()

        # 2) sort factors by (ω*t) first, then γ, then others
        def key(f):
            syms = f.free_symbols
            if em.t in syms or em.omega in syms: return 0
            if em.gamma in syms:              return 1
            return 2
        ordered = sorted(factors, key=key)
        # 3) rebuild without the +1 coefficient
        if coeff == 1:
            e_ord = sp.Mul(*ordered, evaluate=False)
        else:
            e_ord = sp.Mul(coeff, *ordered, evaluate=False)
        # 4) render to LaTeX without reordering
        s = latex(e_ord, order='none')
        s = s.replace(r'\left(t \right)', '') #\left(t \right)
        rows.append(s)
        #print (s)
        #rows.append(latex(e_ord, order='none'))

    # 5) assemble pmatrix
    body = r" \\ ".join(rows)
    if name:
        return rf"{name} = \begin{{pmatrix}}{body}\end{{pmatrix}}"
    else:
        return rf"\begin{{pmatrix}}{body}\end{{pmatrix}}"

# —————————————————————————————————————————————————————————————————————————————————
class LazyAccordion:
    """
    Attach to an existing Accordion instance and render heavy content on-demand.
    Designed to work with an Accordion that already has a child Output widget.
    Usage:
        # existing top-level code stays the same:
        guide_box = Output(...)
        accor_guide = Accordion(children=[guide_box], ...)
        accor_guide.set_title(0, '📖 Quick Guide')
        qgc = QuickGuideAccordion(accor_guide)
    """

    def __init__(self, accor_instance, child_index=0):
        self.accordion = accor_instance
        self.idx = int(child_index)
        self._rendered = False

        # ensure children is mutable list
        children = list(self.accordion.children)

        # make sure the requested child exists
        if self.idx < 0 or self.idx >= len(children):
            raise IndexError(f"Accordion has {len(children)} children; requested index {self.idx}")

        # ensure the target child is an Output widget (wrap if necessary)
        if not isinstance(children[self.idx], Output):
            old_child = children[self.idx]
            out = Output()
            # put the old child inside the new Output for display
            with out:
                display(old_child)
            children[self.idx] = out
            self.accordion.children = tuple(children)
            self.output = out
        else:
            self.output = children[self.idx]

        # observe open events
        self.accordion.observe(self._on_selected, names='selected_index')

        # if the accordion is already open at this index, render immediately
        try:
            if getattr(self.accordion, 'selected_index', None) == self.idx:
                self._do_render()
        except Exception:
            # swallow errors during init and let user open to trigger rendering
            pass

    def _on_selected(self, change):
        # change['new'] is the active index (or None)
        if change.get('new') == self.idx and not self._rendered:
            self._do_render()

    def _do_render(self):
        try:
            # call derived class implementation
            self.render_content()
            self._rendered = True
        except Exception:
            # if rendering fails, show traceback inside the Output for debugging
            with self.output:
                print("Error while rendering lazy accordion content:")
                traceback.print_exc()

    def render_content(self):
        """Override this in derived classes to populate `self.output` (an Output widget).
           Use `with self.output:` and normal IPython display calls.
        """
        raise NotImplementedError("Derived classes must implement render_content()")

    def force_render(self, clear_before=False):
        """Force rendering even if already rendered. Useful for debugging."""
        if clear_before:
            with self.output:
                self.output.clear_output(wait=True)
        self._do_render()

# —————————————————————————————————————————————————————————————————————————————————
class ResearchContext(LazyAccordion):
    def __init__(self, accor_instance, child_index=0):
        super().__init__(accor_instance, child_index=child_index)

    def render_content(self):
        # this method will run once when the accordion is opened (or when created if already open)
        with self.output:
                display(Markdown(r"""
<style>
  .sb {font-weight: 600;}
  .sb2 {font-weight: 600; margin-top:2em; margin-bottom:2em;}

  /* target the table inside the wrapper div produced by the Markdown below */
  #eom_vars_table_wrapper table tr { background-color: var(--jp-layout-color1) !important; }
  /* remove shading applied by JupyterLab to code blocks inside table */
  #eom_vars_table_wrapper table code { background-color: var(--jp-layout-color1) !important; border: none !important; font-weight: 600}
  #eom_vars_table_wrapper table td, #eom_vars_table_wrapper table th {
      padding: 2px 6px !important;
      line-height: 1.1 !important;
  }
</style>
<!-- Angular Dynamics of Magnetically Bounded Bodies with Rotating Fields -->

This application is a symbolic, numerical, and visual analysis framework for studying the angular dynamics of free or magnetically bounded bodies subjected to rotating magnetic fields. Magnetic bounding by rotating fields is a relatively new phenomenon in which a rigid body with a magnetic moment can be trapped at a precise location by a rotating magnetic field with a gradient, without requiring a local minimum of magnetic potential. The effect is essentially realizable with dipole magnets. In this interaction, the body couples to the field through its magnetic moment, which is fixed in the body frame, and may undergo translational and angular oscillations of typically small amplitude. These oscillations allow the body to find a quasi-stable equilibrium with the driving field and the confinement mechanism which impossible within magnetostatic except using diamagnetism.

This work focuses on the angular part of the dynamics, which is responsible for keeping the body in proper alignment with the rotating field. In a basic model, the angular oscillation consists of a symmetric conical motion of the body's magnetic moment around the axis of the rotating field syncronized with this field. This alignment meschanism can be associated with the phase-lag property of driven oscillators; in this case the phase is held close to 𝜋, causing the body’s magnetic moment to be antiparallel to the field in the azimuthal components, while parallel alignment occurs in the zenithal components of the magnetic moment and the rotating field. 

Since antiparallel alignment results in repulsion and parallel in attraction, the stability in translational degrees of freedom is obtained by the system ability to adjust these opposing effects by variation of the zenithal angle of the body -- which, in turn, corresponds to the amplitude of this driven oscillation as a function of the strength of magnetic torque acting on the body.

As the body moves closer in the field gradient, the stronger field increases the magnetic torque, causing the oscillation amplitude to increase. This corresponds to a larger zenithal angle and makes the antiparallel alignment more pronounced. Consequently, the repulsive component grows faster than the attractive component, producing an effective positive translational stiffness.

A related self-regulating mechanism can also arise from translational oscillations, especially in the radial plane with respect to the rotation axis of the driving field in small circles syncronized by the rotating field. Through the phase-lag response of the driven motion, the displacement of the dipole can become nearly opposite in phase to the effective attractive force. This shifts the dipole (therefore the body) in the field gradient in a way that favors the repulsive component of the magnetic interaction, so that the attractive force is effectively used against itself. The result is an effective positive translational stiffness. Although this displacement may be small, it can still be sufficient to produce a bound state in cases where angular oscillations are constrainted or suppressed.

In the model implemented here, the angular degrees of freedom are described by the three Euler angles 𝜃, 𝜙, and 𝜓. The angle 𝜓 denotes rotation of the body about its magnetic axis, while 𝜙 and 𝜃 describe the azimuthal and zenithal orientation of the magnetic moment in a coordinate system whose z-axis is the rotation axis of the driving field. The magnetic moment is assumed to be fixed in the body frame, while the applied magnetic field rotates with user-defined angular velocity, field inclination, and strength.

The equations of motion are derived using the Euler–Lagrange formalism in the Euler-angle coordinates (𝜃, 𝜙, 𝜓), for a rigid body with a diagonal moment-of-inertia tensor whose one principal axis coincides with the magnetic axis. Axisymmetric and isotropic cases are also covered as reductions of the general model.

For an axisymmetric body, the equations of motion admit synchronized steady-state solutions in which the vectors entering the dynamics remain time-invariant in the reference frame of the rotating magnetic field. In this rotating-field frame, the explicit time dependence cancels and the equations of motion become autonomous. These equilibria, supported by numerical simulations and experimental observations, provide the operating points for local linearization. The equations of motion are then expanded to first order about the equilibrium state defined by the generalized coordinates and their first derivatives, yielding the state-matrix representation used for eigenvalue analysis.

The eigenvalue analysis exposes the local stability characteristics and natural angular frequencies of the synchronized solution. In the absence of damping, the system can be marginally stable, with eigenvalues whose real parts are zero. In the presence of damping, stable solutions become asymptotically stable when the real parts of all eigenvalues are negative. When the field inclination angle γ is reduced below the required stability threshold, one real part becomes positive, clearly indicating an unstable regime.

The simulations and eigenvalue results support each other in identifying stability limits and reproducing accurately the observed eigenfrequencies. The imaginary parts of the eigenvalues give the natural angular frequencies of the synchronized solution. In the stable regime, two principal eigenfrequencies are typically observed close to the driving frequency; for small oscillation amplitudes, one lies slightly below the driving frequency and the other slightly above it. The separation of these frequency branches is mainly controlled by the field inclination angle γ, while increasing oscillation amplitude tends to shift both frequencies downward. This downward shift can be interpreted as a softening-type nonlinear effect. Through parameter sweeps, the utility makes it possible to observe frequency branching, stability boundaries, damping effects, and the influence of inertia ratios.

Simulations further indicate the existence of excited bounded modes within the stable regime. In these modes, the zenithal angle of the magnetic moment can vary over a large range, and the corresponding state-space trajectories may form cycloidal, rosette-like, looped, or otherwise structured patterns. In damped cases, these trajectories can converge toward attractors or preferred bounded regions in state space, indicating stable excited forms of motion in addition to the basic synchronized equilibrium. The built-in state-variable plots, phase portraits, and polar plots provide a practical way to examine these modes and compare them with regular synchronized motion, beat formation, nonlinear amplitude limitation, and the onset of instability.

Section C analyzes the behavior of the state variables in the presence of damping. In particular, it is shown that, in the absence of electromagnetically induced spin-up torques, 𝜓̇ converges to −𝜔 cos 𝜃. This condition corresponds to a zero z-component of the body’s angular velocity in local coordinates and provides a useful reference state for setting up damped simulations.

Direct nonlinear simulations and eigenvalue results support each other in identifying stability limits and reproducing the observed eigenfrequencies. Simulations also reveal an angular instability mechanism when the body has a small non-axisymmetry, which is often encountered in experiments. In such cases, the nonlinear dynamics of the system can use this asymmetry to promote resonant angular modes.

Section E reformulates Ucar (2021)’s semi-empirical stability formula in terms of the driving-field elevation angle γ. The formula is applicable to isotropic bodies under the condition 𝜓̇ = −𝜔 and is found to predict stability limits in agreement with both eigenvalue analyses and direct simulations.

A central purpose of the application is to support the investigation of magnetic bound-state dynamics beyond simple numerical simulation. The symbolic derivation of the equations of motion allows the assumptions, coordinate definitions, and model reductions to be inspected directly. The numerical solver enables time-domain simulations, while the animation, plotting, and phase-portrait tools help visualize the physical motion and the evolution of state variables.

The application also includes linearization and eigenvalue analysis tools in Section D. These are useful for studying local stability around synchronized operating states, identifying natural angular frequencies, and comparing predicted small-signal behavior with frequencies observed in full nonlinear simulations. By sweeping parameters such as field rotation rate, field inclination, damping, magnetic torque strength, inertia ratios, and initial conditions, the user can observe how stable, unstable, quasi-synchronized, and transitional regimes emerge.

Therefore, this utility can be used as both a research notebook and an exploratory computational laboratory for angular magnetic dynamics. It may be useful for studying magnetically bounded motion, rotating-field stabilization, resonance-like behavior, damping-dependent equilibrium states, non-axisymmetric instability mechanisms, and the relationship between nonlinear simulations and linearized eigenvalue spectra.

"""))

            
# —————————————————————————————————————————————————————————————————————————————————
class QuickGuideAccordion(LazyAccordion):
    def __init__(self, accor_instance, child_index=0):
        super().__init__(accor_instance, child_index=child_index)

    def render_content(self):
        # this method will run once when the accordion is opened (or when created if already open)
        with self.output:
                display(Markdown(r"""
<style>
  .sb {font-weight: 600;}
  .sb2 {font-weight: 600; margin-top:2em; margin-bottom:2em;}

  /* target the table inside the wrapper div produced by the Markdown below */
  #eom_vars_table_wrapper table tr { background-color: var(--jp-layout-color1) !important; }
  /* remove shading applied by JupyterLab to code blocks inside table */
  #eom_vars_table_wrapper table code { background-color: var(--jp-layout-color1) !important; border: none !important; font-weight: 600}
  #eom_vars_table_wrapper table td, #eom_vars_table_wrapper table th {
      padding: 2px 6px !important;
      line-height: 1.1 !important;
  }
</style>

### 📘 Quick Guide: Using This Utility

**About**
<br>
This is a customized rigid-body simulation and numerical analysis framework for studying angular motion of a body with a magnetic dipole moment and subjected to an external
magnetic field. The body is assumed, in the general case, to have a diagonal moment‐of‐inertia (MOI) tensor, with its magnetic moment aligned along the body’s $z$‑axis and
centered at the center of mass. The model supports an external magnetic field of constant strength that can rotate about the $z$-axis with an elevation angle $\gamma$ about
the $xy$-plane. Using this scheme, the utility can simulate the angular dynamics of a magnetic body under the magnetic bound-state condition in classical physics
(https://doi.org/10.3390/sym13030442) with user-defined parameters. The results can be visualized through animations, and plots that depict the time-evolution of relevant
system variables, together with additional figures.<br>

The utility includes derivation of the equations of motion (<span class='sb'>Section A</span>) starting from three Euler rotations and applying the Euler–Lagrange formalism,
implemented using the SymPy package. Although the system is externally driven and therefore non-conservative, the Euler–Lagrange approach yields the correct equations of
motion because the magnetic torques can be expressed as the spatial derivative of the magnetic potential $V(q,t) = -\vec m\cdot \vec B(q,t)$ where $q$ denotes the generalized
coordinates. Viscous damping is incorporated separately as an external generalized force (torque). All derivations are generated in the <span style="font-family:monospace;
font-weight:600; font-size:inherit;">EOMDiag</span> class, a Python module within this project (see the [member list](#eom-var-list) below). Additionally variables which may
be useful for further evaluations are also listed there.

<span class='sb'>Section B</span> summarizes the equations of motion for bodies with diagonal MOI tensor, and with axisymmetry and isotropy as special cases, also provides
basic vectors used in derivations.

<span class='sb'>Section C</span> covers the characteristics of the state variables in the presence of damping, which are particularly useful for setting up equilibrium
initial conditions in simulations.

<span class='sb'>Section D</span> presents the linearization of equations of motion, Jacobians, state-matrix representation and eigenvalue analysis. Using this section,
one can calculate and plot eigenvalues of a solution with respect to a user-selected parameter. This way, the stability of a solution and eigenfrequencies can be visualized.

<span class='sb'>Section E</span> covers the reformulation and simplification of semi-empirical stability criterion originally proposed by Ucar (2021), expressing it in
terms of the 'tilt' angle $\gamma$ of the rotating field.


### Fast Start
1. **Page Layout**

   Adjust the browser window width so that the interfac fills the page without horizontal scrollbars. The page normally fits within a 1315-pixel width. One can
   zoom [+][-] from browser control to see the page comfortably.

2. **Choose a preset**

   First-time users may run the default configuration provided in startup or select one of the predefined presets. Each configuration specifies all physical parameters
   ( $I_1,\;I_2,\;I_3,\;mB,\;\omega_B,\;\xi$ ) as well as the initial values of the state variables ( $\theta_0, \; \phi_0, \; \psi_0, \; \dot{\theta}_0, \; \dot{\phi}_0,
   \; \dot{\psi}_0$ ) which enter the equations of motion for a body with a diagonal MOI tensor.

3. **Simulate**

   Click on <span class='sb'>▶Simulate</span> to generate time-evolution data belong to angular motion of the body, covering angular positions and velocities for each
   simulation step. In the default simulation settings, steps are set as 2500 steps per second and simulation length is 10 seconds. Step count can be increased to
   aacomodate to higher speeds of the rotating field. This default setup simulates angular motion of a magnetic body with isotropic moment-of-inertia tensor with default
   parameters, which can be visualized as a cube, a sphere or a cylinder with appropriate geometry from the scene view. This step may take several seconds to complete.
   Note that parameters $I_1, \; I_2, \; I_3$ must satisfy $I_1, I_2 < I_3$ in order to represent a physically realizable body.

4. **View plots**

   Expand the <span class='sb'>Plots</span> panel to access graphical representations of the simulation results. By default, only the $\theta(t)$ plot is enabled. Additional
   time evaluation and phase portrait plots can be displayed using the checkboxes at the top of the panel. Displaying a new plot may take a few seconds. If initial conditions
   are automatically set for symmetric and stable motion of the body, time evaluation plots may only show residual variations of values, which can be up to 1e-6 for velocity
   figures and 1e-10 for others. 

5. **Generate Eigenvalue Spectrum**

    Expand the <span class='sb'>Linearization and Eigenvalue Analysis</span> panel. Scroll down (or use the shortcut) to the <span class='sb'>Eigenvalue Analysis</span>
    section. Press the <span class='sb'>Plot</span> button. See real and imaginary parts of eigenvalues in the spectrum. System is stable or quasi-stable when all real
    parts are zero or negative. Imaginary parts bifurcate in the stable zone. These plots are zoomable from the toolbar at the left and allow precise zooming on the zone
    boundary using the <span class='sb'>Zoom on Bifurcation</span> button. One can select the sweep parameter and its range to examine how eigenvalues vary with that
    parameter.

6. **Scene setup**

   Click the <span class='sb'>Scene</span> button, which opens a 3D animation scene at the bottom of the page. Select the shape, the scale and the view angle. View angle can
   be changed either by mouse dragging (hold the left button and move) or by pressing coordinate buttons (each button cycles through four view angles on repeated presses).

7. **Animate**

   Enter <span class='sb'>0.2</span> into <span class='sb'>Span</span> field and <span class='sb'>10</span> into <span class='sb'>Length</span> field, then click
   <span class='sb'>▶Animate</span>. A progress bar is shown; generation typically completes in under 30 seconds. Upon completion, an embedded video player appears at the
   bottom of the page.

8. **Download or inspect**

   Generated animation files are stored in the directories 'RESULTS' or 'RESULTS/TEMP' and may be played back or downloaded from JupyterLab interface.

9. **Explore derivation and characteristics of the model**

   - See sections **A** and **B** which expose the physics model used in generation of simulation data.
   - See sections **C, D** and **E** where the stability and characteristics of the dynamics are evaluated.<br><br>

#### Step 1: Set Parameters
1. In the <span class='sb'>Setup Parameters</span> column enter:
      - $\omega_B$ : Angular velocity of the external field $\vec{B}$
      - $I_1$, $I_2$, $I_3$ : Principal moments of inertia
      - $mB$ : Product of the body’s magnetic moment $\vec{m}$ magnitude and field strength of external field  $\vec{B}$
      - $\gamma$ : Tilt angle of the $\vec{B}$ from $xy$-plane (complement of the zenith angle or elevation angle)<br>
        When $\gamma = 0$, $\vec{B}$ lies entirely in $xy$-plane. However as small tilt $\gamma_\min$ is required for a symmetric solution which generates a $z$-component
        of the field and forces the body's magnetic moment be aligned with it. It is found that there is a minimum $\gamma$ angle ensuring the angular motion of the body
        symmetric regarding $z$-axis and can be calculated exactly for an isotropic body $(I_1$ = $I_2$ = $I_3)$. This formula 
        $\gamma_\min = \sin^{-1}\Bigl(\dfrac{mB}{2 I \omega^2}\Bigr)$ is derived from linearized equation of motion of $\ddot\theta$ around the equilibrium point
        $\theta_{eq}$ from the proposed condition $\left. \dfrac{dF(\theta)}{d\theta}\right|_{\large{\theta=\theta_{eq}}} \le I\,\omega^{2}$ where $F$ is the right side
        of the equation of motion about $\ddot\theta$ in a solution where second derivatives of all variables are zero. This solution admits
        $I\omega^2\sin\theta - mB(\cos(\theta -\gamma)) = 0$ where $\theta$ can be calculated as
        $\theta_{eq} = \tan^{-1}\Bigl(\dfrac{mB \cos\gamma}{I\omega^2 - mB\sin\gamma}\Bigr)$ &nbsp;&nbsp; when &nbsp;
        $\boldsymbol{\phi = \omega t +\pi},$&nbsp;&nbsp;$\boldsymbol{\psi = - \omega t}$.<br>

      - $\gamma_{\Large \epsilon}$ : One may choose a different $\gamma$ from $\gamma_\min$ and the difference can be set or viewed at the this entry.


2. <span class='sb'>Set Initial Angless and Initial Velocitiess</span>

     If the option <span class='sb'>Link parameters for symmetry</span> is checked, utility will try to set initial conditions for an axisymmetric body ($I1 = I2$)
     in the second and third entry columns in order to obtain a solution where the second derivatives of all variables are zero. In such a solution the angle $\theta$
     is set as a positive value less than $\pi/2\; (\,90^\circ\,)$. Beyond that, it can be assumed these parameters do not admit a stable solution. While the equations
     that calculate initial condition for an equilibrium condition are derived from the axisymmetric solution, their results can be used for small non-axisymmetries,
     in order to easily evaluate these cases.

     - Once the parameters in <span class='sb'>Setup Parameters</span> are set, one can modify parameters in <span class='sb'>Initial Angles</span> and
       <span class='sb'>Initial Velocities</span> columns. Setting a parameter in one column may update values in the other column in order to obtain a
       stable solution when the <span class='sb'>link</span> option is checked.

     - There are two buttons, which set the calculation method, right one for setting ($\dot\phi = \omega,\; \dot\psi = -\omega$) which causes
       the body to return to its initial position after one cycle of the rotating field without making a turn. The left button sets the calculation mode to
       ensure $\dot\psi = -\omega \cos\theta$. This option does not satisfy the above condition but sets the z component of the body's angular
       velocity in the body frame as zero ( $\nu_{\large z} = 0$ ). Once this mode is selected, parameters are calculated accordingly.
       
     - It is possible to set initial condition (angles and velocities) using the state variables at a specified time point of a prior or a saved run. This might be useful
       for eliminating spurious oscillations caused by initial values do not satisfy equilibrium conditions when the application cannot provide them directly in the
       defined configuration. To address this, one may first perform a simulation with non-zero damping factor, allowing these oscillations to decay. The state variables
       at a later time when the system is close to equilibrium, can then be extracted and used as the initial conditions for a second simulation. By entering a time value
       in the time field ( $t_0$ ) and pressing the <span class='sb'>'Set From Simul. at 𝑡₀'</span> button, the program selects values of state variables 
       ( $\theta_0,\; \phi_0,\;\psi_0,$ $\dot{\theta}_0,\; \dot{\phi}_0,\; \dot{\psi}_0$ ) from the closest time point. User may set the check box <span class='sb'>％</span>
       to reduce $\phi_0,\; \psi_0$ to the $0 - 2 \pi$ range. After selecting initial value, one may reset $t_0 = 0$, which also defines the  time offset of the simulation
       since time variable begins at $t = t_0$.<br>
       
 
3. <span class='sb'>Set Damping & Test Parameterss</span>
     - <span class='sb'>Damping</span> parameter can be set within $\xi$ entry. It is defined as acceleration per velocity in angular terms and its unit is $s^{-1}$.
       It is converted to a torque factor by multiplication with the moment-of-inertia. As explained in Box C, damping forces the body's z-component of angular
       velocity in local coordinates to zero. This corresponds to the first time derivative of the Euler angle $\psi$ converging to the value $-\omega\,\cos(\theta)$.
       If one needs to keep $\dot\psi$ constant at a specified value, they can press the button [External $\ddot\psi$], to have the program calculate and apply an
       external torque in simulation to keep this angular velocity. One can also set this external torque factor manually at the $\ddot\psi$ entry. This parameter is
       also an acceleration term and converted to torque by multiplying with the body's moment-of-inertia.<br><br>

#### Step 2: Generate Simulation Results

- Click <span class='sb'>▶Simulate</span> to generate time evaluation of rotation variables and their first derivatives. Calculated values (simulation results)
  can be saved in the file system as CSV file if defined parameters (associated to an entry defined as Notes) are saved in the system by pressing 'Save' button.
  Simulation results are automatically saved once the Notes entry is saved. User is asked to add new result, replace the previous result, or not to save results.<br><br>


#### Step 3: Inspect Results
- Open the <span class='sb'>Plot</span> section to inspect a set of graphical representations of the simulation results, presented in the time domain and in phase space.
  The available plots are:
   - $\theta(t)$ and its frequency spectrum
   - Phase of $\phi(t)$ w.r.t. $B(t)$
   - $\dot{\phi}(t)$
   - $\dot{\psi}(t)$
   - ${\nu_z}^w, {\nu_z}^b$ (Z component of angular velocity vector in world and in body frames)
   - Kinetic energy $T(t)$ related to angular motion
   - Potential energy $V(t)$ related to magnetic torque
   - Total energy $T(t) + V(t)$
   - Polar plot of ($\theta$, $\phi$)
   - Spherical plot of ($\theta$, $\phi$) as a projection on a plane
   - Phase portrait of several entities as $(\theta,\,\dot\theta), \; (\phi -\omega t, \, \dot\phi), \; (\psi,\, \dot\psi),\; (\theta, \, T) $ - kinetic$,\; (\theta, \, V) $ - potential$,\; (\theta, \, H) $ - total.
   <br>

- Notes:
   - The magnetic potential can vary between negative and positive values. The zero value corresponds $\vec{m} ⊥ \vec{B}$,
     minimum value corresponds to parallel and maximum to antiparallel alignments.

   - Each plot has left side toolbar allowing one to zoom to a specific part of the plot and to navigate. Some plots have measurement boxes at the bottom showing the
     average and peak to peak amplitude of the signal. These figures can be copied to clipboard by clicking on them.

   - The Sync button at the top right of the plots allows to set the time range of the plot to the time range last zoomed plot. Clicking again the button revert the
     action.<br><br>


#### Step 4: Generate Animation
1. Click <span class='sb'>Scene</span> button to open the visual setup of the animation scene. This includes basic axisymmetric shapes that one can choose, their scales
and set the view angle of the scene. Additional rendering parameters can be defined in the <span class='sb'>Option</span> box on the form. The proportion of the body
(i.e. radius to height ratio of a cylinder) is automatically calculated respecting $I_1 / I_3$ ratio, moment-of-inertia tensor components.

2. Click <span class='sb'>▶Animate</span> to generate an mp4 video of the animation.

---
<a id="eom-var-list"></a>

### Class EOMDiag member variables

Class variables which can be used for symbolic operations. The instance of the class is defined as
<span style="font-family:monospace; font-weight:600;">em</span>.<br>
<div id="eom_vars_table_wrapper" style="display:inline-block;">

|Variable|Description|
|:------------|:----------|
| `t`         | Time                                                                              |
| `I1`        | Component $I_{11}$ of a diagonal moment-of-inertia tensor                         |
| `I2`        | Component $I_{22}$ of a diagonal moment-of-inertia tensor                         |
| `I3`        | Component $I_{33}$ of a diagonal moment-of-inertia tensor                         |
| `I`         | Isotropic moment-of-inertia scalar                                                |
| `omega`     | Angular velocity $\omega$ of the external magnetic field $\vec{B}$                |
| `mB`        | $m\!B$ — product of the magnitudes of body's magnetic moment $\vec{m}$ and field $\vec{B}$ |
| `gamma`     | Tilt (elevation) angle $\gamma$ of the field $\vec{B}$                            |
| `xi`        | Viscous damping constant $\xi$                                                    |
| `th`        | Euler rotation angle $\theta$ (generalized coordinate)                            |
| `ph`        | Euler rotation angle $\phi$ (generalized coordinate)                              |
| `ps`        | Euler rotation angle $\psi$ (generalized coordinate)                              |
| `th_d`      | $\dot\theta$ — first time derivative of $\theta$ (generalized velocity)           |
| `ph_d`      | $\dot\phi$ — first time derivative of $\phi$  (generalized velocity)              |
| `ps_d`      | $\dot\psi$ — first derivative of $\psi$  (generalized velocity)                   |
| `th_dd`     | $\ddot\theta$ — second time derivative of $\theta$ (generalized acceleration)     |
| `ph_dd`     | $\ddot\phi$ — second time derivative of $\phi$ (generalized acceleration)         |
| `ps_dd`     | $\ddot\psi$ — second time derivative of $\psi$ (generalized acceleration)         |
| `R`         | Euler rotation matrix $R$ (ZXZ convention)                                        |
| `nu_bo`     | Body's angular velocity vector $\vec{\nu}_{bo}$ expressed in body-frame           |
| `nu_wo`     | Body's angular velocity vector $\vec{\nu}_{wo}$ expressed in world-frame          |
| `um`        | Unit vector $\hat{m}$ of the body's magnetic moment                               |
| `uB`        | Unit vector $\hat{B}$ of the external magnetic field                              |
| `I_rot`     | Body's moment-of-inertia tensor $I_{rot}$ in world-frame (time dependent)         |
| `T`         | Body's rotational kinetic energy $T$                                              |
| `V`         | Body's rotational potential energy $V$                                            |
| `eom`       | Equations of motion for a body with diagonal MOI. Indexable as eom[n], n=0,1,2.   |
| `eom_axisym`| Equations of motion for axisymmetric body (indexable similarly)                   |
| `eom_iso`   | Equations of motion for a body with isotropic MOI (indexable similarly)           |
| `f_th(…)`   | RHS function returning $\ddot\theta$ (Euler equation)                             |
| `f_ph(…)`   | RHS function returning  $\ddot\phi$  (Euler equation)                             |
| `f_ps(…)`   | RHS function returning  $\ddot\psi$  (Euler equation)                             |
</div><br>


### Variables accesible from other cells
<div id="eom_vars_table_wrapper" style="display:inline-block;">

|Variable|Description|
|:------------|:----------|
| `t_vals`    | Array holding time points of a simulation steps. Indexable as `t_vals[j]` where `j` is a step.              |
| `y_vals`    | Array holding state variables of a simulation. Indexable as `y_vals[i, j]` where `j` is the step index and  |
|             | `i` is the variable index where 0 = $\theta$, 1 = $\phi$, 2 = $\psi$, 3 = $\dot\theta$, 4 = $\dot\phi$, 5 = $\dot\psi$ |
| `JEm`       | Jacobian matrix (6x3) for axisymmetric bodies. |
| `JEmIso`    | Jacobian matrix (6x3) for isotropic bodies. |
| `eigvals_all`| Array of Eigenvalues within a spectrum |
| `eigvecs_all`| Array of Eigenvectors within a spectrum |
</div>
"""))
# <div id="eom_vars_table_wrapper" style="border-bottom:1px solid grey; display:inline-block;">

# —————————————————————————————————————————————————————————————————————————————————
class SupBoxAccordion(LazyAccordion):
    def __init__(self, accor_instance, _em:EOMDiag, eq_param_widgets, child_index=0):
        super().__init__(accor_instance, child_index=child_index)
        self.em = _em
        self.eqpw = eq_param_widgets
        self.calc_phi0_displayed = False

    def calculate_phi0(self, b):
        em = self.em
        (w_om, w_I1, w_I2, w_mB, w_gam, w_th_eq, w_xi) = self.eqpw
        with self.output:
            if self.calc_phi0_displayed:
                # Remove the last output (the previous calculation)
               self.output.outputs = self.output.outputs[:-1]
            #th_cos = calc_theta_for_psi_cos_omega_sub()
            sin_phi0 = float(self.eq_phi_phase.subs({
                em.xi:    w_xi.value * w_I1.value,
                em.th:    w_th_eq.valRad,
                em.omega: w_om.valRad,
                em.mB:    w_mB.value,
                em.gamma: w_gam.valRad}))
            phi = np.pi - np.asin(sin_phi0)
            display(Math(rf"(16)\qquad {rem_par(dotify_latex(Eq(self.phi0, sp.pi - sin_phi0)))} \; = \; {phi:.8f}\: \text{{rad}} \; = \;{phi*180/np.pi:.6f}\: \text{{deg}}"))
            self.calc_phi0_displayed = True


    def render_content(self):
        em = self.em
        # this method will run once when the accordion is opened (or when created if already open)
        with self.output:
            display(Markdown(r"### 1. Finding the equilibrium value of $\dot\psi$ under damping ($\xi \neq 0$) for an axisymmetric body"))
            display(Markdown(r"""<style>
.sb {font-weight: 600;}
.sb2 {font-weight: 600; margin-top:1em; margin-bottom:1em;}
</style>
<div class='sb' style='max-width:1020px; margin-top: 2em;'>
A floating body in the presence of damping, may experience a drag torque that acts on the rotation (spin) about the symmetry (magnetic) axis and can stop that spin if present. This spin corresponds to the z-component of angular velocity $\nu_z$ expressed in the body’s coordinates. In this section, the equilibrium value of state variable
$\dot\psi$ is derived from the equations of motion for an axisymmetric body. It should be noted that a body having zero $\nu_z$ in the body’s coordinates can still rotate
in lab frame which can be seen by tracking a visible marking on the body and measured by a tachometer. This rotation vanishes when $\dot\phi = \omega$ and $\dot\psi = -\omega$,
also when $\dot\phi \neq \omega$ and $\dot\psi \approx -\dot\phi$ with a small adjustment.</div>
"""))
            display(Markdown(r"<div class='sb2'>Equation of motion  about $\ddot\theta$ :</div>"))
            x0 = em.eom_axisym[0]
            display(Math(r"(1) \qquad " + rem_par(dotify_latex(x0)) + "\\,."))

            display(Markdown(r"<div class='sb2'>Equation of motion about $\ddot\phi $ :</div>"))
            x1 = em.eom_axisym[1]
            display(Math(r"(2) \qquad " + rem_par(dotify_latex(x1)) + "\\,."))

            display(Markdown("<div class='sb2'>Equation of motion about $\\ddot\\psi $ :</div>"))
            x2 = em.eom_axisym[2]
            display(Math(r"(3) \qquad " + rem_par(dotify_latex(x2)) + "\\,."))

            display(Markdown("<div class='sb2'>Equilibrium condition admits $\\quad \\ddot\\theta = 0, \\quad \\ddot\\phi = 0, \\quad \\ddot\\psi = 0, \\quad \\dot\\theta = 0$</div>"))

            display(Markdown("<b class='sb2'>Equation of motion (2) of $\\ddot\\phi$ under equilibrium condition</b>"))
            x1 = x1.subs({em.ph_dd:0, em.th_d:0})
            display(Math(r"(4) \qquad " + rem_par(dotify_latex(x1)) + "\\,."))
            display(Markdown("<div class='sb2'>This yields:</div>"))
            s1 = em.xi*sin(em.th)*em.ph_d
            s2 = em.mB*sin(em.omega*em.t-em.ph)*cos(em.gamma)
            eq5 = Eq(s2, s1)
            display(Math(r"(5) \qquad " + rem_par(dotify_latex(eq5)) + "\\,."))

            display(Markdown(r"<div class='sb2'>Equation of motion (3) of $\ddot\psi$ under equilibrium condition</div>"))
            x2 = x2.subs({em.ps_dd:0, em.th_d:0})
            display(Math(r"(6) \qquad " + rem_par(dotify_latex(x2)) + "\\,."))

            x2s = x2.subs(s2, s1)
            display(Markdown("<div class='sb2'>Substitute (5) in (6)</div>"))
            display(Math(r"(7) \qquad " + rem_par(dotify_latex(x2s)) + "\\,."))
            display(Markdown("<div class='sb2'>Simplify</div>"))
            display(Math(r"(8) \qquad " + rem_par(dotify_latex(sp.simplify(x2s))) + "\\,."))
            display(Markdown(r"""
<div class='sb2' style='margin-top:0.8em;'>Eq.8 is satisfied when $\; \dot\psi = - \cos(\theta)\dot\phi\,.$</div>
<div class='sb'>Note: Any initial velocity $\dot\phi$ converges to $-\cos(\theta)\dot\phi \;$ in the presence of damping.</div>
<hr style='width:1020px; margin:2em auto 3em 0; height:1px; border:none;'> 

### 2. Finding the equilibrium value of the phase of $\theta$ rotation under damping ($\xi \neq 0$) for an axisymmetric body

<div class='sb2'>Equation of motion about $\ddot\theta$ for an axisymmetric body by substitution of $\dot\psi \,$ by $\, -\dot\phi \cos\theta$ reads</div>
"""))
            x01 = x0.subs(em.ps_d,-cos(em.th)*em.ph_d)
            #x01 = push_minus_inside(x01.rhs.collect(cos(em.th)*em.ph_d))
            x01 = x01.rhs.collect(cos(em.th)*em.ph_d)
            display(Math(r"(9) \qquad " + rem_par(dotify_latex(Eq(x0.lhs, x01))) + "\\,."))
            display(Markdown(r"""<div class='sb2' style='max-width:1020px;'>
Note that the axial component of moment-of-inertia ($I_3$) vanishes from the equation of motion. This can be expected since z component of angular velocity vector $\nu_z$ in body coordinates becomes zero when $\dot\psi = -\dot\phi \cos \theta $.<br><br>

By applying equilibrium condition ( $\ddot\theta = \dot\theta = 0$ ) this equation becomes</div>
"""))
            eq0 = Eq(x0.lhs, x01).subs({em.th_dd:0, em.th_d:0})
            display(Math(r"(10) \qquad " + rem_par(dotify_latex(eq0)) + "\\,."))
            display(Markdown(r"<div class='sb2'>The equation can be satisfied only when the term $\cos \theta \, \cos \gamma \, \cos(\omega t-\phi)$ becomes constant.</div>"))
            display(Markdown(r"<div class='sb2'>Since we are not interested in the cases $\theta=\pi/2$ or $\gamma=\pi/2$, remaining solution is</div"))
            display(Math(r"(11) \qquad \phi=\omega t + \phi_0 \,.\\[3ex]"))
            self.phi0 = sp.symbols('phi_0')
            eq0 = eq0.subs(em.ph_d,em.omega)
            eq0 = eq0.subs(em.ph, em.omega*em.t + self.phi0)
            display(Math(r"(12) \qquad "+ rem_par(dotify_latex(eq0)) + "\\,."))
            display(Markdown(r"<div class='sb2'>The phase of $\phi$ defined as $\phi_0$ can be derived from (5) by substituting (11) into.</div>"))
            eq51 = sp.simplify(eq5.subs(em.ph, em.omega*em.t + self.phi0))
            display(Math(r"(13) \qquad " + rem_par(dotify_latex(eq51)) + r"\\,.\\[3ex]"))

            self.eq_phi_phase = sp.solve(eq51,sin(self.phi0))[0]
            display(Math(r"(14) \qquad " + rem_par(dotify_latex(Eq(sin(self.phi0), self.eq_phi_phase))) + r"\\,.\\[3ex]"))

            display(Markdown(r"<b class='sb2'>Since $\sin(\phi_0)$ is negative, the angle is equal to $\pi - \phi_0$</b>"))

            display(Math(r"(15) \qquad " + rem_par(dotify_latex(Eq(self.phi0, sp.pi - sp.asin(self.eq_phi_phase)))) + "\\,."))
            btn_calc_phi0 = Button(description='Calculate 𝝓₀ with current parameters', button_style='info', layout=Layout(width='220px', height = '26px', padding='0', margin='1em 0 1em 0.5em'))
            btn_calc_phi0.on_click(self.calculate_phi0)

            display(btn_calc_phi0)

            # keep phi_0 and eq_phi_phase for phi_0 calculation
            del s1, s2, x0, x1, x2, x2s, x01, eq0, eq51

# —————————————————————————————————————————————————————————————————————————————————

def display_eqs(equations):
    # LaTeX versions of the equations
    eqs_latex = equations_in_latex(equations)
    for i, eq in enumerate(eqs_latex):
        display(Math(r'\hspace{3em}'+ rem_par(eq) + ('\\,.' if i == len(eqs_latex) - 1 else '\\,,')))

# —————————————————————————————————————————————————————————————————————————————————
class R_MatrixAccordion(LazyAccordion):
    def __init__(self, accor_instance, _em:EOMDiag, child_index=0):
        super().__init__(accor_instance, child_index=child_index)
        self.em = _em

    def render_content(self):
        # this method will run once when the accordion is opened (or when created if already open)
        em = self.em
        with self.output: # r_matrix_out:
            x = rf"{em.R}"
            x = simplify(x.replace('(t)',''))

            ub_tex =  rem_par(reordered_vector_latex(r"\hat{B}", em, em.uB))
            um_tex = rem_par(reordered_vector_latex(r"\hat{m}", em, em.um))

            nu_bo = re.sub(r'\\frac\{d\}\{d t\}\s*\\(theta|phi|psi)',r'\\dot{\\\1}', reordered_vector_latex(r"\nu_{body}", em, em.nu_bo))
            nu_wo = re.sub(r'\\frac\{d\}\{d t\}\s*\\(theta|phi|psi)',r'\\dot{\\\1}', reordered_vector_latex(r"\nu_{lab}", em, em.nu_wo))

            display(HTML(r"""<div style='font-size:105%; margin-top:2em;'>
$
\text{1. General rotation matrix in ZXZ convention based on Euler angles } \theta, \; \phi, \; \text{and } \psi
$
</div><br><div>
$\hspace{3em} R = R_{\textstyle z}(\phi)\, R_{\textstyle x}(\theta)\, R_{\textstyle z}(\psi) = """ + rem_par(latex(x, mat_delim='(')) + "\\,.$</div>"))

            display(HTML(r"""<div style='font-size:105%; margin-top:2em;'>$
\text{2. Unit vectors }(\hat{B},\,\hat{m})\ \text{of the rotating magnetic field and the body's magnetic moment}$
</div>
<div style='display:flex; gap:20px; margin-left:47px; margin-top:1em;'>
<span style='width:380px; display:inline-block;'>$""" + ub_tex + """\\,,$</span>
<span style='width:380px; display:inline-block;'>$""" + um_tex + """\\,.$</span>
</div>"""))

            display(HTML(r"""<div style='margin-left:47px; margin-top:1em;' >$
\text{Here } \omega \text{ denotes the angular velocity of } \hat{B} \text{ around $z$-axis, and }
\gamma \text{ the elevation angle of } \hat{B} \text{ from the xy-plane.}
$</div>
<div style="margin-left:47px;">$
\text{The angle } \psi \text{ does not enter } \hat{m} \text{ because it corresponds to a self-rotation around the axis of } \hat{m}.
$</div>"""))

            display(HTML(r"""<div style='font-size:105%; margin-top:2em;'>$
\text{3. Body's angular velocity vector } \nu \text{ in the lab and in the body's frames}
$</div><br>
<div style='display:flex; gap:20px; margin-left:47px'>
<span style='width:380px; display:inline-block;'>$""" + rem_par(nu_bo) + """\\,,$</span>
<span style='width:380px; display:inline-block;'>$""" + rem_par(nu_wo) + """\\,.$</span>
</div>"""))

            del x, ub_tex, um_tex, nu_bo, nu_wo

            display(HTML(r"""<div style='font-size:105%; margin-top:2em'>
\(\begin{aligned}
\text{4. Equations of motion based on Euler angles } \theta,\; \phi \text{ and } \psi, \text{ where } I_1, \; I_3 \text{ are radial and axial components}\\
\text{of the moment-of-inertia tensor of an axisymmetric body, and } \xi \text{ is the damping coefficient}
\end{aligned}\)
</div>"""))
            display_eqs(em.eom)
            # ——————————————————————————————————————————————————————————————————————
            display(HTML(r"""<br><div style='font-size:105%; margin-top: 2em; margin-bottom: 1em; '>
\(\begin{aligned}
\hspace{0.4em}\text{5. Reduced equations of motion for an axisymmetric body where } I_1 = I_2.\\
\end{aligned}\)
</div>"""))
            display_eqs(em.eom_axisym)
            # ——————————————————————————————————————————————————————————————————————
            display(HTML(r"""<div style='font-size:105%; margin-top: 1em; margin-bottom: 1em;'>
\(\begin{aligned}
\hspace{0.4em}\text{6. Reduced equations of motion for a body with isotropic moment-of-inertia where } I = I_1 = I_2 = I_3 .\\
\end{aligned}\)
</div>"""))
            display_eqs(em.eom_iso)
            # ——————————————————————————————————————————————————————————————————————

# —————————————————————————————————————————————————————————————————————————————————
# DERIVATION OF EQUATIONS OF MOTION
class DerivationAccordion(LazyAccordion):
    def __init__(self, accor_instance, _em:EOMDiag, child_index=0):
        super().__init__(accor_instance, child_index=child_index)
        self.em = _em

    def render_content(self):
        em = self.em
        # this method will run once when the accordion is opened (or when created if already open)
        with self.output: #derivation_out:
            # ——— Coordinates and symbols ———
            display(Markdown("""
<style>.sb {font-weight: 600;}</style>
<div class='sb' style='margin-top: 2em;'>
Equations of angular motion of a body subject a dynamic magnetic torque are calculated in body coordinates as follow.<br>
Equations generated by this flow are used in numeric integration solver in order generate simulation data.</div><br>"""))

            display(Markdown("**1. Symbols and Generalized Coordinates**"))
            display(Math(r"""
  \omega,\,I,\,mB,\,\gamma,\, \xi,\,t \in \mathbb{R}, \quad
  \theta(t),\;\phi(t),\;\psi(t)\,,\;
  \dot\theta = \frac{d\theta}{dt},\;\dot\phi = \frac{d\phi}{dt},\;\dot\psi = \frac{d\psi}{dt}\,.
"""))
            display(Markdown(r"""<style>.sb {{font-weight: 600;}}</style>
<div class='sb'>
Here $\omega$ is the angular velocity of the rotating field, $I$ the diagonal moment-of-inertia tensor with components $(I_1,\, I_2,\, I_3)$,
$m\!B$ is a scalar corrseponding product of strengths of magnetic moment $\vec{m}$ of the body and strength of the rotating field $\vec{B}$,
$\gamma$ denotes the elevation angle the vector $\vec{B}$ makes with its rotation plane ($xy$), also called tilt angle,
$\xi$ the damping coefficient and $t$ the time.
$\theta(t),\;\phi(t),\;\psi(t)$ are the rotations of the body defined in Euler rotation matrix in ZXZ scheme.
</div>"""))

            # ——— Rotation matrix ———
            display(Markdown("<br>**2. ZXZ Rotation Matrix**"))

            display(Math(r"""
R = R_{\textstyle z}(\phi)\; R_{\textstyle x}(\theta)\; R_{\textstyle z}(\psi),\quad
R_{\textstyle z}(\alpha)=\begin{pmatrix}\cos\alpha & -\sin\alpha &0\\
                         \sin\alpha &  \cos\alpha &0\\
                              0     &      0      &1\end{pmatrix},\quad
R_{\textstyle x}(\beta)=\begin{pmatrix}1&0&0\\0&\cos\beta&-\sin\beta\\0&\sin\beta&\cos\beta\end{pmatrix}\,.
"""))

            # ——— Angular velocity ———
            display(Markdown("**3. Angular Velocity Vector in Body Frame**"))
            s = reordered_vector_latex(None, em, em.nu_bo)
            s = re.sub(r'\\frac\{d\}\{d t\}\s*\\(theta|phi|psi)',r'\\dot{\\\1}', s)
            s = rem_par(s)
            display(Math(fr"""
[\nu]_\times \;=\;R^T\,\dot R
\quad\Longrightarrow\quad
\nu = \mathrm{{vee}}(R^T\,\dot R)
= \begin{{pmatrix}}
 (R^T\,\dot R)_{{3,2}}\\
 (R^T\,\dot R)_{{1,3}}\\
 (R^T\,\dot R)_{{2,1}}
\end{{pmatrix}} = {s}\,.
"""))
            del s

            # ——— Dissipation ———
            display(Markdown("**4. Rayleigh Dissipation Function**"))
            nu2 = dotify_latex(sp.simplify(em.nu_bo.dot(em.nu_bo)))
            display(Math(fr"\mathcal{{R}} = \tfrac12 \xi\,\|\nu\|^2 = \tfrac12 \xi\,\Bigl({nu2}\Bigr)\,."))
            del nu2
            display(Markdown("<br>**5. Unit Vector of Rotating Magnetic Field**"))

            display(Math(rem_par(reordered_vector_latex(r"\hat{B}", em, em.uB)) + "\\, ."))

            display(Markdown("<br>**6. Unit Vector of Body's Magnetic Moment**"))

            display(Math(r"\hat{m} = R\begin{pmatrix}0\\0\\1\end{pmatrix} = " + rem_par(reordered_vector_latex("", em, em.um)) + "\\, ."))

              # ——— Inertia tensor ———
            display(Markdown("<br>**7. Moment of Inertia Tensor For This Model**"))
            display(Math(r"\boldsymbol{I} = \begin{pmatrix} I_1 & 0 & 0\\0 & I_2 & 0\\0 & 0 & I_3\end{pmatrix}\, ."))

            # Energies
            display(Markdown("<br>**8. Kinetic and Potential Energy**"))
            display(Math(fr"""
                T = \tfrac12\, \nu \, \boldsymbol{{I}} \, \nu = {rem_par(dotify_latex(em.T))}\,,
            """))
            display(HTML("<br>"))
            display(Math(fr"""
                V = -\,mB\,(\hat m\cdot\hat B) = {rem_par(dotify_latex(em.V))}\, .
            """))

            display(Markdown("<div class='sb' style='margin-top: 14px;'>Kinetic Energy of a Body with Isotropic Moment of Inertia</div>"))
            display(Math(fr"""
                T_{{iso}} = {rem_par(dotify_latex(sp.simplify(em.T.subs({em.I3:em.I, em.I2:em.I, em.I1:em.I}))))}\,.
            """))
            display(HTML("<br>"))

            # ——— Euler–Lagrange with damping ———
            display(Markdown("**9. Euler–Lagrange Equations with Damping**"))
            display(Math(r"""
              \frac{d}{dt}\Bigl(\frac{\partial (T-V)}{\partial \dot q_i}\Bigr)
              - \frac{\partial (T-V)}{\partial q_i}
              \;=\; Q^{(\mathrm{diss})}_i,\quad
              Q^{(\mathrm{diss})}_i = -\,\frac{\partial \mathcal R}{\partial \dot q_i},
              \quad q_i\in\{\theta,\phi,\psi\}.
            """))

            # 6) ——— Substitution & solution ———
            display(Markdown("<br>**10. Solve for the Angular Accelerations**"))
            display(Math(r"""
              \ddot\theta\;\to\;\theta_{dd},\quad
              \ddot\phi\;\to\;\phi_{dd},\quad
              \ddot\psi\;\to\;\psi_{dd},
            """))
            display(Math(r"""
              \begin{cases}
              E_\theta(\theta,\phi,\psi,\dot\cdot,\ddot\cdot)=0,\\
              E_\phi(\dots)=0,\\
              E_\psi(\dots)=0
              \end{cases}
              \;\Longrightarrow\;
              \{\theta_{dd}=f_1(\cdots),\;\phi_{dd}=f_2(\cdots),\;\psi_{dd}=f_3(\cdots)\}.
            """))


# —————————————————————————————————————————————————————————————————————————————————
class SemiEmpAccordion(LazyAccordion):
    def __init__(self, accor_instance, child_index=0):
        super().__init__(accor_instance, child_index=child_index)

    def render_content(self):
        # this method will run once when the accordion is opened (or when created if already open)
        with self.output: #semi_empiric_box:
            display(Markdown(r"""<div class='sb' style='max-width:1020px; margin-top: 2em;'>
Ucar's 2021 paper contains a semi-empirical criterion $(23)$ for the stability of the angular motion of an isotropic body. Here, this derivation is reformulated
by using elevation angle (also called tilt angle) $\gamma$ of the rotating field and simplified. Assuming a solution in which the motion is symmetric and fully
synchronized with the driving field (so that time-dependent terms vanish), the function $F$ is defined as the right-hand side of the equation of motion about state
variable $\theta$. By taking the derivative $dF/d\theta$, it is shown that $F$ has a maximum slope as it crosses zero at equilibrium angle $\theta_0$. Comparing
this maximum slope value to the quantity $I \omega^2$ (scalar moment-of-inertia multiplied by the square of the driving field angular velocity), it is stated that
the motion can be stable and symmetric around the $z$-axis when the following condition is met.

$
(1)\phantom{0} \qquad \dfrac{d F}{d \theta} < I \omega^2 \, .$

Here $\omega$ denotes the angular velocity of the driving magnetic field $B$. This semi-empirical formula is accurate in the undamped case. In the presence of damping,
the minimum angle $\gamma$ required for stability is found greater than the value corresponding to zero damping and can be determined numerically using the eigenvalue analysis.

The equilibrium solution given in 2021 paper (with Euler angle $\varphi$ relabeled as $\theta\,$) can be written as

$
(2)\phantom{0} \qquad \tau_C \cos \theta_0 + \tau_S \sin \theta_0 - I \omega^2 \sin \theta_0 = 0 \, .
$

Here $\tau_C$ and $\tau_S$ correspond to the maximum strengths of cyclic (rotating) and static magnetic torques such as

$
(3)\phantom{0} \qquad \tau_C = \lvert m \rvert \, \lvert B_{\perp} \rvert \,,
    \qquad \tau_S = \lvert m \rvert \, \lvert B_{\parallel} \rvert \,,
$

where $\lvert m \rvert$ is the strength of the magnetic moment of the floating body,
$\lvert B_{\perp} \rvert$ is the strength of the rotating magnetic field orthogonal to the $z$-axis and
$\lvert B_{\parallel} \rvert$ is the strength of a static magnetic field parallel to $z$-axis.

In the case of $B_{\perp}$ and $B_{\parallel}$ are respectively radial and axial components of a field $B$, these components can be formulated as

$
(4)\phantom{0} \qquad B_{\perp} =     \lvert B \rvert \cos \gamma \,,
    \qquad B_{\parallel} = \lvert B \rvert \sin \gamma \, .
$

where $\gamma$ is deviation angle of $B$ from the radial ($xy$) plane. Applying this to $(2)$, we obtain

$
(5)\phantom{0} \qquad 0 = m\!B \big( \cos\gamma \, \cos\theta_0 + \sin\gamma \, \sin\theta_0 \big) - I \,\omega^2 \sin\theta_0, \qquad m\!B = \lvert m \rvert \, \lvert B \rvert \, .
$

which is equal to the equation of motion about $\ddot\theta$ under equilibrium condition for an isotropic body with zero damping.

This equation allows to derive $\theta_0$ as

$
(6)\phantom{0} \qquad \theta_0 = \tan^{-1}\Big(\dfrac{m\!B \,\cos \gamma}{I\,\omega^2 -m\!B \,\sin \gamma}\Big) \, .
$

Introduce variables $u$ and $v$ by

$
(7)\phantom{0} \qquad u\,\cos v  = I \omega^2 - m\!B\,\sin\gamma\,,
$

$
(8)\phantom{0} \qquad u\,\sin v  = m\!B\,\cos\gamma\,,
$

so that

$
(9)\phantom{0} \qquad u = \sqrt{{(I \omega^2 - m\!B\,\sin\gamma)}^2 + {(m\!B\,\cos\gamma)}^2 \rule{0pt}{0.9em}} \,,
$

$
(10) \qquad v = \tan^{-1}\Big(\dfrac{m\!B\,\cos\gamma}{I \omega^2 - m\!B\,\sin\gamma}\Big) \, .
$

This way under equilibrium condition, the function $F$ ( right hand side of $(5)$ ) can be expressed as

$
(11) \qquad F(\theta_0) = u\, \cos v \, \sin \theta - u\, \sin v \, \cos \theta \, .
$

Factoring and applying trigonometric identity

$
(12) \qquad F(\theta_0) = u\, \sin (\theta_0 - v) \, .
$

Expressing $\tfrac{d F_0}{d \theta}$ in the same way

$
(13) \qquad F'(\theta_0) = u\, \cos (\theta_0 - v) \, .
$

Under equilibrium state $(\theta = \theta_0)$, the term $(v - \theta_0)$ becomes zero since $v = \theta_0$ according $(6)$ and $(10)$. Applying this in $(13)$,
the cosine term becomes equal to one and vanishes.

$
(14) \qquad F'(\theta_0) = u.
$

By using this substitution, the empirical stability condition $(1)$ can be written as

$
(15) \qquad u < I \omega^2 \, .
$

Squaring both sides and arranging terms, the inequality becomes

$
(16) \qquad {(I \omega^2)}^2 - u^2 > 0 \, .
$

By expanding $u^2$ and applying Pythagorean identity as

$
(17) \qquad u^2 ={(I \omega^2)}^2 - 2I \omega^2 \,m\!B\, \sin\gamma + m\!B^2 \,,
$

we can express $(16)$ as

$
(18) \qquad 2 I \omega^2 \,m\!B\, \sin \gamma - m\!B^2 > 0 \, .
$

Simplify ($m\!B$ is a positive figure)

$
(19) \qquad 2 I \omega^2 \sin \gamma - m\!B > 0 \, .
$

Finally obtain the solution of semi-empirical criterion $(1)$ as

$\boxed{
(20) \qquad \sin \gamma > \dfrac{m\!B}{2 I \omega^2}\,.\qquad
}$

As a note, this equation gives precise results in simulations despite being derived from an empirical criterion.

This result can be verified by defining $\sin\gamma_{\min} = \dfrac{m\!B}{2 I \omega^2}$ and applying it to $(9)$.

$
(21) \qquad u \Big|_{\large \gamma=\gamma_{\min}} = I \omega^2\, .
$


Applying this to $(14)$, we obtain the derivative of $F_0$ with respect to $\theta$ when $\gamma = \gamma_{\min}$

$
(22) \qquad F'(\theta_0)\Big|_{\large \gamma=\gamma_{\min}} = I \omega^2 \,,
$

which is consistent with $(1)$ and may help to verify this derivation.
<hr>
The stability criterion obtained in Ucar's 2021 paper:<br>

$
(23) \qquad \tau_S > I \omega^2 - {\big( {(I \omega^2)}^2 - {\tau_C}^2\big)}^{1/2} \, .
$
<hr>
<b>Notes:</b>

In the case where the magnetic field $\vec{B}$ is produced by a magnetic moment $\vec{m}_R$ rotating about the $z$-axis, the relation between the elevation angle $\gamma_m$ of
this moment from the $xy$-plane and the elevation angle of $\vec{B}$ from the same plane (denoted here as $\gamma_B$ for clarity) at any point on the $z$-axis can be found as

$
\boxed{(24)\qquad \tan\gamma_B = - 2\tan\gamma_m\vphantom{\Big(}\,.\;}
$

This relation can be generalized to any field point $\vec B$ by introducing a local coordinate system whose $z$-axis is aligned with a vector $\vec{r}\;$ joining the dipole
moment and the field point. In terms of zenith angles ($\, \zeta_m = \pi/2 - \gamma_m, \; \zeta_B = \pi/2 - \gamma_B\,$), one finds

$
\boxed{(25)\qquad \cot\zeta_B = - 2\cot\zeta_m\vphantom{\Big(}\,.\;}
$

<style>
  .theorem {
    border: 1px solid rgba(0,0,0,0.12);
    padding: 12px 16px;
    border-radius: 8px;
    background: #f0f0f0;
    color: #111;
    line-height: 1.45;
    box-shadow: 0 1px 2px rgba(16,24,40,0.03);
    transition: background .18s ease, color .18s ease, border-color .18s ease;
  }

  .theorem strong { font-size: 1.05em; }

  /* Dark theme support */
  @media (prefers-color-scheme: dark) {
    .theorem {
      background: #595959; /* linear-gradient(180deg, #0f1113 0%, #121217 100%); */
      border: 1px solid rgba(255,255,255,0.06);
      color: #e6eef8;
      box-shadow: 0 1px 6px rgba(0,0,0,0.6);
    }
    .theorem strong { color: #ffffff; }
  }

  /* Optional explicit fallback if you use a `data-theme` attribute */
  [data-theme="dark"] .theorem {
    background: #0f1113;
    border: 1px solid rgba(255,255,255,0.06);
    color: #e6eef8;
  }
</style>

<div class="theorem" style="max-width:1020px;">
<strong style="font-size:1.05em;">Proof:</strong>
<div style="margin-top:8px; line-height:1.4;">

Having a point dipole $\mathbf m$, its field $\mathbf{B}$ at offset pointed by vector $\mathbf{r}$ reads as

$
(26)\qquad \mathbf{B}(\mathbf{r}) \;=\; \dfrac{\mu_0}{4\pi\,r^3}\Big(\,3(\mathbf{m}\cdot\hat{\mathbf r})\,\hat{\mathbf r}\;-\;\mathbf{m}\,\Big),
\qquad \hat{\mathbf r}=\dfrac{\mathbf r}{r}\,.%
$

By decomposing $\mathbf m$ into components parallel and orthogonal to $\mathbf r$

$
(27)\qquad \mathbf{m} = \mathbf{m}_{\parallel} + \mathbf{m}_{\perp}\,
    \qquad \mathbf{m}_{\parallel} = (\mathbf{m}\cdot\mathbf{\hat{r}})\,\mathbf{\hat{r}}\,,
    \quad  \mathbf{m}_{\perp} = \mathbf{m} - \mathbf{m}_{\parallel}\,,
$

and plug in $(26)$, we get

$
(28)\qquad \mathbf{B} \;=\; \dfrac{\mu_0}{4\pi\,r^3}\Big(\,2 \mathbf{m}_{\parallel} - \mathbf{m}_{\perp}\Big)\,.
$

Similarly, by decomposing $\mathbf{B}$ into $\mathbf B_\parallel + \mathbf B_\perp$ with respect to $\mathbf{r}$, we obtain

$
(29)\qquad \mathbf{B}_{\parallel} = 2\,\dfrac{\mu_0}{4\pi\,r^3}\, \mathbf{m}_{\parallel}\,,
$

$
(30)\qquad \mathbf{B}_{\perp} = -\dfrac{\mu_0}{4\pi\,r^3}\,\mathbf{m}_{\perp}\,,
$

Taking component ratios gives

$
(31)\qquad \dfrac{\mathbf{B}_{\parallel}}{\mathbf{B}_{\perp}} = -2\,\dfrac{\mathbf{m}_{\parallel}}{\mathbf{m}_{\perp}}\,
\quad\Longrightarrow\quad \cot\zeta_B = - 2\cot\zeta_m\,,
$

where $\zeta_B$ and $\zeta_m$ are the angles between the corresponding vectors and vector $\mathbf r$.
The negative sign arises from Eq. (30), which shows that the orthogonal components of these vectors are antiparallel.

</div></div> <!-- proof -->
</div>"""))