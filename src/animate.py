'''
Notebook     : Rigid Body Dynamics with Magnetic Moment in a Rotating Field (with Damping)
Authors      : Hamdi Ucar, Daniel Paschall
Date         : 2025-05-19
Revision Date: 2025-07-18
Revision     : 718.0
'''

import time
import datetime
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D, proj3d
from matplotlib.patches import FancyArrowPatch
import os
from typing import Callable, Optional
from ipywidgets import Label, Button, Dropdown, FloatSlider, HBox, VBox, HTML as iHTML, Layout, Output
from IPython.display import display, HTML
from scipy.interpolate import interp1d
import base64

view_elev = 10
view_azim = -60
Shape = 'Spheroid'
Scale = 1
Animation = None

def GetViewAngles():
    global view_elev, view_azim
    return view_elev, view_azim

def SetViewAngles(angles):
    global view_elev, view_azim
    view_elev, view_azim = angles

def GetShape():
    global Shape
    return Shape

shape_opts = ['Spheroid', 'Cylinder', 'Cuboid']

def SetShape(s):
    global Shape
    prefix = s[:3]
    try:
        Shape = next(o for o in shape_opts if o.startswith(prefix))
    except StopIteration:
        Shape = shape_opts[0]    # fallback if nothing matches

def GetScale():
    global Scale
    return Scale

def SetScale(s):
    global Scale
    Scale = s

def CancelAnim():
    global Animation
    if Animation:
        Animation.event_source.stop()

def xyz_buttons_css():
    def make_img_css(class_name, png_path):
        with open(png_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        return f"""<style>
.{class_name} {{width: 28px; height: 28px; padding: 0; margin: 0 5px; border: none; background-image: url("data:image/png;base64,{b64}"); background-repeat: no-repeat; background-position: center; background-size: contain;}}
</style>"""
    # Build & inject the CSS for all three icons at once:
    css = (
        make_img_css("xy", "view_xy.png") +
        make_img_css("yz", "view_yz.png") +
        make_img_css("xz", "view_xz.png")
    )
    display(HTML(css))

# Helper to make a view-button given its class and angles
def make_coord_vectors(fig, ax)->Axes3D:
    # 4) Create an inset 3D axes in the lower‐left corner
    inset_size = 0.18  # fraction of figure
    # [left, bottom, width, height] in figure coords
    ax_inset = fig.add_axes([0.00, 0.00, inset_size, inset_size], projection='3d')
    ax_inset.set_proj_type('ortho')  # use orthographic projection

    ax_inset.set_axis_off()
    ax_inset.margins(x=0, y=0, z=0)
    lm = 0.86
    ax_inset.set_xlim(-lm,lm); ax_inset.set_ylim(-lm,lm); ax_inset.set_zlim(-lm,lm)
    ax_inset.patch.set_alpha(0)  # transparent
    #ax_inset.patch.set_facecolor('whitesmoke') #test
    #ax_inset.patch.set_alpha(1.0)              #test

    # 5) Draw unit triad in the inset
    arrow_len = 1.0
    text_offs = 0.09
    z_corr =1.2
    arrow_kw = dict(length=arrow_len, normalize=False, arrow_length_ratio=0.25, linewidth=1.3)
    arrow_kwz = dict(length=arrow_len*z_corr, normalize=False, arrow_length_ratio=0.25, linewidth=1.3)
    ax_inset.quiver(0,0,0, 1,0,0, color='r', **arrow_kw)
    ax_inset.quiver(0,0,0, 0,1,0, color='g', **arrow_kw)
    ax_inset.quiver(0,0,0, 0,0,1, color='b', **arrow_kwz)

    ax_inset.text(arrow_len+text_offs, -text_offs, -text_offs, 'x', color='r', fontsize=8)
    ax_inset.text(-text_offs, arrow_len+text_offs, -text_offs, 'y', color='g', fontsize=8)
    ax_inset.text(-text_offs, -text_offs, arrow_len*z_corr+text_offs, 'z', color='b', fontsize=8)
    return ax_inset

# ——————— Rotation Helper ———————
def make_R(θ, φ, ψ):
    sθ,cθ = np.sin(θ), np.cos(θ)
    sφ,cφ = np.sin(φ), np.cos(φ)
    sψ,cψ = np.sin(ψ), np.cos(ψ)
    Rz = lambda s,c: np.array([[c,-s,0],[s,c,0],[0,0,1]])
    Rx = lambda s,c: np.array([[1,0,0],[0,c,-s],[0,s,c]])
    return Rz(sψ,cψ) @ Rx(sφ,cφ) @ Rz(sθ,cθ)

# ——————————————————————————————————————————————————————————————————————
def make_spheroid(abc, scale, tc, bc, div_u=24, div_v=36, n_sectors=8, lf=1.2, df=0.8):
    """
    Generate a spheroid mesh and per‐vertex facecolors with meridian shading.

    Parameters
    ----------
    hr       : float
        Ratio of polar radius (c) to equatorial radius (a).  
        Equatorial radius a = scale, polar radius c = scale * hr.
    scale    : float
        Equatorial radius (semi‐axis) of the spheroid.  The spheroid’s cross‐section
        in x–y has radius = scale.  The z‐axis semi‐axis is scale/hr.
    tc, bc   : tuple of 3 floats
        RGB colors for the “top hemisphere” (z ≥ 0) and “bottom hemisphere” (z < 0).
    div_u    : int
        Number of subdivisions along the “latitude” (0→π).  Higher ⇒ finer mesh.
    div_v    : int
        Number of subdivisions along the “longitude” (0→2π).  Higher ⇒ finer mesh.
    n_sectors : int
        How many vertical meridian sectors to alternate shading.  
        Must be an integer ≥1.
    lf, df   : float
        Multiplicative factors for “light” vs. “dark” shading on alternating meridians.
        Typically lf>1 (lighten) and df<1 (darken).

    Returns
    -------
    x, y, z : ndarrays of shape (div_v+1, div_u+1)
        The coordinates of the spheroid’s mesh vertices in Cartesian space.
    colors   : ndarray of shape (div_v+1, div_u+1, 3)
        Per‐vertex RGB colors (floats in [0,1]) after applying hemisphere base colors
        and alternating meridian shading.
    """
    # Equatorial semi‐axes = a,b, polar semi‐axis = c
    #a = scale * abc[0]
    #b = scale * abc[1]
    #c = scale

    # Create parameter arrays: u = polar angle from z+ (0 → π), v = azimuth (0 → 2π)
    u = np.linspace(0.0, np.pi, div_u + 1)      # latitude grid
    v = np.linspace(0.0, 2.0 * np.pi, div_v + 1)  # longitude grid
    phi, theta = np.meshgrid(u, v, indexing='xy')

    # Parametric equations for a spheroid (axis of symmetry along z)
    x = scale * abc[0] * np.sin(phi) * np.cos(theta)
    y = scale * abc[1] * np.sin(phi) * np.sin(theta)
    z = scale * np.cos(phi)

    # Initialize color array (div_v+1, div_u+1, 3)
    colors = np.zeros(x.shape + (3,), dtype=float)

    # Base hemisphere mask: top (z >= 0) vs. bottom (z < 0)
    mask_top = (z >= 0.0)

    # Assign base color: top = tc, bottom = bc
    colors[mask_top, 0] = tc[0]
    colors[mask_top, 1] = tc[1]
    colors[mask_top, 2] = tc[2]
    colors[~mask_top, 0] = bc[0]
    colors[~mask_top, 1] = bc[1]
    colors[~mask_top, 2] = bc[2]

    # Compute which meridian sector each longitude belongs to
    # sector index = floor(theta / (2π / n_sectors)), then mod 2 for alternating
    sector_index = np.floor(theta / (2.0 * np.pi / n_sectors)).astype(int) % 2

    # Apply light/dark shading factor to each vertex’s base color
    # Top‐hemisphere & even sector ⇒ lighten by lf; & odd ⇒ darken by df
    # Bottom‐hemisphere similar
    colors[mask_top & (sector_index == 0)] *= lf
    colors[mask_top & (sector_index == 1)] *= df
    colors[~mask_top & (sector_index == 0)] *= lf
    colors[~mask_top & (sector_index == 1)] *= df

    # Ensure RGB still in [0,1]
    np.clip(colors, 0.0, 1.0, out=colors)

    return x, y, z, colors


# ——————————————————————————————————————————————————————————————————————
def make_cylinder(xyz_ratios, scale, tc, bc, div_z=2, div_theta=36, n_sectors=4, lf=1.2, df=0.8, div_r=12):
    """Generate a vertical cylinder mesh (sides + caps) with meridian shading."""
    # sides
    z_vals = np.linspace(-1/2, 1/2, div_z+1)
    th_vals = np.linspace(0, 2*np.pi, div_theta+1)
    θs, zs = np.meshgrid(th_vals, z_vals)
    x = np.cos(θs)*scale*xyz_ratios[0]
    y = np.sin(θs)*scale*xyz_ratios[1]
    z = zs*scale
    colors = np.zeros(x.shape+(3,))
    top = (zs>=0)
    colors[top]=tc; colors[~top]=bc

    sec = np.floor(θs/(2*np.pi/n_sectors)).astype(int) % 2
    colors[top  & (sec==0)] *= lf
    colors[top  & (sec==1)] *= df
    colors[~top & (sec==0)] *= lf
    colors[~top & (sec==1)] *= df
    np.clip(colors,0,1,out=colors)
    # Caps
    r_vals = np.linspace(0, 1, div_r)
    θc, rc = np.meshgrid(th_vals, r_vals)
    x_cap = rc*np.cos(θc)*scale*xyz_ratios[0]
    y_cap = rc*np.sin(θc)*scale*xyz_ratios[1]
    z_top = np.full_like(x_cap, +scale/2)
    z_bot = np.full_like(x_cap, -scale/2)
    colors_top = np.zeros_like(colors[0:len(r_vals)])
    colors_bot = np.zeros_like(colors_top)
    # broadcasting tc/bc
    colors_top[:] = tc; colors_bot[:] = bc
    sec_c = np.floor(θc/(2*np.pi/n_sectors)).astype(int)%2
    colors_top[sec_c==0] *= lf; colors_top[sec_c==1] *= df
    colors_bot[sec_c==0] *= lf; colors_bot[sec_c==1] *= df
    np.clip(colors_top,0,1,out=colors_top)
    np.clip(colors_bot,0,1,out=colors_bot)
    return (x, y, z, colors), (x_cap, y_cap, z_top, colors_top), (x_cap, y_cap, z_bot, colors_bot)

# ——————————————————————————————————————————————————————————————————————
def make_cuboid(ratios, scale, tc, bc, div=2, n_sectors=None, lf=None, df=None):
    """
    Generate 6 faces of a centered cuboid whose
    half‐height in z is `scale`, and whose half‐widths
    in x and y are `ratios[0]*scale`, `ratios[1]*scale`.

    Returns a list of six (X,Y,Z,colors) tuples.

    Parameters
    ----------
    ratios : tuple of two floats
        (x/z, y/z) ratios.  e.g. (2.0, 0.5) means
        x_half = 2*scale, y_half = 0.5*scale
    scale : float
        half‐height of the cuboid along z (so z_half = scale)
    tc : length‐3 tuple of floats
        RGB color for the ±x and ±y faces
    bc : length‐3 tuple of floats
        RGB color for the ±z faces
    div : int, optional
        number of subdivisions along each edge (mesh resolution)
    """
    # half‐sizes
    z_half = scale
    x_half = ratios[0] * scale
    y_half = ratios[1] * scale

    # coordinate arrays
    vals_x = np.linspace(-x_half, x_half, div)
    vals_y = np.linspace(-y_half, y_half, div)
    vals_z = np.linspace(-z_half, z_half, div)

    # face specs: (fixed_axis, fixed_value, u_axis, v_axis)
    specs = [
        ('x',  x_half, 'y', 'z'),   # +X face
        ('x', -x_half, 'y', 'z'),   # -X face
        ('y',  y_half, 'x', 'z'),   # +Y face
        ('y', -y_half, 'x', 'z'),   # -Y face
        ('z',  z_half, 'x', 'y'),   # +Z face (top)
        ('z', -z_half, 'x', 'y')    # -Z face (bottom)
    ]

    axis_vals = {'x': vals_x, 'y': vals_y, 'z': vals_z}
    faces = []

    for idx, (fix_ax, fix_val, u_ax, v_ax) in enumerate(specs):
        U1 = axis_vals[u_ax]
        V1 = axis_vals[v_ax]
        U, V = np.meshgrid(U1, V1, indexing='xy')

        # build the X, Y, Z coordinate grids
        X = np.zeros_like(U)
        Y = np.zeros_like(U)
        Z = np.zeros_like(U)
        for arr, name in zip((X, Y, Z), ('x', 'y', 'z')):
            if name == fix_ax:
                arr[:, :] = fix_val
            elif name == u_ax:
                arr[:, :] = U
            else:
                arr[:, :] = V

        # choose face color: first four faces (±x, ±y) get tc, last two get bc
        base_color = tc if fix_ax in ('x', 'y') else bc
        # build the RGB array
        colors = np.empty(X.shape + (3,), dtype=float)
        colors[..., 0] = base_color[0]
        colors[..., 1] = base_color[1]
        colors[..., 2] = base_color[2]

        faces.append((X, Y, Z, colors))

    return faces

# ——————————————————————————————————————————————————————————————————————
def AnimateSim(preview_out: Output|None, title:str, sim_vals: tuple|None, path_to_sim_csv: str|None, file_id: str|None, params:tuple, body_specs:tuple,
               vid_Length: float, fps:float, t_start:float, t_span: float,
               callback: Optional[Callable[[int,int],None]]=None, interval: int = 20) -> str:
    """
    Generate a 3D MP4 animation of a floating body's orientation.

    Args:
        path_to_sim_csv (str): Path to the simulation CSV file.
        extracted_params (dict): Initial parameters ('th','ph','ps','phd','psd','lam','mB','gam','omega').
        floater_shape (str): 'cylinder', 'cuboid', or 'spheroid'.
        vid_Length  (float): output video length in seconds.
        fps           (int):   frames per second for the output video.
        t_start     (float): simulation time (s) at which to begin sampling.
        t_span(float|None): length of simulation time (s) to include; if None, uses full run.

    Return: Filename of the generated video
    """

    # ——————————————————————————————————————————————————————————————————————
    def ac_bc_from_I(R1, R2):
        """
        Given I1/I3 and I2/I3 for a solid elliptical cylinder
        (I1 about semi-axis a, I2 about semi-axis b, I3 about symmetry axis),
        return (a/c, b/c).
        """
        # R1 = I1_I3_ratio
        # R2 = I2_I3_ratio

        denom = (R1 + R2 - 1)
        if abs(denom) < 1e-12:
            raise ValueError("Degenerate: R1+R2 must not equal 1")

        # Total normalized cross‐section radius‐squared
        Q = (2.0/3.0) / denom

        s2 = R1 * Q - 1.0/3.0
        r2 = R2 * Q - 1.0/3.0

        if s2 <= 0 or r2 <= 0:
            raise ValueError(f"Non‐physical solution: a^2/c^2={r2}, b^2/c^2={s2}")

        return math.sqrt(r2), math.sqrt(s2)

    # ——————————————————————————————————————————————————————————————————————
    def xz_yz_from_I(R1, R2):
        """
        Given R1 = I_x / I_z and R2 = I_y / I_z for a solid rectangular prism,
        return (x/z, y/z).

        Solves:
            A*(R2-1) + B*R2       = 1
            A*R1       + B*(R1-1) = 1
        with A = (x/z)^2, B = (y/z)^2.
        """
        # set up the 2×2 linear system M @ [A,B] = [1,1]
        M = [[R2 - 1, R2    ],
             [R1    , R1 - 1]]
        det = M[0][0]*M[1][1] - M[0][1]*M[1][0]
        if abs(det) < 1e-16:
            raise ValueError("Singular system: no unique solution for these ratios")
        # Cramer's rule
        # A = det([[1, R2],[1, R1-1]]) / det(M)
        A = (1*(M[1][1]) - M[0][1]*1) / det
        # B = det([[R2-1, 1],[R1, 1]]) / det(M)
        B = (M[0][0]*1 - 1*(M[1][0])) / det

        if A <= 0 or B <= 0:
            raise ValueError(f"Non‐physical solution: a^2={A}, b^2={B}")

        return math.sqrt(A), math.sqrt(B)

    # ——————————————————————————————————————————————————————————————————————
    def build_shape_mesh(shape, scale):
        nonlocal ratios
        #print(f"Shape={shape}    Scale={Scale}")
        if scale == 0:
            return None
        if shape.startswith('Sph'):
            ratios = xz_yz_from_I(I1_I3_ratio, I2_I3_ratio)
            sc = scale / np.max(ratios)
            # use sphere as ref for MOI
            return [make_spheroid(ratios, sc, tc, bc, div_u, div_v, n_sectors, light_factor, dark_factor)]
        if shape.startswith('Cyl'):
            ratios =  ac_bc_from_I(I1_I3_ratio, I2_I3_ratio)
            sc = scale / 0.82 /np.max(ratios) #0.9076 * 2
            mesh_s, mesh_t, mesh_b = make_cylinder(ratios, sc, tc, bc, div_u, div_v, n_sectors, light_factor, dark_factor)
            return [mesh_s, mesh_t, mesh_b]
        if shape.startswith('Cub'):
            ratios = xz_yz_from_I(I1_I3_ratio, I2_I3_ratio)
            sc = scale * 0.5/0.7933 / np.max(ratios)
            return make_cuboid(ratios, sc, tc, bc, 2, n_sectors, light_factor, dark_factor)
        raise ValueError(f"Selected body shape {shape} is not available.")

    # ——————————————————————————————————————————————————————————————————————
    global view_elev, view_azim, Animation
    degree =180/np.pi  #convert rad to degrees
    ratios = None
    last_xyz_id = 0
    start_time = time.perf_counter()
    sim_end_full = 0
    t_end = 0
    folder_path = os.path.dirname(path_to_sim_csv) if path_to_sim_csv else 'RESULTS/TEMP'
    if folder_path:
        folder_path=folder_path+'/'
    try:
        top_color, bot_color, n_sectors, vec_m_col, vec_b_col, light_factor, dark_factor, div_u, div_v, vecB, vecM, text, aa, show_src, fig_size, I1_I3_ratio, I2_I3_ratio = body_specs
    except:
        #'#70A0E0', '#E0A070', 4, '#4040FF', '#FF4040', 1.2, 0.9, 24, 24, 1, 1, 1, 0,0
        print('error on parsing animation parameters, using defaults.')
        top_color = '#6898FF'
        bot_color = '#38F098'
        n_sectors = 4
        vec_m_col = '#4040FF'
        vec_b_col = '#FF4040'
        light_factor = 1.2
        dark_factor = 0.9
        div_u = 18
        div_v = 24
        vecB = 1
        vecM = 1
        text = 1
        aa = 0
        show_src = 1
        I1_I3_ratio = I2_I3_ratio = 1
        fig_size = (6, 6)

    tc = np.array(mcolors.to_rgb(top_color))
    bc = np.array(mcolors.to_rgb(bot_color))

    # --- extract constants ---
    om_B    = params.get('om',  0.0)
    mB      = params.get('mB',  0.0)
    gamma   = params.get('gam', 0.0)
    theta0  = params.get('th',  0.0)
    phi0    = params.get('ph',  0.0)
    psi0    = params.get('ps',  0.0)
    thd0    = params.get('thd', 0.0)
    phd0    = params.get('phd', 0.0)
    psd0    = params.get('psd', 0.0)
    lam     = params.get('lam', 0.0)

    use_simvals = False
    if sim_vals:
        t_vals, y_vals = sim_vals
        if t_vals is not None and len(t_vals) > 0:
            use_simvals = True

    # load the simulation file
    frames=None
    # names & interpolation kinds for all six variables    frames=None
    names = ['the','phi','psi','thd','phd','psd']
    kinds =    ['quadratic'] * 6

    # ————————————————————————————————————————————
    if preview_out:
        frames = [(0.0, theta0, phi0, psi0, thd0, phd0, psd0)]
        total_frames = 1
    else:
        if use_simvals:
            sim_end_full = t_vals[-1]
            if t_start > sim_end_full or t_end > sim_end_full:
                raise ValueError(f"Selected window ({t_start:.3f} - {t_end:.3f}) exceeds simulation length ({sim_end_full:.3f}).")
            # determine time window
            t_end = sim_end_full if t_span is None else t_start + t_span
            i0 = np.searchsorted(t_vals, t_start, side='left')
            i1 = np.searchsorted(t_vals, t_end,   side='right')

            # slice out
            win_times = t_vals[i0:i1]
            win_vals = [y[i0:i1] for y in y_vals]
            # ————————— are we doing interpolation or one to one mapping —————————
            if vid_Length > 0: # interpolation
                total_frames = int(vid_Length * fps)
                frame_times  = np.linspace(t_start, t_end, total_frames)

                interps = {
                    name: interp1d(
                        win_times, win_vals[i],
                        kind=kinds[i],
                        bounds_error=False,
                        fill_value=(win_vals[i][0], win_vals[i][-1])
                    )
                    for i,name in enumerate(names)
                }

                # build frames_list by evaluating each interpolator
                frames = [ (
                        t,
                        interps['the'](t),
                        interps['phi'](t),
                        interps['psi'](t),
                        interps['thd'](t),
                        interps['phd'](t),
                        interps['psd'](t),
                    ) for t in frame_times ]
            else: # ——————————————— one to one mapping ———————————————
                frames = [(t, *ys)
                    for t, *ys in zip(win_times, *win_vals)]
                total_frames = len(frames)
        # ————————————————————————————————————————————
        elif path_to_sim_csv and os.path.exists(path_to_sim_csv):
            try:
                df = pd.read_csv(path_to_sim_csv).apply(pd.to_numeric, errors='coerce')
                base_times = df['t'].values
                base_vals  = [
                    df['theta'].values,
                    df['phi'].values,
                    df['psi'].values,
                    df['theta_dot'].values,
                    df['phi_dot'].values,
                    df['psi_dot'].values,
                    ]
            except Exception as e:
                raise RuntimeError(f"Error on reading CSV file: {e}")

            # determine time window
            sim_end_full = base_times[-1]
            t_end = sim_end_full if t_span is None else t_start + t_span
            if t_start > sim_end_full or t_end > sim_end_full:
                raise ValueError(f"Selected window ({t_start:.3f} - {t_end:.3f}) exceeds simulation length ({sim_end_full:.3f}).")

            mask = (base_times >= t_start) & (base_times <= t_end)
            # slice down to only the needed segment
            win_times = base_times[mask]
            win_vals  = [arr[mask] for arr in base_vals]

            # ————————— are we doing interpolation or one to one mapping —————————
            if vid_Length > 0: # interpolation
                total_frames = int(vid_Length * fps)
                frame_times = np.linspace(t_start, t_end, total_frames)

                interps = {
                    name: interp1d(
                        win_times,
                        win_vals[i],
                        kind=kinds[i],
                        bounds_error=False,
                        fill_value=(win_vals[i][0], win_vals[i][-1])
                    )
                    for i, name in enumerate(names)
                }

                frames = [ (
                        t,
                        interps['the'](t),
                        interps['phi'](t),
                        interps['psi'](t),
                        interps['thd'](t),
                        interps['phd'](t),
                        interps['psd'](t),
                    ) for t in frame_times ]
            else: # ——————————————— one to one mapping ———————————————
               frames = [(t, *ys)
                   for t, *ys in zip(win_times, *win_vals)]
               total_frames = len(frames)

                #frames = [
                #    (t, the, phi, psi, thd, phd, psd)
                #      for t, the, phi, psi, thd, phd, psd
                #      in zip(times_win, *vals_win)
                #]
        # —————————————————————————————————————————————————————————————
        if frames is None:
            raise RuntimeError("No simulation result available.")
    # ————————————————————————————————————————————————————————————————————

    mesh_pieces = build_shape_mesh(Shape, Scale)
    figOut = preview_out or Output()
    with figOut:
        fig = plt.figure(figsize=fig_size, facecolor="#e0e0e0")
        fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0, hspace=0.0)
        fig.canvas.header_visible  = False
        fig.canvas.toolbar_visible = False
        fig.canvas.footer_visible  = False
        fig.tight_layout()

        ax  = fig.add_subplot(111, projection='3d')
        ax.set_box_aspect([1,1,1])
        ax.set_proj_type('ortho')  # use orthographic projection
        ax.set_xlim(-1,1); ax.set_ylim(-1,1); ax.set_zlim(-1,1)
        ax.axis('off')

        ax_inset = make_coord_vectors(fig, ax)
        ax.view_init(elev=view_elev, azim=view_azim)
        ax_inset.view_init(elev=view_elev, azim=view_azim)
        # --- Set up figure and draw the initial surface ---

        # optional legend + title
        legend = [
            Line2D([0],[0],color=vec_m_col, lw=2),
            Line2D([0],[0],color=vec_b_col, lw=2),
        ]
        ax.legend(legend, ['Body axis','B-field'], loc='upper right',fontsize='small'
                 )
        if not preview_out:
            if title:
                fig.suptitle(title)
            if show_src:
                fig.text(0.02,0.02, f"Source: {path_to_sim_csv}", color='gray', fontsize=8)

        # text artists
        var_artists = []
        line_fac = 0.27/fig_size[1] # height of the figure
        for i in range(13):
            t = fig.text(0.005, 0.97 - i*line_fac, '', fontsize=8)
            var_artists.append(t)

        plt.show()

    # --- Animation update function ---
    def update(i):
        #print('update 1')
        # 1) clear previous sphere
        for col in list(ax.collections):
            col.remove()
            #if col not in static_quivers:
            #    col.remove()

        # 2) read angles for this frame
        t, θ, φ, ψ, θd, φd, ψd = frames[i]
        # 3) precompute sin/cos once
        sθ, cθ = np.sin(θ), np.cos(θ)
        sφ, cφ = np.sin(φ), np.cos(φ)
        sψ, cψ = np.sin(ψ), np.cos(ψ)

        # z component of nu vector in body coordinates
        nu_z_wo = cθ * ψd + φd
        nu_z_bo = cθ * φd + ψd

        # 4) build ZXZ rotation
        Rz = lambda s, c: np.array([[c,-s,0],[s,c,0],[0,0,1]])
        Rx = lambda s, c: np.array([[1,0,0],[0,c,-s],[0,s,c]])
        R = Rz(sφ, cφ) @ Rx(sθ, cθ) @ Rz(sψ, cψ)

        # 5) rotate mesh
        if mesh_pieces:
            for X0, Y0, Z0, C0 in mesh_pieces:
                coords = np.vstack((X0.ravel(), Y0.ravel(), Z0.ravel()))
                xr, yr, zr = (R @ coords).reshape((3,) + X0.shape)
                ax.plot_surface(xr, yr, zr, facecolors=C0, rstride=1, cstride=1, linewidth=0, antialiased=aa, shade=True, zorder=2)

        if vecB:
            B=[np.sin(om_B*t)*np.cos(gamma), -np.cos(om_B * t)*np.cos(gamma), np.sin(gamma)]
            ax.quiver(0, 0, 0, *B, color=vec_b_col, length=1.5, normalize=True, zorder=1000, linewidth=2, arrow_length_ratio=0.15)
        if vecM:
            m=[sφ*sθ, -cφ*sθ, cθ]
            ax.quiver(0,0,0, *m,color=vec_m_col, length=1.5, normalize=True, zorder=1000, linewidth=2, arrow_length_ratio=0.15)

        # 7) update live text
        var_artists[0].set_text(f"Time = {t:.3f}s")
        var_artists[0].set_color('black')
        if text:
            # build live text lines
            data_lines = [
                f"t = {t:.3f} s (span {t_start:.3f}-{t_end:.3f} s)",
                #f"ω = {omega_r:.6f}",
                #f"mB  = {mB:.6f}",
                #f"γ   = {gamma:.6f}",
                #f"λ   = {lam:.6f}",
                f"θ    = {θ:.8f}",
                f"φ    = {φ:.8f}",
                f"ψ    = {ψ:.8f}",
                f"φ'   = {φd:.8f}",
                f"ψ'   = {ψd:.8f}",
                f"ʷ νz = {nu_z_wo:.8f}",
                f"ᴮ νz = {nu_z_bo:.8f}",
                f"φ-ωt = {((φ-om_B*t)*degree + 180.0) % 360.0 - 180.0 :.8f}°"
            ]
            # update and color text
            for idx, (obj, txt) in enumerate(zip(var_artists, data_lines)):
                obj.set_text(txt)
                obj.set_color('blue' if idx == 2 else 'black')

        if not preview_out:
            #call progress callback every N frames
            if callback and (i % interval) == 0:
                callback(i, total_frames)
        # --------- end of update function ------------

    if preview_out:
        ################################################################################
        # Load and inline each PNG into a distinct CSS class
        xyz_buttons_css()

        # Helper to make a view-button given its class and angles
        def make_view_button(class_name, id, elev, azim):
            btn = Button(
                description="",  # must be empty or you’ll see “…” text
                tooltip=f"{class_name.upper()} View elev={elev}, azim={azim}",
                layout=Layout(width="28px", height="28px", padding="0", margin=('0 10px 0 10px')) ## top right bottom left)
            )
            btn.id = id
            btn.add_class(class_name)  # attach the CSS background

            _next = {
                # x-y
                (  0,   0): (  0, 180),
                (  0, 180): (180, 180),
                (180, 180): (180,   0),
                (180,   0): (  0,   0),
                # y-z
                ( 90, -90): (-90,  90),
                (-90,  90): ( 90,  90),
                ( 90,  90): (-90, -90),
                (-90, -90): ( 90, -90),
                # x- z
                (  0, -90): (  0,  90),
                (  0,  90): (180,  90),
                (180,  90): (180, -90),
                (180, -90): (  0, -90),
            }

            def _on_click(button):
                nonlocal last_xyz_id
                global view_elev, view_azim
                if last_xyz_id == button.id:
                    e, a = _next[( view_elev, view_azim)]
                else:
                    e = elev; a = azim
                    last_xyz_id = button.id
                ax.view_init(elev=e, azim=a)
                ax_inset.view_init(elev=e, azim=a)
                view_elev = e; view_azim = a
                fig.canvas.draw_idle()

            btn.on_click(_on_click)
            return btn

        #--------------- mouse actions --------------------
        # Mouse‐drag handlers to rotate both axes
        _drag = {'press': False, 'x': 0, 'y': 0}

        def on_press(event):
            if event.inaxes == ax:
                _drag['press'] = True
                _drag['x'], _drag['y'] = event.x, event.y

        def on_release(event):
            _drag['press'] = False

        def on_motion(event):
            global view_elev, view_azim
            nonlocal last_xyz_id
            if not _drag['press'] or event.inaxes != ax:
                return
            dx = event.x - _drag['x']
            dy = event.y - _drag['y']
            _drag['x'], _drag['y'] = event.x, event.y

            # sensitivity factors
            view_azim = ax.azim   - dx * 0.5
            view_elev = ax.elev   + dy * 0.5

            # update both views
            ax.view_init(elev=view_elev, azim=view_azim)
            ax_inset.view_init(elev=view_elev, azim=view_azim)
            last_xyz_id = 0
            fig.canvas.draw_idle()
        #--------------- mouse actions end ----------------

        btn_xy = make_view_button("xy", 1, 90, -90)
        btn_yz = make_view_button("yz", 2, 0,    0)
        btn_xz = make_view_button("xz", 3, 0,  -90)

        # 8) Connect events
        fig.canvas.mpl_connect('button_press_event',   on_press)
        fig.canvas.mpl_connect('button_release_event', on_release)
        fig.canvas.mpl_connect('motion_notify_event',  on_motion)

        shape_dropdown = Dropdown(
            options=shape_opts,
            value=Shape,
            description='Shape',
            layout=Layout(width='140px', margin='0 0 0 0'),
            style={'description_width': '50px'}
        )

        def filtered_shape():
            prefix = Shape[:3]
            try:
                val = next(o for o in opts if o.startswith(prefix))
            except StopIteration:
                val = opts[0]    # fallback if nothing matches
            return val

        shape_size = FloatSlider(
            value = Scale,
            min = 0.0, max=1.8, step=0.05,
            readout=True,
            readout_format='.2f',
            description='Scale',
            style={'description_width': '40px'},
            layout=Layout(
                width='200px',
                margin='0 0 0 10px',
                padding='0px'
            )
        )

        rh_template = "<span style='font-weight:600; font-size:14px;'>a/c={vac:.5f}, b/c={vbc:.5f}</span>"
        radHei_ratio = iHTML(value=rh_template.format(vac=1.0, vbc=1.0), layout=Layout(margin='0 5px 0 5px'))

        def update_rh_ratio():
            if ratios:
                radHei_ratio.value = rh_template.format(vac=ratios[0], vbc=ratios[1])

        def on_shape_change(change):
            nonlocal mesh_pieces
            global Shape
            Shape = change['new']
            mesh_pieces = build_shape_mesh(Shape, Scale)
            update(0)
            update_rh_ratio()
            fig.canvas.draw_idle()

        def on_size_change(change):
            nonlocal mesh_pieces
            global Scale
            Scale = change['new']
            mesh_pieces = build_shape_mesh(Shape, Scale)
            update(0)
            fig.canvas.draw_idle()

        shape_dropdown.observe(on_shape_change, names='value')
        shape_size.observe(on_size_change, names='value')

        update_rh_ratio()
        update(0)
        fig.canvas.draw_idle()
        controls = HBox([
                iHTML("<b>Scene Preview</b>", layout=Layout(margin='0 10px 0 0')),  # top right bottom left
                btn_xy, btn_yz, btn_xz,
                shape_dropdown,
                radHei_ratio,
                shape_size
            ],
            layout=Layout(width='800px', height='35px', justify_content='flex-start', margin='0 0 0 0')            # top right bottom left
        )
        preview_out.clear_output(wait=True)
        with preview_out:
            display(controls)
            display(fig.canvas)
        return 0, 0
    else:
        # --- Create & save animation ---
        #metadata = {
        #    'title': 'Simulation Animation',
        #    'artist': 'Hamdi Ucar, Daniel Paschall',
        #    'comment': 'Generated by www.ucareffect.org simulation program U13'
        #}

        writer = FFMpegWriter(
            fps=fps,
            codec='libx264',
            bitrate=None,
            extra_args=[
                # Video quality parameters
                '-pix_fmt', 'yuv420p',
                '-profile:v', 'high',        # H.264 High Profile for best compatibility
                '-level', '4.0',             # level for 1080p@30fps compatibility
                '-crf', '18',
                '-preset', 'medium',
                # Metadata parameters
                '-movflags', '+use_metadata_tags',  # Ensures metadata is preserved
                '-metadata', f'csv={os.path.basename(path_to_sim_csv) if path_to_sim_csv else ""}',
                '-metadata', f'begin={t_start:.6f}',
                '-metadata', f'span={t_span:.6f}'
            ]
        )

        output_file = f"{folder_path}{file_id if file_id else 'ani'}-{datetime.datetime.now():%Y%m%d_%H%M%S}.mp4"
        Animation = FuncAnimation(fig, update, frames=total_frames, interval=1000/fps, blit=False)
        #Animation.save(output_file, writer='ffmpeg')
        Animation.save(output_file, writer=writer, dpi=fig.dpi)  # writer='ffmpeg',
        plt.close(fig)
        exec_time = time.perf_counter()-start_time
        Animation = None
        return output_file, exec_time
