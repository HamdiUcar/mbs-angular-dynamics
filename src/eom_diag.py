# eom_diag1.py
'''
Derives Equations of angular motion of a rigid body with Magnetic Moment in a Rotating Field (with Damping)
Authors: Hamdi Ucar
Date         : 2025-05-01
Revision Date: 2025-07-18
Revision     : 718.0
'''
import sympy as sp
from   sympy import Eq, latex, sin as sin, cos as cos, tan as tan
from   sympy import trigsimp, ratsimp
import numba

import re
from   typing import Callable

from sympy.printing.latex import LatexPrinter

def dotify_latex(expr):
    s = latex(expr)
    s = re.sub(r'th_{dd}', r'\\ddot{\\theta}', s)
    s = re.sub(r'ph_{dd}', r'\\ddot{\\phi}',   s)
    s = re.sub(r'ps_{dd}', r'\\ddot{\\psi}',   s)
    s = s.replace(r'\theta_{dd}',r'\ddot{\theta}')
    s = s.replace(r'\phi_{dd}',r'\ddot{\phi}')
    s = s.replace(r'\psi_{dd}',r'\ddot{\psi}')
    s = s.replace(r'ph_dot', r'\dot{\phi}')
    s = s.replace(r'ps_dot', r'\dot{\psi}')
    s = s.replace(r'\left(t \right)', '')
    s = re.sub(r'\\frac\{d\}\{d t\} ?\\(theta|phi|psi)', r'\\dot{\\\1}', s)
    s = s.replace('\\frac','\\dfrac')
    #s = re.sub(r'\^\{(\d)\}', r'^\1', s)
    s = re.sub(r'([\^_])\{(\d)\}', r'\1\2', s)
    s = re.sub(r'\b(I_\d)\^(\d)\b', r'{\1}^\2', s)
    s = s.replace('{}', '')
    s = re.sub(r'\\left\(\\dot\{\\(theta|phi|psi)\}\\right\)\^2', r'\\dot{\\\1}\\vphantom{^1}^2', s)
    return s

# ──────────────────────────────────────────────────────────────
def equations_in_latex(eqs):
    """
    Given a list of Sympy Eq(lhs, rhs), apply a series of
    collects, expansions, and regex substitutions to produce
    cleaned-up LaTeX strings.
    """
    eqs_latex = []
    for i, eq in enumerate(eqs):
        eqs_latex.append(dotify_latex(eq))
    return eqs_latex

# ──────────────────────────────────────────────────────────────────────
class EOMDiag:

    # ──────────────────────────────────────────────────────────────
    def __init__(self):
        # ──────────────────────────────────────────────────────────────────────
        def _simplify():
            simple = []
            for i, expr in enumerate(self.dd_sol):
                print(f"simplify ({self.symbols_dd[i]})...", end='')
                if i == 0:
                    q = sp.cancel(expr) * I1 * I2
                    q = sp.simplify(q)
                    q = q.collect([mB, xi])
                    q = q/(I1*I2)
                    q = q.collect(sin(ps)**2*sin(th)*ph_d)
                    q = q.collect(sin(omega*t - ph)*sin(ps)*cos(ps)*cos(gamma))
                    q = q.collect(cos(omega*t - ph)*cos(th)*cos(gamma))
                    q = q.subs(I2 - I2*sin(ps)**2, I2*cos(ps)**2)
                    q = q.collect(I1**2 * sin(ps)*cos(ps)*th_d)
                    q = q.collect(I1 * I3 * sin(ps) * cos(ps)*th_d)
                    q = q.collect(I2**2 * sin(ps) * cos(ps)*th_d)
                    q = q.collect(I2**2* sin(th) * ph_d)
                    q = q.collect(sin(ps) * cos(ps) * th_d) #last one
                    q = q.collect(I2 * I3 * sin(th) * ph_d)
                    q = q.collect(I1**2 * sin(ps)**2 *ph_d)
                    q = q.collect(ph_d)
                    q = q.collect(ps_d) # ok
                    q = q.collect(I2 * I3) #!test
                    q = sp.factor_terms(q)
                    q = q.collect(sin(ps)**2 * th_d)
                    A = cos(th) * ph_d + ps_d
                    q = q.collect(A)
                    q = q.collect(sin(gamma) * sin(ps)**2 * sin(th))
                    A = -I1 + I2
                    q = q.collect(A)
                elif i == 1:
                    q = sp.cancel(expr) * I1 * I2 * sin(th)
                    q = sp.simplify(q)
                    q = q/(I1*I2)
                    q = sp.cancel(q.collect(sin(th))/sin(th))
                    q = q.collect([mB, xi])
                    q = q.collect(sin(ps) * cos(ps) * th_d)
                    q = q.collect(sin(ps)**2*cos(gamma))
                    q = q.collect(cos(gamma))
                    q = q.collect(sin(ps)**2 * sin(th) * ph_d)
                    q = q.collect(ph_d)
                    q = q.collect(sin(ps)*cos(ps))
                    q = q.collect(sin(ps)**2)
                    q = q.collect(cos(omega*t - ph) * cos(th))
                    q = q.collect(sin(omega*t - ph))
                    q = q.collect(sin(gamma) * sin(th))
                    q = q.collect(sin(ps)**2)
                    q = q.collect(ps_d * th_d)
                    q = q.collect(sin(th) * cos(th))
                    q = q.collect(cos(th) * th_d)
                    q = q.collect(sin(th) * ps_d)
                    q = q.collect(I1 - I2)
                elif i == 2:
                    q = sp.cancel(expr) * I1 * I2 * I3
                    q = sp.simplify(q)
                    q = q.collect([mB, xi])
                    q = q.collect(sin(omega * t - ph))
                    q = q.collect(cos(omega * t - ph) * sin(ps) * cos(ps) * cos(gamma))
                    q = q.collect(cos(th) * ph_d)
                    q = sp.factor_terms(q, [I1*I2, I1*I3, I2*I3])
                    q = q.collect(sin(ps)*cos(ps)*th_d/sp.tan(th))
                    q = q.collect(cos(th)*ph_d)
                    q = q.collect(I1**2 * I2)
                    q = q.collect(I1**2 * I3)
                    q = q.collect(I2**2 * I3)
                    q = q.collect(I1 * I2**2)
                    q = q.collect(I1 * I3**2)
                    q = q.collect(I2 * I3**2)
                    q = q.collect(I2**3)
                    q = q.collect(I2 * I3) #ok
                    q = q.collect(I1 * I2 * I3)
                    q = q.collect(sin(ps)**5)
                    q = q.subs(1/tan(th), cos(th)/sin(th)) #test
                    q = q / (I1 * I2 * I3)
                else:
                    raise RuntimeError("_simplify")
                simple.append(q)
                print(' done.')
                q = None

            self.dd_sol = sp.Matrix(simple)

        # ──────────────────────────────────────────────────────────────────────
        def Rx(w):
            return sp.Matrix([
                [1,       0,        0],
                [0, cos(w), -sin(w)],
                [0, sin(w),  cos(w)]
            ])

        def Rz(w):
            return sp.Matrix([
                [ cos(w), -sin(w), 0],
                [ sin(w),  cos(w), 0],
                [      0,       0, 1]
            ])

        def _rotation_matrix(th, ph, ps):
            return Rz(ph) * Rx(th) * Rz(ps)
        # ──────────────────────────────────────────────────────────────────────
        # Symbols and Euler–angle functions
        LatexPrinter._default_settings.update({'mode': 'plain'}) 

        (self.t, self.I1, self.I2, self.I3, self.I, self.mB, self.omega, self.gamma, self.xi) = sp.symbols('t I1 I2 I3 I mB omega gamma xi', real=True)
        t, I1, I2, I3, I, mB, omega, gamma, xi = (self.t, self.I1, self.I2, self.I3, self.I, self.mB, self.omega, self.gamma, self.xi)
        (theta, phi, psi) = sp.symbols('theta phi psi', cls=sp.Function)

        th_dd, ph_dd, ps_dd = sp.symbols('th_dd ph_dd ps_dd', real=True)

        th = theta(t); ph = phi(t); ps = psi(t)

        th_d = sp.diff(th, t); ph_d = sp.diff(ph, t); ps_d = sp.diff(ps, t)

        self.th, self.ph, self.ps, self.th_d, self.ph_d, self.ps_d, self.th_dd, self.ph_dd, self.ps_dd = (th, ph, ps, th_d, ph_d, ps_d,th_dd, ph_dd, ps_dd)

        # Rotation matrix ZXZ
        self.R = R = Rz(self.ph) * Rx(self.th) * Rz(self.ps)

        # skew-symmetric operator
        vee = lambda M: sp.Matrix([M[2,1], M[0,2], M[1,0]])

        # World-frame angular velocity ν = vee( Ṙ · Rᵀ )
        self.nu_wo = vee(sp.simplify(sp.diff(R, t) * R.T))  # not used currently in this derivation

        # Body-frame angular velocity ν = vee( Rᵀ · Ṙ )
        self.nu_bo = vee(sp.simplify(R.T * sp.diff(R, t)))

        # Rayleigh dissipation
        R_diss = sp.Rational(1,2) * self.xi * self.nu_bo.dot(self.nu_bo)

        # 4) Magnetic and inertia tensors
        self.um = um = R * sp.Matrix([0,0,1])
        self.uB = uB = sp.Matrix([
            cos(gamma)*sin(omega*t),
           -cos(gamma)*cos(omega*t),
            sin(gamma)
        ])

        ## ───────── use stepped rotation ───────────
        ##Gilbert Strang, Introduction to Linear Algebra
        ## 1. Section on “Congruence transformations” (Chap 6) treats 
        ## 2. Matrix Cookbook by Kaare Brandt Petersen and Michael Syskind Pedersen
        ##    See the entry on “Orthogonal congruence and diagonalization,”
        #I_ps = Rz(self.ps) * sp.diag(I1, I2, I3) * Rz(self.ps).T
        #I_ps = I_ps.applyfunc(lambda e: sp.collect(e,sin(ps)*cos(ps)))
        #I_th = Rx(self.th) * I_ps * Rx(self.th).T
        #I_th = sp.expand_trig(sp.simplify(I_th))
        #I_rot  = Rz(self.ph) * I_th * Rz(self.ph).T
        ##I_r = I_r.applyfunc(lambda e: sp.trigsimp(sp.expand(e), method='fu'))
        #I_rot = I_rot.applyfunc(lambda e: sp.trigsimp(e, method='fu'))
        # ─────────────────────────────────────────────

        #I_r = sp.simplify(R * sp.diag(I1, I2, I3) * R.T)
        #custom simplification of I_rot
        #I_rot = I_r.as_mutable()
        #I_rot = I_rot.applyfunc(lambda e: sp.expand(e))
        #I_rot = I_rot.applyfunc(lambda e: sp.trigsimp(e, method='combined'))
        #x = I_rot[1,1]
        #I_rot[1,1] = sp.simplify(sp.collect(x.expand(), sin(th)**2))
        #x = I_rot[0,0]
        #I_rot[0,0] = sp.collect(x,sin(ph)**2*sin(th)**2)
        #self.I_rot = I_rot

        ## 5) Kinetic & potential energies
        x = (self.nu_bo.T * sp.diag(I1, I2, I3) * self.nu_bo)[0]
        self.T = sp.Rational(1,2) * sp.collect(sp.simplify(x),[I1,I2,I3])
        self.V = -mB * sp.simplify(um.dot(uB))

        # 6) Euler–Lagrange + damping
        coords  = [th, ph, ps]
        dcoords = [th_d, ph_d, ps_d]
        Qd      = [-sp.diff(R_diss, dqi) for dqi in dcoords]

        eqs = []
        for (q, dq), Qdi in zip(zip(coords, dcoords), Qd):
            dL_dqdot = sp.diff(self.T - self.V, dq)
            EL       = sp.diff(dL_dqdot, t) - sp.diff(self.T - self.V, q)
            eqs.append(sp.simplify(EL - Qdi))

        # 7) Solve for second derivatives
        subs_dd = {
            sp.diff(th, t, 2): th_dd,
            sp.diff(ph, t, 2): ph_dd,
            sp.diff(ps, t, 2): ps_dd
        }

        #self.eqs_sub = [eq.subs(subs_dd) for eq in eqs]

        self.eqs_sub = []
        for eq in eqs:
            x = eq.subs(subs_dd)
            x = sp.expand_trig(x)
            x = sp.cancel(x)
            x = sp.collect(x, [self.I1, self.I2, self.I3, self.mB, self.xi])
            self.eqs_sub.append(x)

        self.symbols_dd = [th_dd, ph_dd, ps_dd]

        A, b = sp.linear_eq_to_matrix(self.eqs_sub, self.symbols_dd)

        # Solve with LU‐factorization
        print("solving equations...", end='')
        self.dd_sol = A.LUsolve(b)
        print(" done.")

        # simplify each solution term
        _simplify()

        # Build solution dict
        sol = dict(zip(self.symbols_dd, self.dd_sol))

        # Lambdify RHS functions, JIT compile with Numba
        print("lambdify...", end='')
        syms = (th, ph, ps, th_d, ph_d, ps_d, I1, I2, I3, mB, omega, gamma, xi, t)
        self.f_th = numba.njit(sp.lambdify(syms, sol[th_dd], 'numpy'))
        self.f_ph = numba.njit(sp.lambdify(syms, sol[ph_dd], 'numpy'))
        self.f_ps = numba.njit(sp.lambdify(syms, sol[ps_dd], 'numpy'))
        print(" done.")

        self.eom = [Eq(sym * I1, sol[sym] * I1) for sym in self.symbols_dd]
        # ─────────────────────────────────────────────
        sol_axisym_iso = {
            s: sp.collect(
                sp.collect(
                    sp.simplify(sol[s].subs({I2: I1})).subs(1/tan(th), cos(th)/sin(th)),
                      [mB, xi]),
                  [I1, I3]) * I1
            for s in self.symbols_dd
        }
        x = sol_axisym_iso[ps_dd]
        x = sp.expand(x)
        x = sp.expand_trig(x)
        x = x.collect([mB, xi, th_d * ph_d])
        x = x.collect(-I1/I3)
        #x = x.collect([sin(th),1/sin(th)])
        x = x.subs(sin(omega*t) * cos(ph) - cos(omega*t) * sin(ph), sin(omega*t - ph))
        sol_axisym_iso[ps_dd] = x
        self.eom_axisym = [Eq(sym * I1, sol_axisym_iso[sym]) for sym in self.symbols_dd]
        # ─────────────────────────────────────────────
        self.sol_iso = {
            s: sp.collect(sp.simplify(sol[s].subs({I3: I, I2: I, I1: I})).subs(1/tan(th), cos(th)/sin(th)), [mB, xi])
            for s in self.symbols_dd
        }
        self.sol_iso[ph_dd] = self.sol_iso[ph_dd].collect(I*th_d)
        self.sol_iso[ps_dd] = sp.simplify(self.sol_iso[ps_dd]*sin(th))/sin(th)
        self.eom_iso = [Eq(sym * I, self.sol_iso[sym] * I) for sym in self.symbols_dd]
        # ─────────────────────────────────────────────
        print("derivation completed.")
