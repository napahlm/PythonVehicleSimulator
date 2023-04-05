#!/usr/bin/env/python3
# -*- coding: utf-8 -*-

"""
testvessel.py
    A modified copy of supply.py
    
    The purpose is to test other thrust allocation methods.
"""

import numpy as np
from python_vehicle_simulator.lib.control import DPpolePlacement
from python_vehicle_simulator.lib.gnc import sat

class testvessel:
    """
    Call:
    testvessel()
    testvessel('DPcontrol', x_d, y_d, psi_d, V_c, beta_c)

    Inputs:
        x_d: desired x position (m)
        y_d: desired y position (m)
        psi_d: desired yaw angle (deg)
        V_c: current speed (m/s)
        beta_c: current direction (deg)
    """

    def __init__(
        self,
        controlSystem="stepInput",
        r_x = 0,
        r_y = 0,
        r_n = 0,
        V_current = 0,
        beta_current = 0,
    ):
        
        # Constants
        D2R = np.pi / 180
        g   = 9.81

        if controlSystem == "DPcontrol":
            self.controlDescription = (
                "Nonlinear DP control (x_d, y_d, psi_d) = ("
                + str(r_x)
                + " m, "
                + str(r_y)
                + " m, "
                + str(r_n)
                + " deg)"
            )
        else:
            self.controlDescription = "Step inputs for n = [n1, n2, n3, n4]"

        self.ref         = np.array([r_x, r_y, r_n*D2R], float)
        self.V_c         = V_current
        self.beta_c      = beta_current * D2R
        self.controlMode = controlSystem

        # Vessel Model
        m               = 6000.0e3  # mass (kg)
        self.L          = 76.2      # Length (m)
        self.W          = 18        # Width (m)
        self.T_n        = 1.0       # prop. rev. time constant (s)
        self.n_max      = np.array([250, 250, 160, 160], float) # RPM saturation limits (N)
        self.nu         = np.array([0, 0, 0, 0, 0, 0], float)   # velocity vector
        self.u_actual   = np.array([0, 0, 0, 0], float)         # RPM inputs
        self.name       = "Test Vessel"

        # Two tunnel thrusters in the bow, no. 1 and 2
        # Two main propellers aft, no. 3 and 4
        self.controls = [
            "#1 Bow thruster (RPM)",
            "#2 Bow thruster (RPM)",
            "#3 Right main propeller (RPM)",
            "#4 Left main propeller (RPM)",
        ]
        self.dimU = len(self.controls)

        # Propulsion configuration
        Ke = np.diag([2.4, 2.4, 17.6, 17.6])
        Te = np.array([[ 0,  0,         1,        1],
                       [ 1,  1,         0,        0],
                       [30, 22, -self.W/2, self.W/2]], float)
        self.B = Te @ Ke

        # 3-DOF model matrices - bis scaling (Fossen 2021, App. D)
        Tbis_inv = np.diag([1.0, 1.0, self.L])
        Mbis = np.array([[1.1274,       0,       0],
                         [     0,  1.8902, -0.0744],
                         [     0, -0.0744,  0.1278]], float)
        Dbis = np.array([[0.0358,       0,       0],
                         [     0,  0.1183, -0.0124],
                         [     0, -0.0041,  0.0308]], float)
        
        self.M3    = m * Tbis_inv @ Mbis @ Tbis_inv
        self.M3inv = np.linalg.inv(self.M3)
        self.D3    = m * np.sqrt(g / self.L) * Tbis_inv @ Dbis @ Tbis_inv

        # DP Control System
        self.e_int = np.array([0, 0, 0], float)  # integral states
        self.x_d   = 0.0  # setpoints
        self.y_d   = 0.0
        self.psi_d = 0.0
        self.wn    = np.diag([0.3, 0.3, 0.1])  # PID pole placement
        self.zeta  = np.diag([1.0, 1.0, 1.0])

    def dynamics(self, eta, nu, u_actual, u_control, sampleTime):
        
        # Input
        n = u_actual

        # Current velocities
        u_c = self.V_c * np.cos(self.beta_c - eta[5])   # current surge velocity
        v_c = self.V_c * np.sin(self.beta_c - eta[5])   # current sway velocity

        nu_c = np.array([u_c, v_c, 0, 0, 0, 0], float)  # current velocity vector
        nu_r = nu - nu_c                                # relative velocity vector

        # Control Forces
        n_squared = np.zeros(self.dimU)
        for i in range(0, self.dimU):
            n[i] = sat(
                n[i], -self.n_max[i], self.n_max[i]
            )  # saturation, physical limits
            n_squared[i] = abs(n[i]) * n[i]

        tau3 = np.matmul(self.B, n_squared)

        # 3 DOF dynamics
        nu3_r   = np.array([nu_r[0], nu_r[1], nu_r[5]])
        nu3_dot = np.matmul(self.M3inv, tau3 - np.matmul(self.D3, nu3_r))

        # 6 DOF ship model
        nu_dot = np.array([nu3_dot[0], nu3_dot[1], 0, 0, 0, nu3_dot[2]])
        n_dot = (u_control - u_actual) / self.T_n

        # Forward Euler Integration
        nu = nu + sampleTime * nu_dot
        n  = n + sampleTime * n_dot

        u_actual = np.array(n, float)

        return nu, u_actual
    
    def controlAllocation(self, tau3):
        """
        u_alloc  = controlAllocation(tau3),  tau3 = [tau_X, tau_Y, tau_N]'
        u_alloc = B' * inv( B * B' ) * tau3
        """
        B_pseudoInv = self.B.T @ np.linalg.inv(self.B @ self.B.T)
        u_alloc = np.matmul(B_pseudoInv, tau3)

        return u_alloc

    def DPcontrol(self, eta, nu, sampleTime):

        eta3 = np.array([eta[0], eta[1], eta[5]])
        nu3  = np.array([nu[0], nu[1], nu[5]])

        [tau3, self.e_int, self.x_d, self.y_d, self.psi_d] = DPpolePlacement(
            self.e_int,
            self.M3,
            self.D3,
            eta3,
            nu3,
            self.x_d,
            self.y_d,
            self.psi_d,
            self.wn,
            self.zeta,
            self.ref,
            sampleTime,
        )

        u_alloc = self.controlAllocation(tau3)

        n = np.zeros(self.dimU)
        for i in range(0, self.dimU):
            n[i] = np.sign(u_alloc[i]) * np.sqrt(np.abs(u_alloc[i]))

        u_control = n

        return u_control
    
    def stepInput(self, t):
        """
        u = stepInput(t) generates propeller step inputs (RPM).
        """
        n = np.array([0, 0, 100, 100], float)

        if t > 30:
            n = np.array([50, 50, 50, 50], float)
        if t > 70:
            n = np.array([0, 0, 0, 0], float)

        u_control = n

        return u_control