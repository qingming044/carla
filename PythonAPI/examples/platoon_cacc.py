#!/usr/bin/env python

# Copyright (c) 2025 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
CARLA 0.9.15 – 3-Vehicle Connected Automated Platoon Simulation
================================================================
Architecture
------------
  WorldManager       – CARLA client / world / sync-mode lifecycle
  VehicleAgent       – per-vehicle state, sensors, actuators
  DistributedKF      – Distributed Kalman Filter (DKF) for state estimation
  CACCController     – Cooperative Adaptive Cruise Control (CACC)
  InverseDynamics    – maps desired acceleration → throttle / brake
  PlatoonSimulation  – orchestrates agents, V2X comms, main loop, logging

Run
---
  python platoon_cacc.py [--host HOST] [--port PORT] [--duration TICKS]

Assumptions / notes
-------------------
* CARLA 0.9.15 Python API is assumed to be on PYTHONPATH.
* ``carla.VehiclePhysicsControl`` exposes ``mass``, ``drag_coefficient``,
  and a list of ``carla.WheelPhysicsControl`` objects accessible via the
  ``wheels`` attribute.  Wheel radius is in centimetres in the CARLA API.
* Longitudinal velocity is obtained from ``vehicle.get_velocity()`` via 1D
  projection onto the forward vector.
* Longitudinal acceleration is estimated as a finite-difference of velocity
  (smoothed by the DKF); CARLA 0.9.15 does not expose an acceleration getter
  directly on actors, so we compute it from successive velocity readings.
* Lateral keep-lane is handled by a simple proportional steer correction.
* Logging writes one CSV row per tick to ``platoon_log.csv`` in the working
  directory.
"""

import argparse
import csv
import math
import sys
import time

try:
    import numpy as np
except ImportError:
    raise RuntimeError("numpy is required: pip install numpy")

try:
    import carla
except ImportError:
    raise RuntimeError(
        "carla module not found – make sure the CARLA PythonAPI egg/wheel "
        "is on your PYTHONPATH"
    )

# ---------------------------------------------------------------------------
# Simulation constants
# ---------------------------------------------------------------------------
DT = 0.01          # fixed time-step [s]
TAU = 0.35         # first-order actuator delay [s]
TARGET_GAP = 20.0  # desired inter-vehicle gap [m]
WARMUP_TICKS = 20  # Phase-0 ticks

# Ramp-up / leader cruise
V_MAX = 12.0       # maximum platoon speed [m/s]
V_RAMP_RATE = 1.0  # m/s per simulated second

# CACC gains
K_P = 1.0
K_D = 1.5
K_F = 1.0

# Noise standard deviations
SIGMA_P = 0.05
SIGMA_V = 0.05
SIGMA_A = 0.10
SIGMA_GAP = 0.10

# Vehicle physics
VEHICLE_MASS = 1529.98       # kg
DRAG_COEFF = 0.28
WHEEL_RADIUS_CM = 33.0       # cm (CARLA API unit)

# Inverse dynamics
RHO = 1.206    # air density [kg/m³]
CD = DRAG_COEFF
FRONTAL_AREA = 2.51  # m²
ROLL_RESIST = 0.005  # rolling-resistance coefficient
G_ACCEL = 9.81       # m/s²
F_MAX_THROTTLE = 6000.0  # [N] – full throttle force equivalent
F_MAX_BRAKE = 8000.0     # [N] – full brake force equivalent

# DKF covariance scalars
Q_SCALAR = 0.1
R_DIAG = [SIGMA_P**2, SIGMA_V**2, SIGMA_A**2, SIGMA_GAP**2]
R_N_DIAG = [SIGMA_P**2, SIGMA_V**2, SIGMA_A**2]

# ---------------------------------------------------------------------------
# Helper – build DKF system matrices
# ---------------------------------------------------------------------------

def _build_system_matrices():
    """Return (A, B, G, H, H_n, Q_mat, R_mat, R_n_mat, R_bar_mat)."""
    A = np.array([
        [1.0, DT,  0.0],
        [0.0, 1.0, DT],
        [0.0, 0.0, 1.0 - DT / TAU],
    ])
    B = np.array([[0.0], [0.0], [DT / TAU]])
    G = np.array([[0.0], [0.0], [1.0]])

    # Local observation matrix (4 × 3)
    H = np.array([
        [1.0,  0.0, 0.0],  # position
        [0.0,  1.0, 0.0],  # velocity
        [0.0,  0.0, 1.0],  # acceleration
        [-1.0, 0.0, 0.0],  # relative gap (self term)
    ])

    # Neighbour observation matrix (4 × 3)
    H_n = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],  # relative gap (front-vehicle term)
    ])

    Q_mat = Q_SCALAR * G @ G.T                           # (3×3)
    R_mat = np.diag(R_DIAG)                              # (4×4)
    R_n_mat = np.diag(R_N_DIAG)                          # (3×3)
    R_bar_mat = R_mat + H_n @ R_n_mat @ H_n.T            # (4×4)

    return A, B, G, H, H_n, Q_mat, R_mat, R_n_mat, R_bar_mat


# Pre-compute shared matrices (immutable, shared across all DKF instances)
_A, _B, _G, _H, _H_n, _Q, _R, _R_n, _R_bar = _build_system_matrices()


# ---------------------------------------------------------------------------
# DistributedKF
# ---------------------------------------------------------------------------

class DistributedKF:
    """
    Distributed Kalman Filter for a single follower vehicle.

    State:  x = [p, v, a]^T

    The filter fuses:
      - Local noisy measurements: (p_meas, v_meas, a_meas, gap_meas)
      - Neighbour (front vehicle) filtered state: Y_n = [p_n, v_n, a_n]^T

    Attributes
    ----------
    x_hat : np.ndarray, shape (3,1)
        Current state estimate.
    P : np.ndarray, shape (3,3)
        Current estimation error covariance.
    innovation : np.ndarray, shape (4,1)
        Last computed innovation (residual) – logged for GLR detection.
    """

    def __init__(self):
        self.x_hat = np.zeros((3, 1))
        self.P = np.eye(3) * 1.0
        self.innovation = np.zeros((4, 1))

    # ------------------------------------------------------------------
    def initialize_state(self, p0: float, v0: float, a0: float):
        """
        Seed the filter with known initial conditions.
        Call once after Phase-0 ground-truth is available to avoid
        large startup innovations.
        """
        self.x_hat = np.array([[p0], [v0], [a0]])
        self.P = np.eye(3) * 0.01

    # ------------------------------------------------------------------
    def predict(self, u_input: float):
        """
        Time-update (prediction) step.

        Parameters
        ----------
        u_input : float
            Control input for the *self* vehicle (used in state propagation).
        """
        u_vec = np.array([[u_input]])
        self.x_hat = _A @ self.x_hat + _B @ u_vec
        self.P = _A @ self.P @ _A.T + _Q

    # ------------------------------------------------------------------
    def update(
        self,
        p_meas: float,
        v_meas: float,
        a_meas: float,
        gap_meas: float,
        Y_n: np.ndarray,
    ):
        """
        Measurement-update step.

        Parameters
        ----------
        p_meas, v_meas, a_meas : float
            Noisy local GNSS / IMU measurements of own state.
        gap_meas : float
            Noisy radar gap measurement (front_pos - self_pos).
        Y_n : np.ndarray, shape (3,1)
            Neighbour (front vehicle) filtered state [p_n, v_n, a_n].
        """
        y = np.array([[p_meas], [v_meas], [a_meas], [gap_meas]])

        # Innovation: residual between measurement and prediction
        y_pred = _H @ self.x_hat + _H_n @ Y_n
        self.innovation = y - y_pred

        # Innovation covariance
        S = _H @ self.P @ _H.T + _R_bar
        # Kalman gain
        K = self.P @ _H.T @ np.linalg.inv(S)
        # State update
        self.x_hat = self.x_hat + K @ self.innovation
        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(3) - K @ _H
        self.P = I_KH @ self.P @ I_KH.T + K @ _R_bar @ K.T

    # ------------------------------------------------------------------
    @property
    def state(self) -> np.ndarray:
        """Return current state estimate as (3,1) array."""
        return self.x_hat

    @property
    def p_est(self) -> float:
        return float(self.x_hat[0, 0])

    @property
    def v_est(self) -> float:
        return float(self.x_hat[1, 0])

    @property
    def a_est(self) -> float:
        return float(self.x_hat[2, 0])


# ---------------------------------------------------------------------------
# CACCController
# ---------------------------------------------------------------------------

class CACCController:
    """
    Cooperative Adaptive Cruise Control (CACC).

    PD + feedforward structure (no integral term for string stability).

    u_self = k_p * err_p + k_d * err_v + k_f * u_front

    where
      err_p = p_front - p_self - TARGET_GAP
      err_v = v_front - v_self
    """

    def __init__(self, k_p: float = K_P, k_d: float = K_D, k_f: float = K_F):
        self.k_p = k_p
        self.k_d = k_d
        self.k_f = k_f

    def compute(
        self,
        p_self: float,
        v_self: float,
        p_front: float,
        v_front: float,
        u_front: float,
    ) -> float:
        """Return desired acceleration command [m/s²]."""
        err_p = p_front - p_self - TARGET_GAP
        err_v = v_front - v_self
        return self.k_p * err_p + self.k_d * err_v + self.k_f * u_front


# ---------------------------------------------------------------------------
# InverseDynamics
# ---------------------------------------------------------------------------

class InverseDynamics:
    """
    Maps a desired longitudinal acceleration [m/s²] to CARLA
    throttle / brake commands in [0, 1].

    F_total = M * u_desired
              + 0.5 * rho * Cd * A_front * v²
              + M * g * f_roll

    throttle = clamp(F_total / F_MAX_THROTTLE, 0, 1)   if F_total > 0
    brake    = clamp(|F_total| / F_MAX_BRAKE,  0, 1)   if F_total ≤ 0
    """

    def __init__(self, mass: float = VEHICLE_MASS):
        self.mass = mass

    def compute(self, u_desired: float, v_current: float):
        """
        Return (throttle, brake) tuple, each in [0, 1].

        Parameters
        ----------
        u_desired : float
            Target acceleration [m/s²].
        v_current : float
            Current longitudinal velocity [m/s].
        """
        v = abs(v_current)
        aero_drag = 0.5 * RHO * CD * FRONTAL_AREA * v * v
        roll_drag = self.mass * G_ACCEL * ROLL_RESIST
        f_total = self.mass * u_desired + aero_drag + roll_drag

        if f_total > 0.0:
            throttle = float(np.clip(f_total / F_MAX_THROTTLE, 0.0, 1.0))
            brake = 0.0
        else:
            throttle = 0.0
            brake = float(np.clip(abs(f_total) / F_MAX_BRAKE, 0.0, 1.0))

        return throttle, brake


# ---------------------------------------------------------------------------
# VehicleAgent
# ---------------------------------------------------------------------------

class VehicleAgent:
    """
    Wraps a CARLA vehicle actor with 1D projection kinematics,
    virtual sensor noise, DKF (followers only), CACC, and inverse dynamics.

    Parameters
    ----------
    vehicle : carla.Vehicle
        The spawned CARLA vehicle actor.
    role : str
        One of 'leader', 'follower1', 'follower2'.
    origin_loc : carla.Location
        Reference origin for 1D projection.
    fwd_vec : carla.Vector3D
        Forward unit vector of the road at the reference origin.
    """

    def __init__(
        self,
        vehicle,
        role: str,
        origin_loc,
        fwd_vec,
    ):
        self.vehicle = vehicle
        self.role = role
        self.origin_loc = origin_loc
        self.fwd_vec = fwd_vec

        # Ground truth (updated each tick)
        self.p_true = 0.0
        self.v_true = 0.0
        self.a_true = 0.0
        self._prev_v_true = 0.0  # for finite-diff acceleration

        # Noisy measurements
        self.p_meas = 0.0
        self.v_meas = 0.0
        self.a_meas = 0.0

        # Last applied control acceleration
        self.u_cmd = 0.0

        # DKF (only for followers)
        if role in ('follower1', 'follower2'):
            self.dkf = DistributedKF()
        else:
            self.dkf = None

        # CACC (only for followers)
        if role in ('follower1', 'follower2'):
            self.cacc = CACCController()
        else:
            self.cacc = None

        # Inverse dynamics (all vehicles)
        self.inv_dyn = InverseDynamics()

        # Lateral reference: store initial lateral offset
        # Used by the micro-adjustment steer PD controller
        self._lateral_ref = None
        self._prev_lateral_err = 0.0

    # ------------------------------------------------------------------
    # Kinematics helpers
    # ------------------------------------------------------------------

    def _dot(self, vec) -> float:
        """Dot product of a CARLA Vector3D with the road forward vector."""
        return (
            vec.x * self.fwd_vec.x
            + vec.y * self.fwd_vec.y
            + vec.z * self.fwd_vec.z
        )

    def _lateral_dot(self, vec) -> float:
        """
        Signed lateral offset (cross-product z component in 2-D).
        lat = fwd_x * vec_y - fwd_y * vec_x
        """
        return self.fwd_vec.x * vec.y - self.fwd_vec.y * vec.x

    def update_ground_truth(self):
        """
        Pull velocity (and position) from CARLA and project onto 1D axis.
        Acceleration is estimated via finite difference.
        """
        loc = self.vehicle.get_location()
        vel = self.vehicle.get_velocity()
        transform = self.vehicle.get_transform()

        # Longitudinal 1-D position
        dloc = carla.Vector3D(
            loc.x - self.origin_loc.x,
            loc.y - self.origin_loc.y,
            loc.z - self.origin_loc.z,
        )
        self.p_true = self._dot(dloc)

        # Longitudinal 1-D velocity
        self.v_true = self._dot(vel)

        # Longitudinal 1-D acceleration (finite difference, updated each tick)
        self.a_true = (self.v_true - self._prev_v_true) / DT
        self._prev_v_true = self.v_true

        # Lateral offset relative to the stored reference
        if self._lateral_ref is None:
            self._lateral_ref = self._lateral_dot(dloc)

    def add_noise(self):
        """Apply zero-mean Gaussian noise to ground-truth to form measurements."""
        rng = np.random
        self.p_meas = self.p_true + rng.normal(0.0, SIGMA_P)
        self.v_meas = self.v_true + rng.normal(0.0, SIGMA_V)
        self.a_meas = self.a_true + rng.normal(0.0, SIGMA_A)

    def gap_measurement(self, front_agent: "VehicleAgent") -> float:
        """
        Return noisy radar gap measurement relative to front_agent.

        The gap is based on *ground-truth* positions (radar measures
        distance, not absolute position), then noise is added.
        """
        gap_true = front_agent.p_true - self.p_true
        return gap_true + np.random.normal(0.0, SIGMA_GAP)

    # ------------------------------------------------------------------
    # State getter (filtered for followers, noisy for leader)
    # ------------------------------------------------------------------

    def get_broadcast_state(self):
        """
        Return (p, v, a) for V2X broadcast.

        - Leader: noisy own measurements.
        - Follower 1 & 2: DKF-filtered state (NEVER raw noisy).
        """
        if self.dkf is not None:
            return (self.dkf.p_est, self.dkf.v_est, self.dkf.a_est)
        else:
            return (self.p_meas, self.v_meas, self.a_meas)

    # ------------------------------------------------------------------
    # DKF
    # ------------------------------------------------------------------

    def dkf_initialize(self):
        """Seed the DKF with current ground-truth to avoid startup transients."""
        if self.dkf is not None:
            self.dkf.initialize_state(self.p_true, self.v_true, self.a_true)

    def dkf_step(self, front_agent: "VehicleAgent", front_broadcast_state):
        """
        Run one DKF predict+update cycle for a follower.

        Parameters
        ----------
        front_agent : VehicleAgent
            The vehicle directly ahead (used for gap measurement).
        front_broadcast_state : tuple
            (p_n, v_n, a_n) – the front vehicle's broadcast state.
            For F1 this is leader's noisy state; for F2 this is F1's
            *filtered* state.
        """
        if self.dkf is None:
            return

        gap_meas = self.gap_measurement(front_agent)
        Y_n = np.array([[front_broadcast_state[0]],
                        [front_broadcast_state[1]],
                        [front_broadcast_state[2]]])

        self.dkf.predict(self.u_cmd)
        self.dkf.update(
            self.p_meas, self.v_meas, self.a_meas, gap_meas, Y_n
        )

    # ------------------------------------------------------------------
    # CACC + inverse dynamics
    # ------------------------------------------------------------------

    def compute_leader_control(self, current_time: float) -> float:
        """
        Ramp-up speed-tracking P controller for the leader.

        Returns desired acceleration [m/s²].
        """
        v_desired = min(V_MAX, current_time * V_RAMP_RATE)
        u_L = 2.0 * (v_desired - self.v_true)
        return u_L

    def compute_follower_control(self, front_broadcast_state, front_u: float) -> float:
        """
        CACC for followers, using DKF-filtered self state and front state.

        Returns desired acceleration [m/s²].
        """
        if self.cacc is None:
            return 0.0

        p_self = self.dkf.p_est
        v_self = self.dkf.v_est
        p_front = front_broadcast_state[0]
        v_front = front_broadcast_state[1]

        return self.cacc.compute(p_self, v_self, p_front, v_front, front_u)

    def apply_control(self, u_desired: float):
        """
        Convert u_desired → CARLA VehicleControl and apply it.
        Includes a simple lateral steer micro-adjustment.
        """
        self.u_cmd = u_desired
        throttle, brake = self.inv_dyn.compute(u_desired, self.v_true)

        # Lateral micro-adjustment (proportional + derivative on lateral error)
        steer = self._compute_lateral_steer()

        ctrl = carla.VehicleControl(
            throttle=throttle,
            brake=brake,
            steer=float(np.clip(steer, -1.0, 1.0)),
            hand_brake=False,
            reverse=False,
            manual_gear_shift=False,
        )
        self.vehicle.apply_control(ctrl)

    def _compute_lateral_steer(self) -> float:
        """
        Simple PD lateral steer correction to keep the vehicle centred on
        its spawn-time lateral reference line.
        """
        loc = self.vehicle.get_location()
        dloc = carla.Vector3D(
            loc.x - self.origin_loc.x,
            loc.y - self.origin_loc.y,
            loc.z - self.origin_loc.z,
        )
        lat_current = self._lateral_dot(dloc)
        if self._lateral_ref is None:
            self._lateral_ref = lat_current

        lat_err = lat_current - self._lateral_ref
        k_lat_p = 0.5
        k_lat_d = 0.1
        d_err = lat_err - self._prev_lateral_err
        steer = -(k_lat_p * lat_err + k_lat_d * d_err / DT)
        self._prev_lateral_err = lat_err
        return steer


# ---------------------------------------------------------------------------
# WorldManager
# ---------------------------------------------------------------------------

class WorldManager:
    """
    Manages the CARLA client connection, world loading, synchronous mode,
    and vehicle spawning / cleanup.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 2000):
        self.host = host
        self.port = port
        self.client = None
        self.world = None
        self._original_settings = None
        self._actors = []

    # ------------------------------------------------------------------
    def connect(self):
        """Connect to the CARLA server and load Town06."""
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(30.0)
        print(f"[WorldManager] Connected to CARLA server at {self.host}:{self.port}")

        # Load Town06
        self.world = self.client.load_world("Town06")
        print("[WorldManager] Loaded Town06")

    # ------------------------------------------------------------------
    def enable_sync_mode(self):
        """Switch the simulator to synchronous mode with DT = 0.01 s."""
        settings = self.world.get_settings()
        self._original_settings = settings
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = DT
        self.world.apply_settings(settings)
        print(f"[WorldManager] Sync mode enabled (fixed_delta_seconds={DT})")

    # ------------------------------------------------------------------
    def disable_sync_mode(self):
        """Restore original simulator settings."""
        if self._original_settings is not None and self.world is not None:
            self.world.apply_settings(self._original_settings)
            print("[WorldManager] Sync mode disabled – original settings restored")

    # ------------------------------------------------------------------
    def spawn_vehicles(self):
        """
        Spawn three Tesla Model 3 vehicles (red) and return them as a
        list [leader_agent, follower1_agent, follower2_agent].

        Spawn positions (relative to spawn_points[4] forward direction):
          Leader    : origin + 40 m * fwd_vec
          Follower 1: origin + 20 m * fwd_vec
          Follower 2: origin
        """
        bp_lib = self.world.get_blueprint_library()
        vehicle_bp = bp_lib.find("vehicle.tesla.model3")

        # Set colour to red
        if vehicle_bp.has_attribute("color"):
            red_attrs = [a for a in vehicle_bp.get_attribute("color").recommended_values
                         if "255,0,0" in a or a == "255,0,0"]
            if red_attrs:
                vehicle_bp.set_attribute("color", red_attrs[0])
            else:
                # Fallback: set directly
                vehicle_bp.set_attribute("color", "255,0,0")

        spawn_points = self.world.get_map().get_spawn_points()
        if len(spawn_points) <= 4:
            raise RuntimeError(
                f"Town06 has only {len(spawn_points)} spawn points; need at least 5."
            )

        origin_tf = spawn_points[4]
        origin_loc = origin_tf.location
        fwd = origin_tf.get_forward_vector()

        def offset_transform(metres: float) -> carla.Transform:
            new_loc = carla.Location(
                x=origin_loc.x + fwd.x * metres,
                y=origin_loc.y + fwd.y * metres,
                z=origin_loc.z + fwd.z * metres + 0.5,  # slight Z lift to avoid ground clip
            )
            return carla.Transform(new_loc, origin_tf.rotation)

        spawn_info = [
            ("leader",    40.0),
            ("follower1", 20.0),
            ("follower2",  0.0),
        ]

        agents = []
        for role, dist in spawn_info:
            tf = offset_transform(dist)
            actor = self.world.try_spawn_actor(vehicle_bp, tf)
            if actor is None:
                raise RuntimeError(
                    f"Failed to spawn {role} at offset {dist} m. "
                    "Try a different spawn point index or check for collisions."
                )
            self._actors.append(actor)
            self._apply_physics(actor)
            agent = VehicleAgent(actor, role, origin_loc, fwd)
            agents.append(agent)
            print(f"[WorldManager] Spawned {role} (id={actor.id}) at offset={dist:.1f}m")

        # Tick once to let the physics engine settle before we read ground truth
        self.world.tick()

        return agents  # [leader, follower1, follower2]

    # ------------------------------------------------------------------
    @staticmethod
    def _apply_physics(vehicle):
        """Overwrite vehicle physics parameters per the spec."""
        phys = vehicle.get_physics_control()
        phys.mass = VEHICLE_MASS
        phys.drag_coefficient = DRAG_COEFF

        # Update wheel radii (CARLA stores radius in cm)
        new_wheels = []
        for w in phys.wheels:
            w.radius = WHEEL_RADIUS_CM
            new_wheels.append(w)
        phys.wheels = new_wheels

        vehicle.apply_physics_control(phys)

    # ------------------------------------------------------------------
    def cleanup(self):
        """Destroy all spawned actors and restore world settings."""
        self.disable_sync_mode()
        for actor in self._actors:
            if actor.is_alive:
                actor.destroy()
        self._actors.clear()
        print("[WorldManager] All actors destroyed")

    # ------------------------------------------------------------------
    def tick(self):
        """Advance the simulation by one fixed time step."""
        return self.world.tick()


# ---------------------------------------------------------------------------
# PlatoonSimulation
# ---------------------------------------------------------------------------

class PlatoonSimulation:
    """
    Top-level orchestrator for the 3-vehicle CACC platoon.

    Phases
    ------
    Phase 0 (ticks 0 … WARMUP_TICKS-1): warm-up
        Physics settles; DKF seeded; no control applied.
    Phase 1 (ticks WARMUP_TICKS … duration-1): closed-loop platooning
        Leader ramps up; F1/F2 follow via DKF + CACC + inverse dynamics.

    V2X data flow
    -------------
    Leader  → broadcasts noisy (p_L, v_L, a_L) and u_L
    F1      → receives leader broadcast; computes DKF; broadcasts *filtered* (p̂, v̂, â)
    F2      → receives F1 filtered broadcast; computes DKF; applies CACC
    """

    def __init__(self, world_mgr: WorldManager, duration_ticks: int = 5000):
        self.world_mgr = world_mgr
        self.duration_ticks = duration_ticks
        self.agents = []        # [leader, f1, f2]
        self.tick_count = 0
        self._log_rows = []
        self._csv_path = "platoon_log.csv"

    # ------------------------------------------------------------------
    def setup(self):
        """Connect, load world, spawn vehicles."""
        self.world_mgr.connect()
        self.world_mgr.enable_sync_mode()
        self.agents = self.world_mgr.spawn_vehicles()

    # ------------------------------------------------------------------
    def run(self):
        """Execute the main simulation loop."""
        leader, f1, f2 = self.agents
        tick = 0
        current_time = 0.0

        try:
            while tick < self.duration_ticks:
                # Advance the world
                self.world_mgr.tick()
                tick += 1
                current_time = tick * DT

                # ----------------------------------------------------------
                # Pull ground truth from CARLA for all vehicles
                # ----------------------------------------------------------
                for agent in self.agents:
                    agent.update_ground_truth()
                    agent.add_noise()

                # ----------------------------------------------------------
                # Phase 0: warm-up
                # ----------------------------------------------------------
                if tick <= WARMUP_TICKS:
                    if tick == WARMUP_TICKS:
                        # Seed DKF with current ground truth
                        for agent in self.agents:
                            agent.dkf_initialize()
                        print(f"[Platoon] Warm-up complete at tick {tick}")
                    # No control during warm-up
                    self._log(tick, current_time, leader, f1, f2,
                              u_leader=0.0, u_f1=0.0, u_f2=0.0)
                    continue

                # ----------------------------------------------------------
                # Phase 1: leader ramp-up + closed-loop followers
                # ----------------------------------------------------------

                # Leader control
                u_leader = leader.compute_leader_control(current_time)
                leader.apply_control(u_leader)

                # Leader V2X broadcast (noisy own state + control command)
                leader_broadcast = leader.get_broadcast_state()   # (p, v, a)

                # ------- Follower 1 -------
                # Receive leader broadcast as neighbour measurement
                f1.dkf_step(
                    front_agent=leader,
                    front_broadcast_state=leader_broadcast,
                )
                u_f1 = f1.compute_follower_control(
                    front_broadcast_state=leader_broadcast,
                    front_u=u_leader,
                )
                f1.apply_control(u_f1)

                # F1 V2X broadcast – FILTERED state only
                f1_broadcast = f1.get_broadcast_state()           # DKF-filtered

                # ------- Follower 2 -------
                # Receive F1 filtered broadcast as neighbour measurement
                f2.dkf_step(
                    front_agent=f1,
                    front_broadcast_state=f1_broadcast,
                )
                u_f2 = f2.compute_follower_control(
                    front_broadcast_state=f1_broadcast,
                    front_u=u_f1,
                )
                f2.apply_control(u_f2)

                # Logging
                self._log(tick, current_time, leader, f1, f2, u_leader, u_f1, u_f2)

                # Console heartbeat every 100 ticks
                if tick % 100 == 0:
                    gap_f1 = leader.p_true - f1.p_true
                    gap_f2 = f1.p_true - f2.p_true
                    print(
                        f"[t={current_time:6.2f}s] "
                        f"v_L={leader.v_true:5.2f}  "
                        f"v_F1={f1.v_true:5.2f}  "
                        f"v_F2={f2.v_true:5.2f}  "
                        f"gap_F1={gap_f1:5.2f}m  "
                        f"gap_F2={gap_f2:5.2f}m"
                    )

        except KeyboardInterrupt:
            print("\n[Platoon] Interrupted by user")
        finally:
            self._flush_log()

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    _LOG_HEADER = [
        "tick", "time_s",
        # Leader
        "L_p_true", "L_v_true", "L_a_true",
        "L_p_meas", "L_v_meas", "L_a_meas",
        "L_u",
        # Follower 1
        "F1_p_true", "F1_v_true", "F1_a_true",
        "F1_p_meas", "F1_v_meas", "F1_a_meas",
        "F1_p_est",  "F1_v_est",  "F1_a_est",
        "F1_inn0",   "F1_inn1",   "F1_inn2",   "F1_inn3",
        "F1_u",
        # Follower 2
        "F2_p_true", "F2_v_true", "F2_a_true",
        "F2_p_meas", "F2_v_meas", "F2_a_meas",
        "F2_p_est",  "F2_v_est",  "F2_a_est",
        "F2_inn0",   "F2_inn1",   "F2_inn2",   "F2_inn3",
        "F2_u",
    ]

    def _log(self, tick, t, L, F1, F2, u_leader, u_f1, u_f2):
        """Append one row to the in-memory log buffer."""
        inn_f1 = F1.dkf.innovation.flatten() if F1.dkf else [0.0]*4
        inn_f2 = F2.dkf.innovation.flatten() if F2.dkf else [0.0]*4

        row = [
            tick, f"{t:.4f}",
            L.p_true,  L.v_true,  L.a_true,
            L.p_meas,  L.v_meas,  L.a_meas,
            u_leader,
            F1.p_true, F1.v_true, F1.a_true,
            F1.p_meas, F1.v_meas, F1.a_meas,
        ]
        if F1.dkf:
            row += [F1.dkf.p_est, F1.dkf.v_est, F1.dkf.a_est]
        else:
            row += [0.0, 0.0, 0.0]
        row += list(inn_f1) + [u_f1]

        row += [
            F2.p_true, F2.v_true, F2.a_true,
            F2.p_meas, F2.v_meas, F2.a_meas,
        ]
        if F2.dkf:
            row += [F2.dkf.p_est, F2.dkf.v_est, F2.dkf.a_est]
        else:
            row += [0.0, 0.0, 0.0]
        row += list(inn_f2) + [u_f2]

        self._log_rows.append(row)

    def _flush_log(self):
        """Write buffered log rows to CSV."""
        if not self._log_rows:
            return
        try:
            with open(self._csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(self._LOG_HEADER)
                writer.writerows(self._log_rows)
            print(f"[Platoon] Log written to '{self._csv_path}' ({len(self._log_rows)} rows)")
        except OSError as e:
            print(f"[Platoon] WARNING: Could not write log: {e}")

    # ------------------------------------------------------------------
    def teardown(self):
        """Clean up CARLA actors and restore world settings."""
        self.world_mgr.cleanup()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="CARLA 0.9.15 – 3-Vehicle CACC Platoon Simulation"
    )
    parser.add_argument("--host", default="127.0.0.1", help="CARLA server host")
    parser.add_argument("--port", type=int, default=2000, help="CARLA server port")
    parser.add_argument(
        "--duration",
        type=int,
        default=5000,
        help="Number of simulation ticks to run (default: 5000 = 50 s)",
    )
    args = parser.parse_args()

    world_mgr = WorldManager(host=args.host, port=args.port)
    sim = PlatoonSimulation(world_mgr=world_mgr, duration_ticks=args.duration)

    try:
        sim.setup()
        sim.run()
    finally:
        sim.teardown()


if __name__ == "__main__":
    main()
