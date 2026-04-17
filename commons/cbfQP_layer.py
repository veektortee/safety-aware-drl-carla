import numpy as np
import osqp
import scipy.sparse as sp

class CBFSafetyLayer:
    """
    Enhanced Control Barrier Function (CBF) Safety Layer
    Solves a small QP to minimally modify the actor's action
    with support for:
    - Dynamic speed-dependent Lie derivatives
    - Trust score modulation
    - Steering/throttle rate limiting
    - Dynamic speed limit constraint
    - Proactive lane keeping
    - Constraint violation fallback verification
    """

    def __init__(
        self,
        alpha=1.0,
        d_min=5.0,        # minimum safe distance (meters)
        y_max=1.5,        # max lane deviation (meters)
        v_max=15.0,       # default max speed (m/s)
        vehicle_width=1.8, # vehicle width (meters)
        v_nominal=10.0,   # nominal speed for Lie deriv scaling (m/s)
        max_steering_rate=0.5,  # max steering change per step (rad)
        max_accel_change=0.3,   # max throttle/brake change per step
        alpha_lane=0.5,   # CBF alpha for lane keeping (softer than collision)
    ):
        self.alpha = alpha
        self.alpha_lane = alpha_lane  # Softer correction for lane keeping
        self.d_min = d_min
        self.y_max = y_max
        self.v_max = v_max
        self.vehicle_width = vehicle_width
        self.v_nominal = v_nominal
        self.max_steering_rate = max_steering_rate
        self.max_accel_change = max_accel_change

        # OSQP solver instance
        self.solver = osqp.OSQP()

        # Action dimension: [steer, throttle, brake]
        self.u_dim = 3
        
        # Previous action for rate limiting
        self.u_prev = np.array([0.0, 0.0, 0.0])
        
        # Tracking metrics
        self.correction_count = 0
        self.constraint_violations = {'collision': 0, 'lane': 0, 'speed': 0}
        self.correction_magnitude = 0.0  # L2-norm of latest action correction
        
        # Safety metrics (for reward shaping)
        self.collision_prevented = False
        self.avoidance_efficiency = 0.0  # 0-1: how close to constraint violation

    def _apply_rate_limiting(self, u_raw: np.ndarray) -> np.ndarray:
        """Apply steering and throttle rate limits for smooth corrections"""
        u_limited = u_raw.copy()
        
        # Steering rate limit
        max_steer_change = self.max_steering_rate
        u_limited[0] = np.clip(
            u_limited[0],
            self.u_prev[0] - max_steer_change,
            self.u_prev[0] + max_steer_change
        )
        
        # Throttle/brake rate limit (combined)
        accel = u_limited[1] - u_limited[2]  # throttle - brake
        accel_prev = self.u_prev[1] - self.u_prev[2]
        accel_limited = np.clip(
            accel,
            accel_prev - self.max_accel_change,
            accel_prev + self.max_accel_change
        )
        
        # Split back into throttle/brake (zero out the decreasing one)
        if accel_limited > 0:
            u_limited[1] = accel_limited
            u_limited[2] = 0.0
        else:
            u_limited[1] = 0.0
            u_limited[2] = -accel_limited
        
        return u_limited

    def _get_speed_dependent_lie_derivatives(self, state: dict) -> tuple:
        """
        Compute speed-dependent Lie derivatives for collision and speed constraints
        
        Returns:
            (A_col, A_speed): Lie derivative vectors for collision and speed constraints
        """
        speed = state.get('speed', 0.5)
        speed = max(speed, 0.1)  # Avoid division by zero
        
        # Speed scaling factor: more effective at nominal speed
        speed_scale = speed / self.v_nominal if speed > 0.1 else 0.1
        
        # Collision avoidance: throttle effect increases with speed
        throttle_effect = -1.0 * speed_scale
        # Brake effect moderates with speed (saturates at high speed)
        brake_effect = 2.0 * min(1.0, speed_scale)
        A_col = np.array([0.0, throttle_effect, brake_effect])
        
        # Speed limit: same as collision (throttle ↓, brake ↑)
        A_speed = np.array([0.0, -1.0 * speed_scale, 1.0 * min(1.0, speed_scale)])
        
        return A_col, A_speed

    def _compute_proactive_lane_offset(self, state: dict, action: np.ndarray) -> float:
        """
        Predict future lane offset using steering action
        Proactive mitigation: constraint on predicted offset, not just current
        """
        current_offset = abs(state.get('lane_offset', 0.0))
        speed = state.get('speed', 0.5)
        
        # Heuristic: steering-to-yaw rate (Ackermann model approximation)
        # yaw_rate ≈ (speed / wheelbase) * tan(steering)
        # For small angles: yaw_rate ≈ (speed / wheelbase) * steering
        wheelbase = 2.7  # CARLA standard
        
        if speed > 0.5:
            steering_action = action[0]
            dt = 0.05  # 50ms per step at 20 FPS
            yaw_rate = (speed / wheelbase) * steering_action
            offset_delta = yaw_rate * dt
            predicted_offset = current_offset + offset_delta
        else:
            predicted_offset = current_offset
        
        return predicted_offset

    def compute_safe_action(
        self,
        u_actor: np.ndarray,
        state: dict,
        trust_score: float = 1.0,
        verbose: bool = False
    ) -> np.ndarray:
        """
        Compute safe action by solving QP with CBF constraints
        
        Args:
            u_actor: np.array shape (3,) - [steering, throttle, brake] from actor
            state: dict with keys:
                - d_collision: distance to nearest obstacle (meters)
                - ttc: time-to-collision (seconds)
                - lane_offset: lateral distance from lane center (meters)
                - speed: current vehicle speed (m/s)
                - speed_limit: dynamic speed limit (m/s), optional
                - yaw_rate: current yaw rate (rad/s), optional
            trust_score: float in [0, 1] - ensemble confidence
                High trust (1.0) → normal correction
                Low trust (0.0) → conservative (scale down correction)
            verbose: bool - Enable detailed logging of constraints and solver
        
        Returns:
            u_safe: np.array shape (3,) - safe action after corrections
        """

        # Store for rate limiting
        self.u_prev_stored = self.u_prev.copy()

        # =============================================
        # Cost: minimize ||u_safe - u_actor||²
        # =============================================
        P = sp.eye(self.u_dim)
        q = -u_actor

        # =============================================
        # Barrier Constraints: A u ≥ b
        # =============================================
        A = []
        b = []

        # ---- 1. Collision Avoidance ----
        h_col = state["d_collision"] - self.d_min
        A_col, _ = self._get_speed_dependent_lie_derivatives(state)
        
        # Predict: would uncorrected action violate collision constraint?
        # u_actor projected on A_col: A_col · u_actor
        uncorrected_h_dot = np.dot(A_col, u_actor)
        predicted_h_col_uncorrected = h_col + uncorrected_h_dot  # Next step with u_actor
        
        # Track if collision would occur without correction
        will_collide_uncorrected = predicted_h_col_uncorrected < 0
        self.collision_prevented = False  # Reset; set true if we fix it
        
        b_col = -self.alpha * h_col
        
        A.append(A_col)
        b.append(b_col)
        if h_col < 0:
            self.constraint_violations['collision'] += 1
            if verbose:
                print(f"[CBF-COLLISION] VIOLATION! h_col={h_col:.4f}")
                print(f"  d_collision={state['d_collision']:.2f}m, d_min={self.d_min}m")

        # ---- 2. Lane Keeping (Gradual Correction) ----
        # Use predicted offset for proactive constraint
        predicted_offset = self._compute_proactive_lane_offset(state, u_actor)
        h_lane = self.y_max - predicted_offset
        
        # Proportional steering: magnitude scales with offset
        current_offset = state.get('lane_offset', 0.0)
        offset_sign = np.sign(current_offset)
        if offset_sign == 0:
            offset_sign = 1.0
        
        # Proportional gain: larger offset → stronger correction
        # Scaled between 0.2 (small offset) to 1.0 (at boundary)
        offset_magnitude = abs(current_offset)
        proportional_gain = np.clip(offset_magnitude / self.y_max, 0.2, 1.0)
        
        steer_dir = -offset_sign * proportional_gain  # gradual correction direction & magnitude
        
        A_lane = np.array([steer_dir, 0.0, 0.0])
        b_lane = -self.alpha_lane * h_lane  # Use softer alpha for gradual correction
        
        A.append(A_lane)
        b.append(b_lane)
        if h_lane < 0:
            self.constraint_violations['lane'] += 1

        # ---- 3. Speed Limit (Dynamic) ----
        # Use dynamic speed limit from state, or fall back to self.v_max
        speed_limit = state.get('speed_limit', self.v_max)
        h_speed = speed_limit - state["speed"]
        
        _, A_speed = self._get_speed_dependent_lie_derivatives(state)
        b_speed = -self.alpha * h_speed
        
        A.append(A_speed)
        b.append(b_speed)
        if h_speed < 0:
            self.constraint_violations['speed'] += 1

        # Stack constraints
        A = np.vstack(A)
        b = np.array(b)

        # Convert to OSQP format: l ≤ A u ≤ ∞
        A_osqp = sp.csc_matrix(A)
        l = b
        u = np.full(len(b), np.inf)

        # =============================================
        # Setup & Solve QP
        # =============================================
        try:
            self.solver.setup(
                P=P,
                q=q,
                A=A_osqp,
                l=l,
                u=u,
                verbose=False,
                polish=True
            )

            res = self.solver.solve()
            
            if verbose:
                print(f"[CBF-SOLVER] Status: '{res.info.status}' | Constraints: {len(A)} | Iter: {res.info.iter}")

            if res.info.status != "solved":
                if verbose:
                    print(f"[CBF-WARNING] Solver status not 'solved' - using fallback action")
                # Fallback: verify and use previous action
                u_safe = self._fallback_safe_action(state)
            else:
                u_safe = res.x
        except Exception as e:
            print(f"[CBF] QP solver exception: {e}")
            u_safe = self._fallback_safe_action(state)

        # =============================================
        # Apply Rate Limiting
        # =============================================
        u_safe = self._apply_rate_limiting(u_safe)

        # =============================================
        # Trust Score Modulation
        # =============================================
        if trust_score < 1.0:
            uncertainty_factor = 0.3 * (1.0 - trust_score)
            # Conservative blend: keep more of fallback when uncertain
            fallback = np.array([0.0, 0.0, 1.0])  # emergency brake
            u_safe = (1.0 - uncertainty_factor) * u_safe + uncertainty_factor * fallback

        # Clip to valid action range
        u_safe = np.clip(u_safe, -1.0, 1.0)

        # =============================================
        # Check if Collision Was Prevented
        # =============================================
        if will_collide_uncorrected:
            # Uncorrected action would cause collision; did our correction prevent it?
            corrected_h_dot = np.dot(A_col, u_safe)
            predicted_h_col_corrected = h_col + corrected_h_dot
            if predicted_h_col_corrected >= 0:
                self.collision_prevented = True
                # Efficiency: how far above constraint are we?
                self.avoidance_efficiency = min(predicted_h_col_corrected / self.d_min, 1.0)
            else:
                self.collision_prevented = False
                self.avoidance_efficiency = 0.0
        else:
            # No collision threat; no prevention needed
            self.collision_prevented = False
            self.avoidance_efficiency = max(0.0, min(h_col / self.d_min, 1.0))  # margin metric

        # =============================================
        # Track Correction Magnitude
        # =============================================
        correction_mag = float(np.linalg.norm(u_safe - u_actor))
        self.correction_magnitude = correction_mag  # Store for later retrieval
        if correction_mag > 0.01:  # Only count non-negligible corrections
            self.correction_count += 1

        # Update previous action for next step
        self.u_prev = u_safe.copy()

        return u_safe

    def _fallback_safe_action(self, state: dict) -> np.ndarray:
        """
        Generate fallback action when QP solver fails
        Verify constraints before returning
        """
        # Try: maintain previous action (safest: no sudden changes)
        u_fallback = self.u_prev.copy()
        
        # Verify against critical constraints
        violations = self._check_constraint_violations(u_fallback, state)
        
        if violations['collision'] or violations['lane'] or violations['speed']:
            # If previous action violates, apply emergency brake cautiously
            u_fallback = np.array([0.0, 0.0, 0.5])  # Moderate brake (not max)
        
        return u_fallback

    def _check_constraint_violations(self, action: np.ndarray, state: dict) -> dict:
        """Check if an action violates safety constraints (simplified check)"""
        violations = {'collision': False, 'lane': False, 'speed': False}
        
        # Simplified forward dynamics: assume action effects are immediate
        # In practice, this is a heuristic check
        
        # Collision: throttle increases collision risk
        if state["d_collision"] < self.d_min and action[1] > 0.1:
            violations['collision'] = True
        
        # Lane: steering away from lane center increases risk
        lane_offset = state.get('lane_offset', 0.0)
        if abs(lane_offset) > self.y_max and action[0] * np.sign(lane_offset) > 0.1:
            violations['lane'] = True
        
        # Speed: throttle increases speed
        speed_limit = state.get('speed_limit', self.v_max)
        if state["speed"] > speed_limit and action[1] > 0.1:
            violations['speed'] = True
        
        return violations

    def reset_metrics(self):
        """Reset tracking metrics for new episode"""
        self.correction_count = 0
        self.constraint_violations = {'collision': 0, 'lane': 0, 'speed': 0}
        self.correction_magnitude = 0.0
        self.u_prev = np.array([0.0, 0.0, 0.0])
        
        # Reset safety metrics
        self.collision_prevented = False
        self.avoidance_efficiency = 0.0

    def get_correction_magnitude(self):
        """Get the magnitude of the latest action correction"""
        return self.correction_magnitude
