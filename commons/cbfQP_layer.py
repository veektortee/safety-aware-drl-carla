import numpy as np
import osqp
import scipy.sparse as sp

class CBFSafetyLayer:
    """
    Control Barrier Function (CBF) Safety Layer
    Solves a small QP to minimally modify the actor's action
    when trust score is low.
    """

    def __init__(
        self,
        alpha=1.0,
        d_min=5.0,        # minimum safe distance (meters)
        y_max=1.5,        # max lane deviation (meters)
        v_max=15.0        # max speed (m/s)
    ):
        self.alpha = alpha
        self.d_min = d_min
        self.y_max = y_max
        self.v_max = v_max

        # OSQP solver instance
        self.solver = osqp.OSQP()

        # Action dimension: [steer, throttle, brake]
        self.u_dim = 3

    def compute_safe_action(self, u_actor, state):
        """
        Args:
            u_actor: np.array shape (3,)
            state: dict with keys:
                - d_collision
                - ttc
                - lane_offset
                - speed
        Returns:
            u_safe: np.array shape (3,)
        """

        # -----------------------------
        # Cost: ||u_safe - u_actor||²
        # -----------------------------
        P = sp.eye(self.u_dim)
        q = -u_actor

        # -----------------------------
        # Barrier Constraints: A u ≥ b
        # -----------------------------
        A = []
        b = []

        # ---- 1. Collision Avoidance ----
        h_col = state["d_collision"] - self.d_min

        # ∂h/∂u approximation
        A_col = np.array([0.0, -1.0, 2.0])   # throttle ↓, brake ↑
        b_col = -self.alpha * h_col

        A.append(A_col)
        b.append(b_col)

        # ---- 2. Lane Keeping ----
        h_lane = self.y_max - abs(state["lane_offset"])
        steer_dir = -np.sign(state["lane_offset"])  # correct direction

        A_lane = np.array([steer_dir, 0.0, 0.0])
        b_lane = -self.alpha * h_lane

        A.append(A_lane)
        b.append(b_lane)

        # ---- 3. Speed Limit ----
        h_speed = self.v_max - state["speed"]

        A_speed = np.array([0.0, -1.0, 1.0])  # throttle ↓, brake ↑
        b_speed = -self.alpha * h_speed

        A.append(A_speed)
        b.append(b_speed)

        # Stack constraints
        A = np.vstack(A)
        b = np.array(b)

        # Convert to OSQP format: l ≤ A u ≤ ∞
        A_osqp = sp.csc_matrix(A)
        l = b
        u = np.full(len(b), np.inf)

        # -----------------------------
        # Setup & Solve QP
        # -----------------------------
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

        if res.info.status != "solved":
            # Fallback: emergency braking
            return np.array([0.0, 0.0, 1.0])

        return res.x
