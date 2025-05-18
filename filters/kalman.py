# --- kalman_spec.py ---
"""
KalmanSpec Quick Guide

This class defines the structure of a Kalman filter model, including the latent state evolution,
initialization, and noise structures. It supports composable modeling via method chaining.

### Key Concepts:

- The latent state βₜ holds time-varying exposures to predictors (e.g., factors, yields, credit spreads).
- Q is the process noise covariance: it controls how much we allow βₜ to evolve each step.
  - High Q → flexible exposures that can change rapidly
  - Low Q → stable exposures that change slowly
- T is the transition matrix: defines the evolution rule for βₜ
  - Default is identity (random walk)
  - Can be AR(1), mean-reverting, or custom
- R is the observation noise variance: reflects how noisy the returns are
  - High R → we trust the return signal less
  - Can be constant or time-varying (e.g. adaptive to volatility)

### Core Usage:

spec = (
    KalmanSpec(K=4)
    .estimate_initial_state(H, y)              # OLS estimate for beta_0 and P_0
    .add_intercept(intercept_Q_scale=1e-5)     # Adds time-varying intercept alpha
    .add_AR1(phi=0.95)                         # AR(1) drift for exposures
    .set_Q_from_factor_vols(H)                 # Process noise scaled to factor volatility
    .add_adaptive_R(R_series)                  # Time-varying observation noise
)

### Available Modifiers:

- .describe(): returns dictionary with structure, dimensionality, time-variation, etc.
- .summary_str(): prints a formatted description of the model

All modifier functions include shape checks and warnings to protect against common mistakes.
"""
import numpy as np
import copy
import numpy as np
import pandas as pd

class KalmanSpec:
    def __init__(self, K, Q=None, R=None, name="KalmanSpec"):
        self.name = name
        self.K = K
        self.beta_0 = np.zeros((K, 1))
        self.P_0 = np.eye(K)
        self.init_method = "default"

        self.transition_fn = lambda t: np.eye(K)
        self.process_noise_fn = lambda t: Q if Q is not None else np.eye(K) * 1e-5
        self.observation_fn = lambda t, H_t: H_t.reshape(1, self.K)
        self.obs_noise_fn = lambda t: R if R is not None else np.array([[1e-4]])

    def copy(self):
        return copy.deepcopy(self)

    def validate(self, y, H):
        assert y.ndim == 2 and y.shape[1] == 1, "y must be a column vector"
        assert H.shape[0] == y.shape[0], "H and y must have the same number of rows"
        assert H.shape[1] == self.K or H.shape[1] == self.K - 1, f"H must have {self.K} or {self.K - 1} columns"

    def _inspect_R(self):
        try:
            R_sample = [self.obs_noise_fn(i)[0, 0] for i in range(10)]
            is_var = len(set(np.round(R_sample, 10))) > 1
            if is_var:
                return True, f"time-varying (sample: {R_sample[:3]} ... {R_sample[-3:]})"
            return False, self.obs_noise_fn(0).flatten().tolist()
        except:
            return False, "unreadable"

    def _inspect_Q(self):
        try:
            Q_sample = [np.diag(self.process_noise_fn(i)).tolist() for i in range(10)]
            is_var = any(np.any(np.abs(np.array(Q_sample[0]) - np.array(q)) > 1e-10) for q in Q_sample[1:])
            if is_var:
                return True, f"time-varying (diag sample: {Q_sample[:3]} ... {Q_sample[-3:]})"
            return False, np.diag(self.process_noise_fn(0)).tolist()
        except:
            return False, "unreadable"

    def _inspect_T(self):
        try:
            T_sample = [self.transition_fn(i) for i in range(5)]
            is_var = any(not np.allclose(T_sample[0], T_sample[i]) for i in range(1, 5))
            if is_var:
                return True, f"time-varying (T₀ sample shown) {T_sample[0].tolist()}"
            return False, T_sample[0].tolist()
        except:
            return False, "unreadable"

    def describe(self, target_col=None, factor_cols=None):
        is_R_var, R_val = self._inspect_R()
        is_Q_var, Q_val = self._inspect_Q()
        is_T_var, T_val = self._inspect_T()

        T0 = self.transition_fn(0)

        return {
            "model_name": self.name,
            "state_dim": self.K,
            "target_col": target_col,
            "factor_cols": factor_cols,
            "initial_beta_0": self.beta_0.flatten().tolist(),
            "initial_P_0_diag": np.diag(self.P_0).tolist(),
            "Q": Q_val,
            "R": R_val,
            "T": T_val,
            "initialization": self.init_method,
            "has_intercept": self.K > 1 and np.allclose(self.beta_0[0], 0),
            "is_adaptive_R": is_R_var,
            "is_AR1": "AR1" in self.name or np.allclose(T0, T0[0, 0] * np.eye(self.K)),
            "AR1_phi": T0[0, 0] if np.allclose(T0, T0[0, 0] * np.eye(self.K)) else None,
            "Q_scale": float(np.mean(np.diag(self.process_noise_fn(0)))) if not is_Q_var else "time-varying",
            "R_scale": float(self.obs_noise_fn(0)[0, 0]) if not is_R_var else "time-varying",
            "transition_matrix_T_0": T0.tolist()
        }

    def summary_str(self):
        desc = self.describe()
        return "\n".join([f"{k}: {v}" for k, v in desc.items()])

    def add_intercept(self, intercept_Q_scale=1e-5):
        old_K = self.K
        self.K += 1

        # Expand beta_0 and P_0
        self.beta_0 = np.vstack([np.zeros((1, 1)), self.beta_0])
        self.P_0 = np.block([
            [np.ones((1, 1)), np.zeros((1, old_K))],
            [np.zeros((old_K, 1)), self.P_0]
        ])

        # Expand transition_fn
        prev_T_fn = self.transition_fn
        self.transition_fn = lambda t: np.block([
            [np.array([[1]]), np.zeros((1, old_K))],
            [np.zeros((old_K, 1)), prev_T_fn(t)]
        ])

        # Expand process_noise_fn
        prev_Q_fn = self.process_noise_fn
        self.process_noise_fn = lambda t: np.block([
            [np.array([[intercept_Q_scale]]), np.zeros((1, old_K))],
            [np.zeros((old_K, 1)), prev_Q_fn(t)]
        ])

        # Expand observation_fn
        self.observation_fn = lambda t, H_t: np.hstack([np.ones(1), H_t]).reshape(1, self.K)
        self.name += "_with_intercept"
        return self

    def add_adaptive_R(self, R_series):
        self.obs_noise_fn = lambda t: np.array([[R_series[t]]])
        self.name += "_with_adaptive_R"
        return self

    def set_custom_Q(self, Q_matrix):
        if Q_matrix.shape != (self.K, self.K):
            print("Warning: custom Q shape does not match state dimension. Did you add an intercept after computing Q?")
        self.process_noise_fn = lambda t: Q_matrix
        self.name += "_custom_Q"
        return self

    def add_AR1(self, phi=0.95):
        try:
            T_current = self.transition_fn(0)
            if T_current.shape != (self.K, self.K):
                print("Warning: AR1 update may overwrite a custom or expanded transition matrix. Please ensure this is intentional.")
        except:
            print("Warning: unable to inspect current transition matrix shape. Proceed with caution.")
        self.transition_fn = lambda t: np.eye(self.K) * phi
        self.name += f"_AR1_phi{phi:.2f}"
        return self

    def set_custom_T(self, T_fn):
        try:
            test_T = T_fn(0)
            if test_T.shape != (self.K, self.K):
                print("Warning: custom T shape does not match state dimension. Did you add an intercept?")
        except:
            print("Warning: unable to verify custom T dimensions. Proceed with caution.")
        self.transition_fn = T_fn
        self.name += "_custom_T"
        return self

    def set_Q_from_factor_vols(self, H):
        factor_vols = np.var(H, axis=0)
        Q = np.diag(factor_vols * 1e-4)
        return self.set_custom_Q(Q)

    def estimate_initial_state(self, H, y, method="ols"):
        if method == "ols":
            beta_ols = np.linalg.lstsq(H, y, rcond=None)[0]
            residuals = y - H @ beta_ols
            sigma2 = np.var(residuals)
            self.beta_0 = beta_ols
            self.P_0 = np.eye(self.K) * sigma2
            self.init_method = "ols"
        else:
            raise NotImplementedError(f"Unknown init method: {method}")
        return self

    def estimate_R_from_ols(self, H, y):
        beta_ols = np.linalg.lstsq(H, y, rcond=None)[0]
        residuals = y - H @ beta_ols
        R = np.var(residuals)
        self.obs_noise_fn = lambda t: np.array([[R]])
        self.name += "_init_R_ols"
        return self

class KalmanEngine:
    def __init__(self, spec):
        self.spec = spec

    def _log_likelihood(self, innovation, S):
        """Compute Gaussian log-likelihood for a single timestep."""
        return -0.5 * (
            np.log(2 * np.pi)
            + np.log(np.linalg.det(S))
            + (innovation.T @ np.linalg.inv(S) @ innovation)[0, 0]
        )

    def run(self, df, target_col, factor_cols, burn=0):
        y = df[[target_col]].values
        H = df[factor_cols].values
        index = df.index

        self.spec.validate(y, H)
        T_steps, K = len(y), self.spec.K

        beta_preds = np.zeros((T_steps, K))
        beta_covs = []
        y_preds = np.zeros(T_steps)
        innovations = np.zeros(T_steps)
        kalman_gains = np.zeros((T_steps, K))
        log_likelihoods = np.zeros(T_steps)

        beta_prev = self.spec.beta_0
        P_prev = self.spec.P_0

        for t in range(T_steps):
            H_t = H[t]
            y_t = y[t].reshape(1, 1)
            T_t = self.spec.transition_fn(t)
            Q_t = self.spec.process_noise_fn(t)
            H_obs_t = self.spec.observation_fn(t, H_t)
            R_t = self.spec.obs_noise_fn(t)

            # Prediction step
            beta_pred = T_t @ beta_prev
            P_pred = T_t @ P_prev @ T_t.T + Q_t

            # Measrurement step
            y_hat = H_obs_t @ beta_pred
            innovation = y_t - y_hat
            S = H_obs_t @ P_pred @ H_obs_t.T + R_t
            K_t = P_pred @ H_obs_t.T @ np.linalg.inv(S)

            # Update step
            beta_post = beta_pred + K_t @ innovation
            P_post = (np.eye(K) - K_t @ H_obs_t) @ P_pred

            # Store results and calculate log-likelihood
            log_likelihoods[t] = self._log_likelihood(innovation, S)

            beta_preds[t] = beta_post.flatten()
            beta_covs.append(P_post)
            y_preds[t] = y_hat
            innovations[t] = innovation
            kalman_gains[t] = K_t.flatten()

            beta_prev = beta_post
            P_prev = P_post

        # --- Slice burn-in period ---
        idx = slice(burn, None)
        time_index = df.index[burn:]
        cols = factor_cols

        return {
            "beta": pd.DataFrame(beta_preds[idx], index=time_index, columns=cols),
            "beta_cov": dict(zip(time_index, beta_covs[burn:])),
            "y_pred": pd.Series(y_preds[idx], index=time_index, name="y_pred"),
            "residuals": pd.Series(innovations[idx], index=time_index, name="residuals"),
            "kalman_gain": pd.DataFrame(kalman_gains[idx], index=time_index, columns=cols),
            "log_likelihood": np.sum(log_likelihoods[burn:]),
            "log_likelihood_t": pd.Series(log_likelihoods[burn:], index=time_index, name="log_likelihood"),
            "meta": self.spec.describe(target_col=target_col, factor_cols=factor_cols),
            "H": pd.DataFrame(H[idx], index=time_index, columns=factor_cols)
        }

class KalmanMLEOptimizer:
    def __init__(self, spec_template, y, H, burn=0):
        self.template = spec_template
        self.y = y
        self.H = H
        self.burn = burn

    def objective(self, params):
        q_scale, r_scale = params
        spec = self.template.copy()
        spec.set_custom_Q(np.eye(spec.K) * q_scale)
        spec.obs_noise_fn = lambda t: np.array([[r_scale]])
        engine = KalmanEngine(spec)
        result = engine.run(self.y, self.H, burn=self.burn)
        return -result["log_likelihood"]  # minimize negative log-likelihood

    def fit(self, bounds=((1e-7, 1e-2), (1e-7, 1e-2))):
        from scipy.optimize import minimize
        init = [1e-5, 1e-4]
        result = minimize(self.objective, init, bounds=bounds, method='L-BFGS-B')
        return result

