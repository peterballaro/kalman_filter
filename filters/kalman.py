import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Optional, Union, List, Dict, Callable

def estimate_mle_R(spec_template, df: pd.DataFrame, target_col: str, factor_cols: list[str], burn: int = 0,
                    method: str = 'L-BFGS-B', bounds: tuple = (1e-6, 1e2)) -> tuple[float, minimize]:
    def neg_log_likelihood(log_R_scalar):
        R_val = np.exp(log_R_scalar)
        spec = spec_template.copy()
        spec.obs_noise_fn = lambda t: np.array([[R_val]])
        engine = KalmanEngine(spec)
        results = engine.run(df, target_col=target_col, factor_cols=factor_cols, burn=burn)
        return -results["log_likelihood"]

    result = minimize(
        neg_log_likelihood,
        x0=np.log(1e-2),
        bounds=[(np.log(bounds[0]), np.log(bounds[1]))],
        method=method
    )

    best_log_R = result.x[0]
    best_R = np.exp(best_log_R)
    return best_R, result


def estimate_mle_Q(spec_template, df: pd.DataFrame, target_col: str, factor_cols: list[str], burn: int = 0,
                    method: str = 'L-BFGS-B', bounds: tuple = (1e-8, 1e1)) -> tuple[float, minimize]:
    def neg_log_likelihood(log_Q_scalar):
        Q_val = np.exp(log_Q_scalar)
        spec = spec_template.copy()
        spec.process_noise_fn = lambda t: np.eye(spec.K) * Q_val
        engine = KalmanEngine(spec)
        results = engine.run(df, target_col=target_col, factor_cols=factor_cols, burn=burn)
        return -results["log_likelihood"]

    result = minimize(
        neg_log_likelihood,
        x0=np.log(1e-2),
        bounds=[(np.log(bounds[0]), np.log(bounds[1]))],
        method=method
    )

    best_log_Q = result.x[0]
    best_Q = np.exp(best_log_Q)
    return best_Q, result


class KalmanSpec:
    def __init__(self, K: int, name: str = "Kalman Model"):
        self.K = K
        self.name = name
        self.beta_0 = np.zeros((K, 1))
        self.P_0 = np.eye(K) * 1e2
        self.transition_fn: Callable[[int], np.ndarray] = lambda t: np.eye(self.K)
        self.process_noise_fn: Callable[[int], np.ndarray] = lambda t: np.eye(self.K) * 1e-4
        self.observation_fn: Callable[[int, np.ndarray], np.ndarray] = lambda t, H_t: H_t.reshape(1, self.K)
        self.obs_noise_fn: Callable[[int], np.ndarray] = lambda t: np.eye(1) * 1e-2
        self.has_intercept: bool = False

    def set_intercept(self, init_val: float = 0.0) -> 'KalmanSpec':
        self.K += 1
        self.beta_0 = np.insert(self.beta_0, 0, init_val).reshape(-1, 1)
        self.P_0 = np.pad(self.P_0, ((1, 0), (1, 0)), mode="constant")
        self.P_0[0, 0] = 1e6
        self.has_intercept = True
        self.observation_fn = lambda t, H_t: np.hstack([np.ones(1), H_t]).reshape(1, self.K)
        return self

    def set_initial_state_from_ols(self, H: pd.DataFrame, y: pd.Series) -> 'KalmanSpec':
        H_mat = H.copy()
        if self.has_intercept:
            H_mat = pd.concat([pd.Series(1.0, index=H.index, name="Intercept"), H], axis=1)
        beta_ols = np.linalg.pinv(H_mat.values) @ y.values
        self.beta_0 = beta_ols.reshape(-1, 1)
        return self

    def set_Q_from_factor_vols(self, H: pd.DataFrame) -> 'KalmanSpec':
        Q = np.diag(H.var().values)
        if self.has_intercept:
            Q = np.pad(Q, ((1, 0), (1, 0)), mode="constant")
        self.process_noise_fn = lambda t: Q
        return self
    
    def set_Q_from_rolling_beta_var(
        self,
        df: pd.DataFrame,
        target_col: str,
        factor_cols: list[str],
        window: int = 20,
        min_val: float = 1e-8,
        scale: float = 1.0
    ) -> 'KalmanSpec':
        """
        Set Q_t using rolling OLS beta variance.
        Useful for tracking recent instability in factor exposures.
        """
        y = df[target_col]
        H = df[factor_cols]

        if self.has_intercept:
            H = pd.concat([pd.Series(1.0, index=H.index, name="Intercept"), H], axis=1)

        beta_hist = []
        for t in range(len(df)):
            if t < window:
                beta_hist.append(np.full(H.shape[1], np.nan))
                continue
            H_window = H.iloc[t - window:t].values
            y_window = y.iloc[t - window:t].values
            beta = np.linalg.pinv(H_window) @ y_window
            beta_hist.append(beta)

        beta_df = pd.DataFrame(beta_hist, index=df.index, columns=H.columns)
        beta_var = beta_df.rolling(window).var().clip(lower=min_val).bfill()

        Q_list = [np.diag(row.values) * scale for _, row in beta_var.iterrows()]
        self.process_noise_fn = lambda t: Q_list[min(t, len(Q_list) - 1)]
        self.meta = getattr(self, "meta", {})
        self.meta["Q_mode"] = f"rolling_beta_var_window_{window}"
        return self

    def set_Q_from_rolling_residual_vol(
        self,
        df: pd.DataFrame,
        target_col: str,
        factor_cols: list[str],
        window: int = 20,
        scale: float = 1.0,
        burn: int = 0,
        min_val: float = 1e-8
    ) -> 'KalmanSpec':
        """
        Runs a Kalman filter with the current spec, collects residuals,
        then builds a time-varying Q_t matrix using rolling residual variance.

        Parameters:
            df: full DataFrame
            target_col: column name of the target return
            factor_cols: list of factor names
            window: rolling window size
            scale: scalar multiplier applied to residual variance
            burn: drop-initial observations
            min_val: clip variance to floor

        Returns:
            Updated KalmanSpec with dynamic process_noise_fn
        """
        engine = KalmanEngine(self.copy())
        results = engine.run(df, target_col=target_col, factor_cols=factor_cols, burn=burn)
        residuals = results["residuals"]

        var_series = residuals.rolling(window).var().clip(lower=min_val).bfill()
        Q_list = [np.eye(self.K) * v * scale for v in var_series]

        self.process_noise_fn = lambda t: Q_list[min(t, len(Q_list) - 1)]
        self.meta = getattr(self, "meta", {})
        self.meta["Q_mode"] = f"rolling_resid_var_from_filter_window_{window}"
        return self

    def set_R_from_ols(self, H: pd.DataFrame, y: pd.Series) -> 'KalmanSpec':
        H_mat = H.copy()
        if self.has_intercept:
            H_mat = pd.concat([pd.Series(1.0, index=H.index, name="Intercept"), H], axis=1)
        beta_ols = np.linalg.pinv(H_mat.values) @ y.values
        residuals = y.values - H_mat.values @ beta_ols
        R = np.var(residuals)
        self.obs_noise_fn = lambda t: np.array([[R]])
        return self

    def set_R_from_rolling_factor_vols(self, H: pd.DataFrame, window: int = 20, min_val: float = 1e-6) -> 'KalmanSpec':
        avg_vol = H.rolling(window).std().mean(axis=1)
        R_series = (avg_vol ** 2).clip(lower=min_val).bfill()
        self.obs_noise_fn = lambda t: np.array([[R_series.iloc[t]]])
        self.meta = getattr(self, "meta", {})
        self.meta["R_mode"] = f"factor_vol_window_{window}"
        return self

    def set_Q_from_mle(self, df: pd.DataFrame, target_col: str, factor_cols: list[str], burn: int = 0) -> 'KalmanSpec':
        best_Q, _ = estimate_mle_Q(self.copy(), df, target_col, factor_cols, burn=burn)
        self.process_noise_fn = lambda t: np.eye(self.K) * best_Q
        self.meta = getattr(self, "meta", {})
        self.meta["Q_scalar"] = best_Q
        return self

    def set_R_from_mle(self, df: pd.DataFrame, target_col: str, factor_cols: list[str], burn: int = 0) -> 'KalmanSpec':
        best_R, _ = estimate_mle_R(self.copy(), df, target_col, factor_cols, burn=burn)
        self.obs_noise_fn = lambda t: np.array([[best_R]])
        self.meta = getattr(self, "meta", {})
        self.meta["R_scalar"] = best_R
        return self
    
    def set_R_from_rolling_ols_residuals(self, df: pd.DataFrame, target_col: str, factor_cols: list[str], window: int = 20, min_val: float = 1e-6) -> 'KalmanSpec':
        """
        Sets R_t from rolling window OLS residual variance.

        Parameters:
            df: full DataFrame with both target and factor columns
            target_col: name of the target return column
            factor_cols: list of factor column names
            window: rolling window size for OLS
            min_val: lower bound on R_t to avoid zero variance

        Returns:
            Updated KalmanSpec with time-varying R_t
        """
        y = df[target_col]
        H = df[factor_cols]

        if self.has_intercept:
            H = pd.concat([pd.Series(1.0, index=H.index, name="Intercept"), H], axis=1)

        R_vals = []
        for t in range(len(df)):
            if t < window:
                R_vals.append(np.nan)
                continue

            H_window = H.iloc[t - window:t]
            y_window = y.iloc[t - window:t]
            beta = np.linalg.pinv(H_window.values) @ y_window.values
            resid = y_window.values - H_window.values @ beta
            R_t = np.var(resid)
            R_vals.append(max(R_t, min_val))

        R_series = pd.Series(R_vals, index=df.index).bfill()
        self.obs_noise_fn = lambda t: np.array([[R_series.iloc[t]]])
        self.meta = getattr(self, "meta", {})
        self.meta["R_mode"] = f"rolling_ols_window_{window}"
        return self


    def validate(self, y: np.ndarray, H: np.ndarray):
        assert y.ndim == 2 and y.shape[1] == 1
        assert H.ndim == 2
        assert H.shape[0] == y.shape[0]
        assert H.shape[1] == self.K or H.shape[1] == self.K - 1

    def describe(self, target_col: str, factor_cols: list[str]) -> dict:
        self.meta = getattr(self, "meta", {})
        desc = {
            "Model Name": self.name,
            "K (State Dim)": self.K,
            "Has Intercept": self.has_intercept,
            "Target Column": target_col,
            "Factor Columns": factor_cols,
            "Initial Beta": self.beta_0.flatten().tolist(),
            "Initial Covariance (P_0 diag)": np.diag(self.P_0).tolist(),
            "Q Type": "dynamic" if callable(self.process_noise_fn) else "constant",
            "R Type": "dynamic" if callable(self.obs_noise_fn) else "constant",
            "Q Scalar": self.meta.get("Q_scalar", None),
            "R Scalar": self.meta.get("R_scalar", None),
            "Q Mode": self.meta.get("Q_mode", None),
            "R Mode": self.meta.get("R_mode", None),
            "Observation Function": str(type(self.observation_fn)),
            "Transition Function": str(type(self.transition_fn))
        }
        return desc

    def copy(self) -> 'KalmanSpec':
        new_spec = KalmanSpec(K=self.K, name=self.name)
        new_spec.beta_0 = self.beta_0.copy()
        new_spec.P_0 = self.P_0.copy()
        new_spec.transition_fn = self.transition_fn
        new_spec.process_noise_fn = self.process_noise_fn
        new_spec.observation_fn = self.observation_fn
        new_spec.obs_noise_fn = self.obs_noise_fn
        new_spec.has_intercept = self.has_intercept
        return new_spec

class KalmanEngine:
    def __init__(self, spec):
        self.spec = spec

    def _log_likelihood(self, innovation, S):
        return -0.5 * (np.log(np.linalg.det(S)) + innovation.T @ np.linalg.inv(S) @ innovation + np.log(2 * np.pi))

    def run(self, df, target_col, factor_cols, burn=0):
        y = df[[target_col]].values
        H = df[factor_cols].values
        index = df.index

        self.spec.validate(y, H)
        T_steps = len(y)
        K = self.spec.K

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

            # Use posteior from previous tep for prior in this step
            beta_pred = T_t @ beta_prev
            # Uncertainty in the prior will always be higher than postetior 
            P_pred = T_t @ P_prev @ T_t.T + Q_t

            # Update stage
            # residual 
            y_hat = H_obs_t @ beta_pred
            innovation = y_t - y_hat
            S = H_obs_t @ P_pred @ H_obs_t.T + R_t
            K_t = P_pred @ H_obs_t.T @ np.linalg.inv(S)

            # Use Kalman gain to update the state and get a new posterior
            # We will end up with with a posteior that is a mix of the prior and the observation
            # The more uncertain the observation is, the more we trust the prior
            # The more uncertain the prior is, the more we trust the observation
            # Prior uncertainty is given by P_pred and is a function of Q_t and T_t and beta_prev and P_prev
            # Observation uncertainty is given by S and is a function of R_t
            beta_post = beta_pred + K_t @ innovation
            P_post = (np.eye(K) - K_t @ H_obs_t) @ P_pred

            log_likelihoods[t] = self._log_likelihood(innovation, S)

            beta_preds[t] = beta_post.flatten()
            beta_covs.append(P_post)
            y_preds[t] = y_hat
            innovations[t] = innovation
            kalman_gains[t] = K_t.flatten()

            beta_prev = beta_post
            P_prev = P_post

        idx = slice(burn, None)
        time_index = df.index[burn:]

        if self.spec.has_intercept:
            cols = ["Intercept"] + factor_cols
        else:
            cols = factor_cols

        return {
            "beta": pd.DataFrame(beta_preds[idx], index=time_index, columns=cols),
            "beta_cov": dict(zip(time_index, beta_covs[burn:])),
            "y_pred": pd.Series(y_preds[idx], index=time_index, name="y_pred"),
            "residuals": pd.Series(innovations[idx], index=time_index, name="residuals"),
            "kalman_gain": pd.DataFrame(kalman_gains[idx], index=time_index, columns=cols),
            "log_likelihood": np.sum(log_likelihoods[burn:]),
            "log_likelihood_t": pd.Series(log_likelihoods[burn:], index=time_index, name="log_likelihood"),
            "meta": self.spec.describe(target_col=target_col, factor_cols=cols),
            "H": pd.DataFrame(H[idx], index=time_index, columns=factor_cols)
        }