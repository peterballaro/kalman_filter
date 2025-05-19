import numpy as np
import pandas as pd

import numpy as np
from scipy.optimize import minimize

def estimate_mle_R(spec_template, df, target_col, factor_cols, burn=0, method='L-BFGS-B', bounds=(1e-6, 1e2)):
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


def estimate_mle_Q(spec_template, df, target_col, factor_cols, burn=0, method='L-BFGS-B', bounds=(1e-8, 1e1)):
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
    def __init__(self, K, name="Kalman Model"):
        self.K = K
        self.name = name
        self.beta_0 = np.zeros((K, 1))
        self.P_0 = np.eye(K) * 1e2
        self.transition_fn = lambda t: np.eye(self.K)
        self.process_noise_fn = lambda t: np.eye(self.K) * 1e-4
        self.observation_fn = lambda t, H_t: H_t.reshape(1, self.K)
        self.obs_noise_fn = lambda t: np.eye(1) * 1e-2
        self.has_intercept = False

    def add_intercept(self, init_val: float = 0.0):
        # TODO Interface this better with the rest of the code.
        self.K += 1
        self.beta_0 = np.insert(self.beta_0, 0, init_val).reshape(-1, 1)
        self.P_0 = np.pad(self.P_0, ((1, 0), (1, 0)), mode="constant")
        self.P_0[0, 0] = 1e6
        self.has_intercept = True
        self.observation_fn = lambda t, H_t: np.hstack([np.ones(1), H_t]).reshape(1, self.K)
        return self

    def set_Q_from_mle(self, df, target_col, factor_cols, burn=0):
        best_Q, _ = estimate_mle_Q(self.copy(), df, target_col, factor_cols, burn=burn)
        self.process_noise_fn = lambda t: np.eye(self.K) * best_Q
        self.meta = getattr(self, "meta", {})
        self.meta["Q_scalar"] = best_Q
        return self

    def set_R_from_mle(self, df, target_col, factor_cols, burn=0):

        best_R, _ = estimate_mle_R(self.copy(), df, target_col, factor_cols, burn=burn)
        self.obs_noise_fn = lambda t: np.array([[best_R]])
        self.meta = getattr(self, "meta", {})
        self.meta["R_scalar"] = best_R
        return self

    def set_Q_from_factor_vols(self, H):
        Q = np.diag(np.var(H, axis=0))
        if self.has_intercept:
            Q = np.pad(Q, ((1, 0), (1, 0)), mode="constant")
        self.process_noise_fn = lambda t: Q
        return self

    def estimate_R_from_ols(self, H, y):
        if self.has_intercept:
            H = np.hstack([np.ones((H.shape[0], 1)), H])
        beta_ols = np.linalg.pinv(H) @ y
        residuals = y - H @ beta_ols
        R = np.var(residuals)
        self.obs_noise_fn = lambda t: np.array([[R]])
        return self

    def estimate_initial_state(self, H, y):
        if self.has_intercept:
            H = np.hstack([np.ones((H.shape[0], 1)), H])
        beta_ols = np.linalg.pinv(H) @ y
        self.beta_0 = beta_ols
        return self

    def validate(self, y, H):
        assert y.ndim == 2 and y.shape[1] == 1
        assert H.ndim == 2
        assert H.shape[0] == y.shape[0]
        assert H.shape[1] == self.K or H.shape[1] == self.K - 1

    def describe(self, target_col, factor_cols):
        return {
            "model_name": self.name,
            "target_col": target_col,
            "factor_cols": factor_cols,
            "K": self.K,
            "has_intercept": self.has_intercept
        }

    def copy(self):
        new_spec = KalmanSpec(K=self.K, name=self.name)
        new_spec.beta_0 = self.beta_0.copy()
        new_spec.P_0 = self.P_0.copy()
        new_spec.transition_fn = self.transition_fn
        new_spec.process_noise_fn = self.process_noise_fn
        new_spec.observation_fn = self.observation_fn
        new_spec.obs_noise_fn = self.obs_noise_fn
        new_spec.has_intercept = getattr(self, "has_intercept", False)
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

            beta_pred = T_t @ beta_prev
            P_pred = T_t @ P_prev @ T_t.T + Q_t

            y_hat = H_obs_t @ beta_pred
            innovation = y_t - y_hat
            S = H_obs_t @ P_pred @ H_obs_t.T + R_t
            K_t = P_pred @ H_obs_t.T @ np.linalg.inv(S)

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
