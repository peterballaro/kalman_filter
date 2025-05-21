import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from viz.viz_tools import get_sci_template, attach_line_end_labels

# Register the template globally
pio.templates["sci_template"] = get_sci_template()

DEFAULT_PANELS = [
    "actual_vs_predicted",
    "residuals",
    "log_likelihood",
    "gain_norm",
    "drift_norm",
    "rolling_rmse"
]

ALL_PANELS = DEFAULT_PANELS + [
    "acf_residuals",
    "residual_hist",
    "cumulative_resid"
]

def drop_static_components(df: pd.DataFrame, exclude: list[str] = ["Intercept"]) -> pd.DataFrame:
    return df.drop(columns=[c for c in exclude if c in df.columns], errors="ignore")


class ModelDiagnosticsPlotter:
    def __init__(self, results):
        self.results = results
        self.index = results["residuals"].index
        self.meta = results.get("meta", {})
        self.annotations = []
        self.available_plots = {
            "actual_vs_predicted": self._plot_actual_vs_predicted,
            "residuals": self._plot_residuals,
            "log_likelihood": self._plot_log_likelihood,
            "gain_norm": self._plot_gain_norm,
            "drift_norm": self._plot_drift_norm,
            "rolling_rmse": self._plot_rolling_rmse,
            "acf_residuals": self._plot_acf_residuals,
            "residual_hist": self._plot_residual_hist,
            "cumulative_resid": self._plot_cumulative_resid
        }

    def plot(self, include=None, height_per_row=300):
        if include is None:
            include = DEFAULT_PANELS
        elif not isinstance(include, list):
            raise TypeError("`include` must be a list of plot names or None.")

        # Validate plot types
        valid_plots = list(self.available_plots.keys())
        invalid = [name for name in include if name not in valid_plots]
        if invalid:
            print(f"Invalid plot type(s): {invalid}")
            print(f"Valid plot types are: {valid_plots}")
            raise ValueError(f"Invalid plot type(s): {invalid}")

        self.annotations = []
        n = len(include)
        max_cols = 3
        cols = min(n, max_cols)
        rows = int(np.ceil(n / cols))

        subplot_title_map = {
            "actual_vs_predicted": "Actual vs Predicted",
            "residuals": "Residuals: yₜ - Hₜβₜ",
            "log_likelihood": "Log Likelihood: log p(yₜ | βₜ)",
            "gain_norm": "Gain Norm: ||Kₜ||₂ = √∑Kₜ²",
            "drift_norm": "Drift Norm: ||βₜ - βₜ₋₁||₂",
            "rolling_rmse": "Rolling RMSE (window=12)",
            "acf_residuals": "ACF of Residuals",
            "residual_hist": "Histogram of Residuals",
            "cumulative_resid": "Cumulative Sum of Residuals"
        }
        titles = [subplot_title_map.get(name, name.title()) for name in include]

        fig = make_subplots(
            rows=rows,
            cols=cols,
            shared_xaxes=False,
            shared_yaxes=False,
            subplot_titles=titles,
            vertical_spacing=0.12
        )

        row, col = 1, 1
        for name in include:
            if name in self.available_plots:
                traces = self.available_plots[name]()
                for t in traces:
                    fig.add_trace(t, row=row, col=col)
                col += 1
                if col > cols:
                    col = 1
                    row += 1

        # Combine plot-specific axis labels if set (e.g. from actual_vs_predicted)
        axis_labels = getattr(self, "_axis_labels", {})

        fig.update_layout(
            height=height_per_row * rows + 50,
            title=dict(
                text=self._generate_title(),
                y=0.995,
                x=0.5,
                xanchor='center',
                yanchor='top'
            ),
            showlegend=False,
            template="sci_template",
            annotations=fig.layout.annotations + tuple(self.annotations),
            **axis_labels
        )

        return fig


    def _make_labeled_trace(self, trace):
        fig = go.Figure([trace])
        attach_line_end_labels(fig)
        return fig.data

    def _plot_actual_vs_predicted(self, top_n=5):
        actual = self.results["residuals"] + self.results["y_pred"]
        predicted = self.results["y_pred"]
        residuals = self.results["residuals"]
        index = self.results["y_pred"].index

        # Regression
        slope, intercept = np.polyfit(predicted, actual, 1)
        r = np.corrcoef(predicted, actual)[0, 1]
        r2 = r ** 2

        # Size encoding
        resid_scaled = np.abs(residuals)
        sizes = 8 + 40 * (resid_scaled - resid_scaled.min()) / (resid_scaled.max() - resid_scaled.min() + 1e-6)

        # Regression line
        trend_x = np.linspace(predicted.min(), predicted.max(), 100)
        trend_y = slope * trend_x + intercept

        # Hover
        hover_text = [
            f"Date: {d.strftime('%Y-%m-%d')}<br>Residual: {rval:.5f}"
            for d, rval in zip(index, residuals)
        ]

        # Main scatter
        trace_scatter = go.Scatter(
            x=predicted,
            y=actual,
            mode="markers",
            name="Fit",
            marker=dict(
                size=sizes,
                opacity=0.6,
                color="rgba(0, 120, 200, 0.5)",
                line=dict(width=1, color="white")
            ),
            text=hover_text,
            hovertemplate="%{text}<br>Predicted: %{x:.4f}<br>Actual: %{y:.4f}<extra></extra>"
        )

        # Trend line
        trace_trend = go.Scatter(
            x=trend_x,
            y=trend_y,
            mode="lines",
            name="Trend",
            line=dict(color="green", dash="dash"),
            hoverinfo="skip"
        )

        # Top-N residual label points
        top_n_indices = np.argsort(-np.abs(residuals.values))[:top_n]
        top_dates = index[top_n_indices]
        top_pred = predicted.iloc[top_n_indices]
        top_actual = actual.iloc[top_n_indices]
        date_labels = top_dates.strftime("%Y-%m-%d")

        trace_labels = go.Scatter(
            x=top_pred,
            y=top_actual,
            mode="text",
            text=date_labels,
            textposition="top center",
            textfont=dict(size=11, color="black"),
            showlegend=False,
            hoverinfo="skip"
        )

        # Add annotation with regression info (pinned to subplot)
        annotation_text = (
            f"Slope: {slope:.2f}<br>"
            f"Intercept: {intercept:.2%}<br>"
            f"R²: {r2:.3f}<br>"
            f"Corr: {r:.3f}"
        )
        self.annotations.append(dict(
            text=annotation_text,
            xref="x domain", yref="y domain",
            x=0.01, y=0.99,
            xanchor="left", yanchor="top",
            showarrow=False,
            font=dict(size=12),
            align="left",
            bgcolor="white",
            bordercolor="black",
            borderwidth=1
        ))

        # Set axis labels in update_layout call
        self._axis_labels = dict(
            xaxis_title="Predicted",
            yaxis_title="Actual"
        )

        return [trace_scatter, trace_trend, trace_labels]


    def _plot_residuals(self):
        resid = self.results["residuals"]
        std = resid.std()
        upper = 1.96 * std
        lower = -1.96 * std

        trace_resid = go.Scatter(
            x=resid.index,
            y=resid.values,
            mode="lines",
            name="Residuals",
            line=dict(color="blue")
        )

        trace_upper = go.Scatter(
            x=resid.index,
            y=[upper] * len(resid),
            mode="lines",
            name="+1.96 SD",
            line=dict(color="red", dash="dash")
        )

        trace_lower = go.Scatter(
            x=resid.index,
            y=[lower] * len(resid),
            mode="lines",
            name="-1.96 SD",
            line=dict(color="red", dash="dash")
        )

        extreme_outliers = resid[(resid > 3.5 * std) | (resid < -3.5 * std)]
        trace_labels = go.Scatter(
            x=extreme_outliers.index,
            y=extreme_outliers.values,
            mode="text",
            text=[d.strftime("%Y-%m-%d") for d in extreme_outliers.index],
            textposition="top center",
            textfont=dict(size=11, color="black"),
            showlegend=False,
            hoverinfo="skip"
        )

        return [trace_resid, trace_upper, trace_lower, trace_labels]

    def _plot_log_likelihood(self):
        ll = self.results.get("log_likelihood_t")
        if ll is None:
            return [go.Scatter(x=self.index, y=[None] * len(self.index), name="LL")]
        trace = go.Scatter(x=ll.index, y=ll.values, mode="lines", name="Log-Likelihood")
        return list(self._make_labeled_trace(trace))

    def _plot_gain_norm(self):
        gains = self.results["kalman_gain"]
        norms = gains.apply(np.linalg.norm, axis=1)
        trace = go.Scatter(x=gains.index, y=norms, mode="lines", name="Gain Norm")
        return list(self._make_labeled_trace(trace))

    def _plot_drift_norm(self):
        betas = self.results["beta"]
        drift = betas.diff().dropna().apply(np.linalg.norm, axis=1)
        trace = go.Scatter(x=drift.index, y=drift.values, mode="lines", name="Drift Norm")
        return list(self._make_labeled_trace(trace))

    def _plot_rolling_rmse(self, window=12):
        resid = self.results["residuals"]
        rmse = resid.rolling(window).apply(lambda x: np.sqrt(np.mean(x**2)))
        trace = go.Scatter(x=rmse.index, y=rmse.values, mode="lines", name="Rolling RMSE")
        return list(self._make_labeled_trace(trace))

    def _plot_acf_residuals(self, lags=20):
        import statsmodels.api as sm
        resid = self.results["residuals"].fillna(0)
        acf_vals = sm.tsa.acf(resid, nlags=lags)
        return [go.Bar(x=list(range(lags + 1)), y=acf_vals, name="ACF")]

    def _plot_residual_hist(self):
        resid = self.results["residuals"].dropna()
        return [go.Histogram(x=resid.values, name="Residual Dist", nbinsx=30)]

    def _plot_cumulative_resid(self):
        resid = self.results["residuals"].cumsum()
        trace = go.Scatter(x=resid.index, y=resid.values, mode="lines", name="Cumulative Residual")
        return list(self._make_labeled_trace(trace))

    def _generate_title(self):
        m = self.meta
        q = m.get("Q_scale", "custom")
        r = m.get("R_scale", "custom")
        name = m.get("model_name", "Kalman Model")
        target = m.get("target_col", "Target")
        return f"Diagnostics for {name} (Target = {target}, Q = {q}, R = {r})"

def summarize_model_diagnostics(results: dict) -> pd.DataFrame:
    meta = results.get("meta", {})
    residuals = results["residuals"]
    y_pred = results["y_pred"]
    index = residuals.index
    n_obs = len(residuals)
    date_range = f"{index[0].date()} → {index[-1].date()}"

    rmse_series = (residuals**2).rolling(window=1).mean().apply(np.sqrt)
    final_rmse = rmse_series.iloc[-1]
    mean_rmse = np.sqrt(np.mean(residuals**2))
    sse = np.sum(residuals**2)

    ll = results.get("log_likelihood_t", None)
    log_likelihood_total = ll.sum() if ll is not None else np.nan

    gains = drop_static_components(results.get("kalman_gain", pd.DataFrame()))
    gain_norms = gains.apply(np.linalg.norm, axis=1) if not gains.empty else None
    mean_gain_norm = gain_norms.mean() if gain_norms is not None else np.nan

    betas = drop_static_components(results.get("beta", pd.DataFrame()))
    drift = betas.diff().dropna().apply(np.linalg.norm, axis=1) if not betas.empty else None
    mean_drift_norm = drift.mean() if drift is not None else np.nan

    return pd.DataFrame([{
        "Model Name": meta.get("model_name", "Unknown"),
        "Target": meta.get("target_col", "y"),
        "Date Range": date_range,
        "# Observations": n_obs,
        "Final RMSE": round(final_rmse, 6),
        "Mean RMSE": round(mean_rmse, 6),
        "Total SSE": round(sse, 6),
        "Cumulative Log-Likelihood": round(log_likelihood_total, 6),
        "Mean Gain Norm": round(mean_gain_norm, 6),
        "Mean Drift Norm": round(mean_drift_norm, 6)
    }])


def summarize_factor_dynamics(results: dict) -> pd.DataFrame:
    """
    Returns a one-row-per-factor DataFrame summarizing the behavior of each factor
    in a Kalman filter run. Requires 'beta', 'kalman_gain', and 'H' in the results dict.
    
    Columns:
        - Factor: Name of the factor
        - Avg Beta: Mean exposure to the factor across time
        - Beta Std: Std dev of the exposure across time
        - Final Beta: Last estimated exposure
        - Beta Z-Score: Final beta, standardized relative to its own history
        - Min Beta / Max Beta: Historical extremes in beta
        - Drift Volatility: Std dev of βₜ - βₜ₋₁ (how much the exposure moves)
        - Mean Gain: Avg Kalman gain (sensitivity to new info)
        - Avg Contribution: Mean of Hₜₖ × βₜₖ (contribution to predicted y)
        - Final Contribution: Last value of Hₜₖ × βₜₖ
    """
    beta = results["beta"]  # T × K DataFrame
    gains = results.get("kalman_gain")  # T × K DataFrame or None
    H = results["H"]  # T × K DataFrame
    meta = results.get("meta", {})
    summaries = []

    for factor in beta.columns:
        beta_k = beta[factor]        # exposure time series for factor k
        H_k = H[factor]              # corresponding factor returns
        drift = beta_k.diff().dropna()      # βₜ - βₜ₋₁
        contrib = beta_k * H_k              # Hₜₖ × βₜₖ
        gain_k = gains[factor] if gains is not None and factor in gains.columns else None

        mean_beta = beta_k.mean()           # average exposure over time
        std_beta = beta_k.std()             # variability of exposure
        final_beta = beta_k.iloc[-1]        # latest exposure
        z_score = (final_beta - mean_beta) / std_beta if std_beta > 0 else np.nan

        row = {
            "Model Name": meta.get("model_name", "Unknown"),
            "Factor": factor,
            "Avg Beta": mean_beta,
            "Beta Std": std_beta,
            "Final Beta": final_beta,
            "Beta Z-Score": z_score,
            "Min Beta": beta_k.min(),
            "Max Beta": beta_k.max(),
            "Drift Volatility": drift.std(),  # how much exposure moves step to step
            "Mean Gain": gain_k.mean() if gain_k is not None else np.nan,
            "Avg Contribution": contrib.mean(),      # avg influence on prediction
            "Final Contribution": contrib.iloc[-1]   # last-period influence on prediction
        }

        summaries.append(row)

    return pd.DataFrame(summaries)

def plot_factor_contributions(results: dict, include_factors: list[str] | None = None) -> go.Figure:
    """
    Creates a stacked area plot showing per-period factor contributions to the predicted return.

    Inputs:
        - results: dict from KalmanEngine.run() with 'beta', 'H', 'y_pred'
        - include_factors: optional list of factor names (defaults to all)

    Output:
        - Plotly Figure with stacked factor contributions + predicted return overlay
    """
    beta = results["beta"]
    H = results["H"]
    y_pred = results["y_pred"]

    if include_factors is None:
        include_factors = beta.columns.tolist()

    index = beta.index
    contrib_df = pd.DataFrame(index=index)

    # Compute contributions per factor
    for factor in include_factors:
        contrib_df[factor] = beta[factor] * H[factor]

    # Optional: sort stacking order for aesthetics
    contrib_df = contrib_df[contrib_df.mean().sort_values().index]

    # --- Plot stacked area ---
    fig = go.Figure()
    cumulative = np.zeros(len(contrib_df))

    for i, factor in enumerate(contrib_df.columns):
        values = cumulative + contrib_df[factor].values
        dates = contrib_df.index.strftime("%Y-%m-%d")
        hover_text = [f"{factor}<br>Date: {d}<br>Contribution: {v:.5f}" for d, v in zip(dates, values)]

        trace = go.Scatter(
            x=contrib_df.index,
            y=values,
            name=factor,
            mode="lines",
            fill="tonexty" if i > 0 else "tozeroy",
            stackgroup="contrib",
            line=dict(width=0.5),
            text=hover_text,
            hovertemplate="%{text}<extra></extra>"
        )
        fig.add_trace(trace)
        cumulative = values

    # --- Add predicted return as black line ---
    pred_hover = [f"Predicted<br>Date: {d}<br>{v:.5f}" for d, v in zip(y_pred.index.strftime("%Y-%m-%d"), y_pred.values)]
    fig.add_trace(go.Scatter(
        x=y_pred.index,
        y=y_pred.values,
        name="Predicted Return",
        mode="lines",
        line=dict(color="black", width=2),
        text=pred_hover,
        hovertemplate="%{text}<extra></extra>"
    ))

    fig.update_layout(
        title="Factor Contributions to Predicted Return Over Time",
        xaxis_title="Date",
        yaxis_title="Predicted Return",
        template="sci_template",  
        legend_title="Factors"
    )

    return fig

def plot_beta_grid(results: dict, n_cols: int = 3, show_gain: bool = False) -> go.Figure:
    """
    Grid of subplots for each factor showing:
    - Beta path (blue)
    - ±1 std shading
    - Optional Kalman gain overlay on secondary y-axis (dotted orange)
    - Optional dashed line at y=0 if beta crosses zero
    - End-of-line labels for beta traces only (via attach_line_end_labels)
    """
    beta = results["beta"]
    gain = results["kalman_gain"]
    index = beta.index
    factors = beta.columns.tolist()

    n_factors = len(factors)
    cols = min(n_cols, n_factors)
    rows = int(np.ceil(n_factors / cols))

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=factors,
        shared_xaxes=False,
        shared_yaxes=False,
        specs=[[{"secondary_y": True} for _ in range(cols)] for _ in range(rows)],
        vertical_spacing=0.10,
        horizontal_spacing=0.05
    )

    row, col = 1, 1
    trace_names = []

    for factor in factors:
        beta_series = beta[factor]
        gain_series = gain[factor]
        std = beta_series.std()
        upper_band = beta_series + std
        lower_band = beta_series - std

        # --- Beta Trace ---
        beta_trace_name = f"{factor} β"
        trace_names.append(beta_trace_name)

        beta_trace = go.Scatter(
            x=index,
            y=beta_series,
            name=beta_trace_name,
            mode="lines",
            line=dict(color="royalblue", width=2),
            showlegend=False
        )

        # --- Shaded ±1 std ---
        band_trace = go.Scatter(
            x=list(index) + list(index[::-1]),
            y=list(upper_band) + list(lower_band[::-1]),
            fill="toself",
            fillcolor="rgba(65, 105, 225, 0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=False
        )

        fig.add_trace(band_trace, row=row, col=col)
        fig.add_trace(beta_trace, row=row, col=col, secondary_y=False)

        # --- Kalman Gain Trace (optional) ---
        if show_gain:
            gain_trace = go.Scatter(
                x=index,
                y=gain_series,
                name=f"{factor} Gain",
                mode="lines",
                line=dict(color="darkorange", width=1.5, dash="dot"),
                showlegend=False
            )
            fig.add_trace(gain_trace, row=row, col=col, secondary_y=True)

        # --- y = 0 line if beta crosses zero
        if beta_series.min() < 0 and beta_series.max() > 0:
            fig.add_trace(go.Scatter(
                x=[index[0], index[-1]],
                y=[0, 0],
                mode="none", 
                line=dict(color="gray", width=1, dash="dash"),
                hoverinfo="skip",
                showlegend=False
            ), row=row, col=col, secondary_y=False)

        # --- Clean axis labels
        fig.update_yaxes(title_text=None, row=row, col=col, secondary_y=False)
        if show_gain:
            fig.update_yaxes(title_text=None, row=row, col=col, secondary_y=True)

        col += 1
        if col > cols:
            col = 1
            row += 1

        # --- Layout
    # --- Extract model name from metadata
    model_name = results.get("meta", {}).get("model_name", "Model")

    fig.update_layout(
        height=300 * rows,
        width=1100,
        title_text=f"{model_name} – Betas",
        template="sci_template",
        hovermode=None,
        margin=dict(t=60, b=40)
    )

    # --- Apply end labels only to β lines
    attach_line_end_labels(
        fig,
        trace_names=trace_names,
        font_size=13,
        text_anchor='middle left'
    )

    # --- Ensure no y-axis titles anywhere
    for i in range(1, len(fig.layout.annotations) + 1):
        fig.update_layout({f"yaxis{i}": dict(title=None)})

    return fig



