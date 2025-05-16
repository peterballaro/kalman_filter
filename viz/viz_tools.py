from datetime import datetime
from typing import Optional, Union, Dict, List
import plotly.io as pio
import plotly.graph_objects as go
from plotly.graph_objects import Figure

def get_sci_template():
    timestamp = datetime.now().strftime("Generated %Y-%m-%d %H:%M")

    hover_2d = "%{y:.2f}<extra></extra>"
    hover_xy = "%{x}, %{y:.2f}<extra></extra>"
    hover_3d = "%{x:.2f}, %{y:.2f}, %{z:.2f}<extra></extra>"
    return {
        "layout": {
            "template": "simple_white",
            "font": {"family": "Arial", "size": 13},
            "margin": {"l": 40, "r": 40, "t": 40, "b": 40},
            "title": {
                "y": 0.92,
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top",
                "font": {"size": 16}
            },
            "xaxis": {
                "showgrid": True,
                "gridcolor": "lightgray",
                "showline": True,
                "linecolor": "black",
                "ticks": "outside",
                "tickfont": {"size": 14},
                "hoverformat": ".2f",
                "title": {"standoff": 10}
            },
            "yaxis": {
                "showgrid": True,
                "gridcolor": "lightgray",
                "zeroline": False,
                "showline": True,
                "linecolor": "black",
                "tickfont": {"size": 14},
                "hoverformat": ".2f",
                "title": {"standoff": 10}
            },
            "yaxis2": {
                "overlaying": "y",
                "side": "left",
                "position": 0.02,
                "showgrid": False,
                "showline": True,
                "tickcolor": "#ff6600",
                "linecolor": "black",
                "tickfont": {"size": 14, "color": "#ff6600"},
                "titlefont": {"color": "#ff6600"},
                "title": {"standoff": 8},
                "ticklabelposition": "inside"
            },
            "legend": {
                "orientation": "v",
                "xanchor": "left",
                "x": 1.02,
                "y": 1,
                "font": {"size": 14}
            },
            "height": 400,
            "annotations": [
                {
                    "text": timestamp,
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0.99,
                    "y": 0.01,
                    "xanchor": "right",
                    "yanchor": "bottom",
                    "showarrow": False,
                    "font": {"size": 10, "color": "gray"}
                }
            ]
        },
        "data": {
            "scatter": [{"hovertemplate": hover_xy, "line": {"width": 3}}],
            "bar": [{"hovertemplate": hover_2d}],
            "box": [{"hovertemplate": hover_2d}],
            "violin": [{"hovertemplate": hover_2d}],
            "heatmap": [{"hovertemplate": hover_2d}],
            "pie": [{"hovertemplate": "%{label}: %{value:.2f}<extra></extra>"}],
            "funnel": [{"hovertemplate": hover_2d}],
            "scatter3d": [{"hovertemplate": hover_3d}],
            "surface": [{"hovertemplate": hover_3d}]
        }
    }

pio.templates["sci_template"] = get_sci_template()

def attach_line_end_labels(
    fig: Figure,
    trace_names: Optional[Union[List[str], List[int]]] = None,
    precision: Union[int, Dict[str, int]] = 2,
    font_size: int = 12,
    color_override: Optional[Union[str, Dict[str, str]]] = None,
    show_marker: bool = False,
    y_offset: float = 0.0,
    text_anchor: str = "middle right",
    verbose: bool = False
) -> Figure:
    for i, trace in enumerate(fig.data):
        try:
            if not isinstance(trace, (go.Scatter, go.Scattergl)):
                continue
            if not trace.mode or 'lines' not in trace.mode:
                continue
            if trace.x is None or trace.y is None or len(trace.x) == 0 or len(trace.y) == 0:
                continue

            trace_id = trace.name or f"trace_{i}"

            if trace_names is not None:
                if isinstance(trace_names[0], int) and i not in trace_names:
                    continue
                if isinstance(trace_names[0], str) and trace_id not in trace_names:
                    continue

            x_last = trace.x[-1]
            y_last = trace.y[-1] + y_offset
            xaxis = trace.xaxis if getattr(trace, "xaxis", None) else "x"
            yaxis = trace.yaxis if getattr(trace, "yaxis", None) else "y"

            prec = precision.get(trace_id, 2) if isinstance(precision, dict) else precision
            label = f"{trace.y[-1]:.{prec}f}"

            trace_color = getattr(trace.line, "color", "black")
            text_color = (
                color_override.get(trace_id, trace_color) if isinstance(color_override, dict)
                else color_override or trace_color
            )

            if not getattr(trace, "legendgroup", None) and trace.name:
                trace.legendgroup = trace.name

            fig.add_trace(go.Scatter(
                x=[x_last],
                y=[y_last],
                mode="text+markers" if show_marker else "text",
                text=[label],
                textposition=text_anchor,
                textfont=dict(size=font_size, color=text_color),
                marker=dict(size=6, color=text_color) if show_marker else None,
                legendgroup=trace.legendgroup or trace.name or trace_id,
                showlegend=False,
                xaxis=xaxis,
                yaxis=yaxis
            ))
        except Exception as e:
            if verbose:
                trace_name = getattr(trace, 'name', f'trace_{i}')
                print(f"[attach_line_end_labels] Skipped trace '{trace_name}' (index {i}): {e}")
    return fig

def safe_center_dual_yaxes(
    fig: Figure,
    center_y1: Optional[float] = None,
    center_y2: Optional[float] = None,
    margin: float = 0.1,
    verbose: bool = False
) -> Figure:
    def get_bounds(values, center):
        span = max(abs(max(values) - center), abs(min(values) - center))
        return center - (1 + margin) * span, center + (1 + margin) * span

    y1_values = []
    y2_values = []
    y2_used = False

    for trace in fig.data:
        if not isinstance(trace, (go.Scatter, go.Scattergl)):
            continue
        if trace.y is None or len(trace.y) == 0:
            continue

        axis = getattr(trace, "yaxis", "y")
        if axis == "y2":
            y2_values.extend(trace.y)
            y2_used = True
        else:
            y1_values.extend(trace.y)

    if center_y1 is not None and y1_values:
        y1_min, y1_max = get_bounds(y1_values, center_y1)
        fig.layout.yaxis.range = [y1_min, y1_max]
        if verbose:
            print(f"yaxis centered on {center_y1} with range: ({y1_min:.2f}, {y1_max:.2f})")

    if center_y2 is not None:
        if "yaxis2" in fig.layout and y2_values:
            y2_min, y2_max = get_bounds(y2_values, center_y2)
            fig.layout.yaxis2.range = [y2_min, y2_max]
            if verbose:
                print(f"yaxis2 centered on {center_y2} with range: ({y2_min:.2f}, {y2_max:.2f})")
        elif verbose:
            print("Warning: yaxis2 not found in layout. Skipping secondary centering.")

    return fig
