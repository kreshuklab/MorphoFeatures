"""Consistent categorical projections for the worker, Tools and Results."""

import io

import numpy as np
import pandas as pd


def category_colors(values):
    from matplotlib import colormaps
    from matplotlib.colors import to_hex

    names = sorted({str(value) for value in values if pd.notna(value)})
    palette = colormaps["tab20"] if len(names) <= 20 else colormaps["gist_ncar"]
    return {name: to_hex(palette(i / max(len(names) - 1, 1))) for i, name in enumerate(names)}


def projection_panels(frame, columns=None):
    panels = [("cluster", "Embedding clusters")]
    if "known_label" in frame and frame.known_label.notna().any():
        panels.append(("known_label", "Known cell types"))
    if "predicted_label" in frame and frame.predicted_label.notna().any():
        panels.append(("predicted_label", "Predicted cell types"))
    if columns is not None:
        available = dict(panels)
        if not columns or not set(columns) <= set(available):
            raise ValueError("Choose available cluster, known-label or predicted-label panels")
        panels = [(column, available[column]) for column in columns]
    return panels


def label_palette(frame):
    if "class_colors" in frame.attrs:
        return frame.attrs["class_colors"]
    return category_colors(
        [
            v
            for column in ("known_label", "predicted_label")
            if column in frame
            for v in frame[column]
        ]
    )


def projection_figure(
    frame,
    title="Morphology",
    *,
    methods=None,
    unlabeled_opacity=0.15,
    point_size=10,
    panels=None,
):
    import textwrap

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if not 0 <= unlabeled_opacity <= 1:
        raise ValueError("Unlabeled opacity must be between zero and one")
    methods = methods or (["pca"] + (["umap"] if "umap_1" in frame else []))
    panels = projection_panels(frame, panels)
    class_colors = label_palette(frame)
    categories = {}
    for column, _ in panels:
        palette = category_colors(frame[column]) if column == "cluster" else class_colors
        categories[column] = [
            (name, color)
            for name, color in palette.items()
            if frame[column].astype("string").eq(name).fillna(False).any()
        ]
        if frame[column].isna().any():
            categories[column].append(("Unlabeled", "#aaaaaa"))
    wrapped = {
        column: [(textwrap.fill(name, 32), color) for name, color in values]
        for column, values in categories.items()
    }
    legend_height = max(
        0.55,
        max(sum(name.count("\n") + 1 for name, _ in values) for values in wrapped.values()) * 0.20
        + 0.25,
    )
    figure = plt.figure(figsize=(4.6 * len(panels), (4.5 + legend_height) * len(methods) + 0.35))
    grid = figure.add_gridspec(
        2 * len(methods),
        len(panels),
        height_ratios=[value for _ in methods for value in (4.0, legend_height)],
    )
    unlabeled = frame.get("known_label", pd.Series("labeled", index=frame.index)).isna().to_numpy()
    if unlabeled.all():
        unlabeled[:] = False
    for row, method in enumerate(methods):
        x, y = frame[method + "_1"].to_numpy(), frame[method + "_2"].to_numpy()
        reference = None
        for col, (column, heading) in enumerate(panels):
            axis = figure.add_subplot(grid[2 * row, col], sharex=reference, sharey=reference)
            reference = axis
            values = frame[column].astype("string")
            for name, color in categories[column]:
                keep = (
                    values.isna().to_numpy()
                    if name == "Unlabeled" and color == "#aaaaaa"
                    else values.eq(name).fillna(False).to_numpy(dtype=bool)
                )
                axis.scatter(
                    x[keep],
                    y[keep],
                    s=point_size,
                    c=color,
                    alpha=np.where(unlabeled[keep], unlabeled_opacity, 0.85),
                    linewidths=0,
                )
            axis.set(xlabel=method.upper() + " 1", ylabel=method.upper() + " 2", title=heading)
            axis.spines[["top", "right"]].set_visible(False)
            legend_axis = figure.add_subplot(grid[2 * row + 1, col])
            legend_axis.set_axis_off()
            handles = [
                Line2D([], [], marker="o", linestyle="", color=color, markersize=6, label=name)
                for name, color in wrapped[column]
            ]
            legend_axis.legend(
                handles=handles,
                loc="upper left",
                fontsize=9,
                frameon=False,
                borderaxespad=0,
                handletextpad=0.6,
                labelspacing=0.55,
            )
    figure.suptitle(title, fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, 0.97), h_pad=1.5, w_pad=2.5)
    return figure


def figure_bytes(figure, format="svg", *, dpi=180):
    buffer = io.BytesIO()
    figure.savefig(buffer, format=format, dpi=dpi, bbox_inches="tight", facecolor="white")
    return buffer.getvalue()


def save_projection_figure(frame, destination, title, settings=None):
    import matplotlib.pyplot as plt

    figure = projection_figure(
        frame, title, unlabeled_opacity=float((settings or {}).get("unlabeled_opacity", 0.15))
    )
    figure.savefig(destination, bbox_inches="tight")
    plt.close(figure)


def interactive_projection(
    frame, method="umap", *, unlabeled_opacity=0.15, point_size=5, panels=None
):
    import html
    import textwrap

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    panels = projection_panels(frame, panels)
    figure = make_subplots(
        rows=1,
        cols=len(panels),
        subplot_titles=[title for _, title in panels],
        shared_xaxes=True,
        shared_yaxes=True,
    )
    class_colors = label_palette(frame)
    unlabeled = frame.get("known_label", pd.Series("labeled", index=frame.index)).isna()
    if unlabeled.all():
        unlabeled = pd.Series(False, index=frame.index)
    legends = {}
    max_lines = 0
    for col, (column, _) in enumerate(panels, 1):
        legend_id = "legend" if col == 1 else f"legend{col}"
        domain = getattr(figure.layout, "xaxis" if col == 1 else f"xaxis{col}").domain
        legends[legend_id] = {
            "x": domain[0],
            "y": -0.22,
            "xanchor": "left",
            "yanchor": "top",
            "orientation": "v",
            "font": {"size": 12},
            "itemsizing": "constant",
            # Inherit Streamlit's paper and text colors, including dark/custom
            # themes. A forced white box makes inherited light text invisible.
            "bgcolor": "rgba(0,0,0,0)",
        }
        if "maxheight" in go.layout.Legend()._valid_props:
            legends[legend_id]["maxheight"] = 210
        lines = 0
        colors = category_colors(frame[column]) if column == "cluster" else class_colors
        values = frame[column].astype("string").fillna("Unlabeled")
        for category in sorted(values.unique()):
            wrapped_name = textwrap.wrap(str(category), 28)
            lines += len(wrapped_name)
            display_name = "<br>".join(html.escape(line) for line in wrapped_name)
            keep = values == category
            points = frame.loc[keep]
            custom = np.column_stack(
                (
                    points.label_id.astype(str),
                    points.get("known_label", pd.Series("", index=points.index))
                    .fillna("Unlabeled")
                    .astype(str),
                )
            )
            figure.add_trace(
                go.Scattergl(
                    x=points[method + "_1"],
                    y=points[method + "_2"],
                    mode="markers",
                    name=display_name,
                    legend=legend_id,
                    customdata=custom,
                    marker={
                        "color": colors.get(str(category), "#aaaaaa"),
                        "size": point_size,
                        "opacity": np.where(unlabeled.loc[keep], unlabeled_opacity, 0.85),
                    },
                    hovertemplate="ID %{customdata[0]}<br>Known: %{customdata[1]}<br>%{x:.3f}, %{y:.3f}<extra>%{fullData.name}</extra>",
                ),
                row=1,
                col=col,
            )
        max_lines = max(max_lines, lines)
    legend_pixels = (
        min(210, max_lines * 20 + 10)
        if "maxheight" in go.layout.Legend()._valid_props
        else max_lines * 20 + 10
    )
    figure.update_layout(
        height=470 + legend_pixels,
        template="plotly_white",
        clickmode="event+select",
        dragmode="lasso",
        **legends,
        margin={"t": 45, "b": 85 + legend_pixels},
        uirevision=method,
    )
    for col in range(1, len(panels) + 1):
        figure.update_xaxes(title_text=method.upper() + " 1", matches="x", row=1, col=col)
        figure.update_yaxes(title_text=method.upper() + " 2", matches="y", row=1, col=col)
    return figure


def selected_object_ids(event, allowed_ids):
    """Decode IDs from customdata, never rounded browser numbers or trace indices."""
    allowed = {str(int(value)): int(value) for value in allowed_ids}
    result = []
    for point in (event or {}).get("selection", {}).get("points", []):
        custom = point.get("customdata", [])
        if custom and str(custom[0]) in allowed:
            value = allowed[str(custom[0])]
            if value not in result:
                result.append(value)
    return result
