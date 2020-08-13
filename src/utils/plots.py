from typing import Tuple

import altair as alt
import numpy as np
import pandas as pd
from scipy.stats import zscore

from src.gabor_analysis.gabor_fit import GaborFit
from src.spikeloader import SpikeLoader


def gabor_interactive(
    f: SpikeLoader, g: GaborFit, n_samples: int = 500, rf_size: Tuple[int, int] = (150, 50)
) -> alt.Chart:
    names = [
        "sigma - size",
        "theta - rot",
        "lambda - freq",
        "gamma - shape",
        "phi - phase",
        "gabor_x",
        "gabor_y",
        "corr",
    ]
    df = pd.DataFrame(data=g.params_fit, columns=GaborFit.KEY.keys()).join(f.pos)
    df = g.params_fit.copy()
    df.columns = names
    df = df.join(f.pos)
    clip = 5
    df["imgs"] = list(
        np.rint(
            np.clip(zscore(g.rf_pcaed, axis=(1, 2)), -clip, clip) * (128 / clip) + 128
        ).astype(
            np.uint8
        )  # Convert to 8-bit image.
    )

    brush = alt.selection_interval(resolve="global")
    base = alt.Chart(df.sample(n_samples))
    scatter = (
        base.mark_point(stroke="gray", strokeWidth=0.8, strokeOpacity=0.4)
        .encode(
            x="x",
            y="y",
            fill=alt.condition(
                brush,
                "gabor_x",
                alt.ColorValue("gray"),
                scale=alt.Scale(scheme="redyellowblue"),
                sort="descending",
            ),
            opacity=alt.condition(brush, alt.value(0.9), alt.value(0.05)),
        )
        .add_selection(brush)
    )

    def make_int_hist(*args, **kwargs):
        hist = base.mark_bar()

        background = (
            hist.transform_joinaggregate(total="count(*)")
            .transform_calculate(pct="1 / datum.total")
            .encode(x=alt.X(*args, **kwargs), y=alt.Y("sum(pct):Q"), color=alt.value("#ddd"))
            .add_selection(brush)
        )

        highlight = (
            hist.transform_filter(brush)
            .transform_joinaggregate(total="count(*)")
            .transform_calculate(pct="1 / datum.total")
            .encode(x=alt.X(*args, **kwargs), y=alt.Y("sum(pct):Q"),)
        )

        return (background + highlight).properties(width=260, height=160)

    imgs = (
        base.transform_window(index="count()")
        .transform_filter(brush)
        .transform_sample(12)
        .transform_flatten(["imgs"])
        .transform_window(row="count()", groupby=["index"])
        .transform_flatten(["imgs"])
        .transform_window(column="count()", groupby=["index", "row"],)
        .mark_rect()
        .encode(
            alt.X("column:O", axis=None),
            alt.Y("row:O", axis=None),
            alt.Color("imgs:Q", scale=alt.Scale(scheme="redblue"), legend=None),
            alt.Facet("index:N", columns=3),
        )
        .properties(width=rf_size[0], height=rf_size[1])
    )

    chart = (
        (scatter | imgs).properties(
            title={"text": "Physical Positions", "subtitle": "Color depicts Gabor x center"}
        )
        & make_int_hist(alt.repeat("column"), type="quantitative", bin=alt.Bin(maxbins=40))
        .repeat(column=["sigma - size", "lambda - freq", "theta - rot"])
        .properties(title="Gabor Fit Parameters")
        & make_int_hist(
            alt.repeat("column"), type="quantitative", bin=alt.Bin(maxbins=40)
        ).repeat(column=["phi - phase", "gabor_x", "gabor_y"])
    )

    chart.resolve_scale(color="independent").resolve_legend(color="independent")
    return chart
