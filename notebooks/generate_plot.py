#!/usr/bin/env python
# -*- coding: utf-8 -*-

from plotly import tools
from plotly import graph_objs as go
import numpy as np

from settings.artistinfo import LINKIN_PARK_ALBUMS


coords = []
for i in xrange(1, 5):
    for j in xrange(2):
        coords.append((i, j % 2 + 1))


def rel_freq_plot(df):

    fig = tools.make_subplots(
        rows=4, cols=2,
        subplot_titles=tuple(
            i.title().replace("-", " ") for i in LINKIN_PARK_ALBUMS
        ),
        print_grid=False,
    )

    for i in xrange(len(LINKIN_PARK_ALBUMS)):
        fig.append_trace(
            go.Bar(x=df[i].index, y=df[i]["freq"],
                   showlegend=False),
            coords[i][0], coords[i][-1]
        )

    fig["layout"].update(
        height=2500, width=1000,
        title="Top Relative Frequency of Terms",
        margin=go.Margin(r=150, b=100),
    )

    return fig


def cos_sim_plot(df):

    # set colorscale
    colorscale = [[0, "rgb(255, 255, 255)"], [1.0, "rgb(204, 12, 12)"]]
    fig = tools.make_subplots(
        rows=4, cols=2,
        subplot_titles=tuple(i.title().replace("-", " ")
                             for i in LINKIN_PARK_ALBUMS),
        print_grid=False,
    )

    for i in xrange(len(LINKIN_PARK_ALBUMS)):
        mask = np.zeros_like(df[i].values)
        mask[np.tril_indices_from(mask, k=-1)] = 1
        mask *= df[i].values

        fig.append_trace(
            go.Heatmap(
                z=mask,
                x=list(df[i]),
                y=df[i].index,
                colorscale=colorscale,
                showlegend=False,
                zmin=0,
                zmax=0.6,
                colorbar={
                    "title": "Similarity",
                }
            ),
            coords[i][0], coords[i][-1]
        )

    for attr in fig["layout"]:
        if "xaxis" in attr:
            fig["layout"][attr].update(tickangle=90)

    fig["layout"].update(
        height=2000, width=900,
        title="Cosine Similarity",
        margin=go.Margin(l=190, b=150),
    )

    return fig


def phrase_sent_plot(df):

    values = df["sent_score"]
    colors = [
        "rgba(226, 43, 43, 0.7)" if x < 0
        else "rgba(82, 183, 77, 0.7)" for x in values
    ]

    data = [go.Bar(
        x=values,
        y=df["phrase"],
        orientation="h",
        marker=dict(color=colors),
    )]

    layout = go.Layout(
        autosize=False,
        margin=go.Margin(l=400, pad=4),
    )

    fig = go.Figure(data=data, layout=layout)
    return fig


def doc_sent_plot(df):

    data = []
    for album in LINKIN_PARK_ALBUMS:
        album = album.title().replace("-", " ")
        norm_sent = df["norm_comp"][df["album"] == album]
        data.append(go.Box(
            y=norm_sent,
            name=album.title().replace("-", " "),
            boxpoints="all",
            jitter=0.5,
            whiskerwidth=0.2,
            line=dict(width=1),
            showlegend=False,
        ))

    return data


def phrase_sent_scatter(df):

    data = []
    for album in LINKIN_PARK_ALBUMS:
        album = album.title().replace("-", " ")
        data.append(
            go.Scatter(
                x=df["num_words"][df["album"] == album],
                y=df["sent_score"][df["album"] == album],
                mode="markers",
                marker=dict(
                    size=14,
                    line=dict(width=1),
                    color=album,
                    opacity=0.5
                ),
                name=album,
                text=df.index[df["album"] == album]
            )
        )

    layout = go.Layout(
        title="Sentiment Score of Phrases",
        hovermode="closest",
        xaxis={
            "title": "Number of words",
            "ticklen": 5,
            "zeroline": False,
            "gridwidth": 2,
        },
        yaxis={
            "title": "Sentiment Score",
            "ticklen": 5,
            "gridwidth": 2,
        },
        showlegend=True
    )

    fig = go.Figure(data=data, layout=layout)

    return fig


def valence_arousal_dims():

    data = [go.Scatter(
        x=range(11),
        y=range(11),
        mode="text"
    )]

    layout = {
        "title": "Valence-Arousal Dimensions",
        "xaxis": {
            "title": "Valence (Sentiment)",
            "range": [0, 10],
            "zeroline": False,
            "tick0": 1,
            "dtick": 1,
        },
        "yaxis": {
            "title": "Arousal (Intensity)",
            "range": [0, 10],
            "tick0": 1,
            "dtick": 1,
        },
        "width": 800,
        "height": 800,
        "shapes": [
            # unfilled circle
            {
                "type": "circle",
                "xref": "x",
                "yref": "y",
                "x0": 1,
                "y0": 1,
                "x1": 9,
                "y1": 9,
                "line": {
                    "color": "rgba(50, 171, 96, 1)",
                },
            }, {
                "type": "line",
                "x0": 5,
                "y0": 1,
                "x1": 5,
                "y1": 9,
                "line": {
                    "color": "rgba(0, 0, 0, 0.5)",
                    "dash": "dot",
                },
            }, {
                "type": "line",
                "x0": 1,
                "y0": 5,
                "x1": 9,
                "y1": 5,
                "line": {
                    "color": "rgba(0, 0, 0, 0.5)",
                    "dash": "dot",
                },
            },
        ],
        "annotations": [
            {
                "x": 3,
                "y": 6,
                "text": "Calm",
            }, {
                "x": 3,
                "y": 3,
                "text": "Depression",
            }, {
                "x": 6.5,
                "y": 6,
                "text": "Excitement",
            }, {
                "x": 6.5,
                "y": 3,
                "text": "Anger/Fear",
            },
        ]

    }

    for attr in layout["annotations"]:
        attr.update(
            xref="x",
            yref="y",
            font={
                "family": "Courier New, monospace",
                "size": 20,
                "color": "#000000"
            },
            align="center",
            arrowcolor="#ffffff",
            ax=20,
            ay=-30,
            opacity=0.8
        )

    fig = {
        "data": data,
        "layout": layout,
    }

    return fig


def valence_arousal_plot(df, df1):

    fig = tools.make_subplots(
        rows=4, cols=2,
        subplot_titles=tuple(i.title().replace("-", " ")
                             for i in LINKIN_PARK_ALBUMS),
        print_grid=False
    )

    for i, album in enumerate(LINKIN_PARK_ALBUMS):
        album = album.title().replace("-", " ")
        norm_sentiment = df1[df1["album"] == album]["norm_comp"].tolist()
        norm_sentiment = [(4 * x + 5) for x in norm_sentiment]
        fig.append_trace(
            go.Scatter(
                x=df[df["album"] == album]["arousal_pred"].tolist(),
                y=norm_sentiment,
                mode="markers",
                name=album.title().replace("-", " "),
                text=df[df["album"] == album].index.tolist(),
                marker=dict(
                    size=10,
                    line=dict(width=1),
                    color=album,
                    opacity=0.5
                ),
                error_x=dict(
                    type="data",
                    color="rgba(0, 0, 0, 0.2)",
                    array=df[df["album"] == album]["arousal_std_dev"].tolist(),
                ),
            ),
            coords[i][0], coords[i][-1]
        )

    for attr in fig["layout"]:
        if "xaxis" in attr or "yaxis" in attr:
            fig["layout"][attr].update(
                range=[1, 9],
                tick0=1,
                dtick=1,
            )

    for attr in range(8):
        fig["layout"]["shapes"].append(
            {
                "yref": "y" + str(attr),
                "xref": "x" + str(attr),
                "type": "line",
                "x0": 5,
                "y0": 1,
                "x1": 5,
                "y1": 9,
                "line": {
                    "color": "rgba(0, 0, 0, 0.5)",
                    "dash": "dot",
                },
            },
        )
        fig["layout"]["shapes"].append(
            {
                "yref": "y" + str(attr),
                "xref": "x" + str(attr),
                "type": "line",
                "x0": 1,
                "y0": 5,
                "x1": 9,
                "y1": 5,
                "line": {
                    "color": "rgba(0, 0, 0, 0.5)",
                    "dash": "dot",
                },
            },
        )

    fig["layout"].update(
        height=2000, width=900,
        title="Valence-Arousal Charts",
        showlegend=False,
    )

    return fig
