
from analysis.chaos_results.data_loading_funcs import load_paper_nbeats_smape, load_our_nbeats_smape, load_gams_smape
import plotly.graph_objects as go
import numpy as np

def main():
    paper_smape_results, paper_smape = load_paper_nbeats_smape()

    our_smape_results, our_horizon_results = load_our_nbeats_smape()

    gams_smapes, gams_horizons = load_gams_smape()

    nbeats_mean_smapes_list = np.zeros((len(our_smape_results)))
    nbeats_std_smapes_list = np.zeros((len(our_smape_results)))
    nbeats_paper_list = np.zeros((len(our_smape_results)))
    gams_smapes_list = np.zeros((len(our_smape_results)))
    labels = []
    for i, system in enumerate(our_smape_results.keys()):
        gams_smapes_list[i] = gams_smapes[system] if system in gams_smapes.keys() else np.nan
        nbeats_paper_list[i] = paper_smape_results[system][0]
        nbeats_mean_smapes_list[i] = np.nanmean(our_smape_results[system])
        nbeats_std_smapes_list[i] = np.nanstd(our_smape_results[system])
        labels.append(system)

    order = np.argsort(nbeats_paper_list)

    nbeats_paper_list = nbeats_paper_list[order]
    nbeats_mean_smapes_list = nbeats_mean_smapes_list[order]
    nbeats_std_smapes_list = nbeats_std_smapes_list[order]
    gams_smapes_list = gams_smapes_list[order]
    labels = [labels[i] for i in order]

    fig = go.Figure(
        go.Scatter(
            x=labels,
            y=nbeats_mean_smapes_list,
            mode="markers",
            name="Our NBeats (mean ± std)",
            error_y=dict(type="data", array=nbeats_std_smapes_list, color="red", visible=True),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=labels,
            y=nbeats_paper_list,
            mode="markers",
            name="Paper NBeats",
            marker=dict(symbol="square", color="grey", size=9),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=labels,
            y=gams_smapes_list,
            mode="markers",
            name="Gams",
            marker=dict(symbol="diamond", color="green", size=11),
        )
    )

    fig.update_xaxes(
        tickangle=-90,
        tickmode="linear",   # one tick per category...
        dtick=1,             # ...every single one
        tickfont=dict(size=9),
    )
    fig.update_layout(
        yaxis_title="sMAPE",
        width=max(1200, 14 * len(labels)),  # ~14 px per label
        height=650,
        margin=dict(b=250),  # more room for long vertical labels
    )

    fig.show()


if __name__ == "__main__":
    main()