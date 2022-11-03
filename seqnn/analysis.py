import numpy as np
import matplotlib.pyplot as plt
import seqnn.model.likelihood
from seqnn.utils import ensure_list


def plot_prediction(
    model,
    past,
    future,
    tags_pred,
    tags_control=None,
    lims={},
    titles={},
    ncols=None,
    width=3,
    height=2,
):
    plt.style.use("ggplot")

    assert isinstance(
        model.get_likelihood(), seqnn.model.likelihood.LikGaussian
    ), "Prediction visualization implemented only for Gaussian likelihood"
    pred_all = model.predict(past, future)

    tags_pred = ensure_list(tags_pred)
    tags_control = ensure_list(tags_control)
    nplots = len(tags_pred + tags_control)

    if ncols is None:
        ncols = int(np.ceil(np.sqrt(nplots)))
    nrows = int(np.ceil(nplots / ncols))
    fig, ax = plt.subplots(nrows, ncols, figsize=(ncols * width, nrows * height))
    if nplots > 1:
        ax = ax.flatten()

    index = 0
    for tag in tags_pred:
        obs_past = model.get_tags(past, tag)
        obs_future = model.get_tags(future, tag)
        pred = model.get_tags(pred_all["mean"], tag)
        assert (
            pred.shape[0] == 1
        ), "Prediction visualization for batch_size > 0 not implemented"
        obs_past = obs_past[0, :, 0]
        obs_future = obs_future[0, :, 0]
        pred = pred[0, :, 0]

        x_past = np.arange(-len(obs_past), 0) + 1
        x_future = np.arange(len(obs_future)) + 1
        ax[index].plot(x_past, obs_past, ".", color="k")
        ax[index].plot(x_future, obs_future, ".", color="gray")
        ax[index].plot(x_future, pred, "-", color="C1")
        if tag in lims:
            ax[index].set_ylim(lims[tag])
        title = titles[tag] if tag in titles else tag
        ax[index].set_title(title)
        index += 1

    for tag in tags_control:
        obs_past = model.get_tags(past, tag)[0, :, 0]
        obs_future = model.get_tags(future, tag)[0, :, 0]
        x_past = np.arange(-len(obs_past), 0) + 1
        x_future = np.arange(len(obs_future)) + 1
        ax[index].plot(x_past, obs_past, ".", color="k")
        ax[index].plot(x_future, obs_future, ".", color="gray")
        if tag in lims:
            ax[index].set_ylim(lims[tag])
        title = titles[tag] if tag in titles else tag
        ax[index].set_title(title)
        index += 1
    plt.tight_layout()
