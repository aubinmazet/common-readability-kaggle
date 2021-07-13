import seaborn as sns
from matplotlib import pyplot as plt


def plot_metrics(history):
    metric_list = list(history.keys())
    size = len(metric_list) // 2
    fig, axes = plt.subplots(size, 1, sharex="col", figsize=(20, size * 5))
    axes = axes.flatten()

    for index in range(len(metric_list) // 2):
        metric_name = metric_list[index]
        val_metric_name = metric_list[index + size]
        axes[index].plot(history[metric_name], label="Train %s" % metric_name)
        axes[index].plot(history[val_metric_name], label="Validation %s" % metric_name)
        axes[index].legend(loc="best", fontsize=16)
        axes[index].set_title(metric_name)

    plt.xlabel("Epochs", fontsize=16)
    sns.despine()
    plt.show()
