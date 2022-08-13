import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

def draw_bar_chart(xtick_labels, first_dataset,
                   first_dataset_title,
                   second_dataset,
                   second_dataset_title,
                   x_label=None,
                   y_label=None,
                   title=None):
    label_locations = np.arange(len(xtick_labels))
    width = 0.35

    # plt.figure()
    fig, ax = plt.subplots(figsize=(8, 4))
    rects1 = ax.bar(label_locations - width / 2, first_dataset, width,
                    label=first_dataset_title)
    rects2 = ax.bar(label_locations + width / 2, second_dataset, width,
                    label=second_dataset_title)

    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)
    ax.set_xticks(label_locations, xtick_labels)
    ax.legend()

    ax.bar_label(rects1, fmt='%.2f')
    ax.bar_label(rects2, fmt='%.2f')

    plt.yscale("log")
    plt.gca().yaxis.set_major_formatter(ScalarFormatter())

    fig.tight_layout()
    figure_path = 'screenshots/tc.png'
    fig.savefig(figure_path, dpi=600, bbox_inches="tight")
    print(f"Figure saved in {figure_path}")



if __name__ == "__main__":
    dataset_labels = ['SF.cedge', 'p2p-Gnutella09', 'p2p-Gnutella04',
                      'cal.cedge', 'TG.cedge', 'OL.cedge']
    first_dataset = [64.235956, 3.881797, 14.104549, 3.883682, 1.191066,
                     0.557429]
    first_dataset_title = "cuDF"
    second_dataset = [4582.067635, 167.143291, 569.249333, 5.769209, 1.400392,
                      0.474801]
    second_dataset_title = "Pandas"
    x_label = "Datasets"
    y_label = "Execution Time (seconds in log scale)"
    title = "cuDF and Pandas DF execution time comparison"
    draw_bar_chart(dataset_labels, first_dataset, first_dataset_title,
                   second_dataset, second_dataset_title,
                   x_label, y_label)
