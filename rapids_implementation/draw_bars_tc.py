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
    fig, ax = plt.subplots(figsize=(10, 6))
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

    ax.bar_label(rects1, fmt='%.2f', padding=5)
    ax.bar_label(rects2, fmt='%.2f', padding=5)

    plt.yscale("log")
    plt.gca().yaxis.set_major_formatter(ScalarFormatter())

    fig.tight_layout()
    # plt.figure()
    plt.show()


if __name__ == "__main__":
    dataset_labels = ['p2p-Gnutella09', 'p2p-Gnutella04',
                      'cal.cedge', 'SF.cedge', 'TG.cedge', 'OL.cedge']
    first_dataset = [4.636441, 13.969293, 4.054477, 64.265064, 1.157264,
                     0.544352]
    first_dataset_title = "cuDF"
    second_dataset = [166.969241, 577.280656, 5.427228, 4650.348536, 1.609337,
                      0.490264]
    second_dataset_title = "Pandas"
    x_label = "Datasets"
    y_label = "Execution Time (s)"
    title = "cuDF and Pandas DF execution time comparison"
    draw_bar_chart(dataset_labels, first_dataset, first_dataset_title,
                   second_dataset, second_dataset_title,
                   x_label, y_label)

