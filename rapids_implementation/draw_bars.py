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
    dataset_labels = ['File I/O', 'Rename',
                      'Join', 'Deduplication', 'Union']
    first_dataset = [7.532238, 0.031103, 2.354040, 4.165711, 0.345340]
    first_dataset_title = "cuDF"
    second_dataset = [67.287993, 1.622508, 80.349599, 218.142479, 2.469050]
    second_dataset_title = "Pandas"
    x_label = "Operations"
    y_label = "Execution Time (s)"
    title = "CUDF and Pandas time for California road network dataset"
    draw_bar_chart(dataset_labels, first_dataset, first_dataset_title,
                   second_dataset, second_dataset_title,
                   x_label, y_label)
