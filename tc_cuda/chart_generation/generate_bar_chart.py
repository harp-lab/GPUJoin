import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter


def draw_bar_chart(xtick_labels, datasets, dataset_titles,
                   x_label=None,
                   y_label=None,
                   title=None, figure_path=None):
    label_locations = np.arange(len(xtick_labels))
    total_width = 0.9
    single_width = total_width / len(datasets)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i in range(len(datasets)):
        mid = len(datasets) / 2
        if i < mid:
            rect = ax.bar(label_locations - (mid - i) * single_width + (single_width / 2),
                          datasets[i], single_width,
                          label=dataset_titles[i])
        else:
            rect = ax.bar(label_locations + (i - mid) * single_width + (single_width / 2),
                          datasets[i], single_width,
                          label=dataset_titles[i])
        ax.bar_label(rect, fmt='%.2f')

    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)
    ax.set_xticks(label_locations, xtick_labels)
    ax.legend()

    plt.yscale("log")
    plt.gca().yaxis.set_major_formatter(ScalarFormatter())

    fig.tight_layout()
    if figure_path == None:
        figure_path = 'screenshots/bar.png'
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved in {figure_path}")


def show_cuda_pinned_unified():
    dataset_labels = ['CA-HepTh', 'SF.cedge', 'p2p-Gnutella09', 'p2p-Gnutella04', 'cal.cedge', 'TG.cedge', 'OL.cedge']
    pinned_dataset = [4.3180, 11.2749, 0.7202, 2.0920, 0.4894, 0.1989, 0.1481]
    pinned_dataset_title = "Pinned memory"
    unified_dataset = [11.4198, 45.7082, 2.2018, 7.3043, 1.1011, 0.3776, 0.3453]
    unified_dataset_title = "Unified memory"
    x_label = "Datasets"
    y_label = "Execution Time (seconds in log scale)"
    output_filename = os.path.join("output", "pinned_vs_unified.png")
    draw_bar_chart(dataset_labels, [pinned_dataset, unified_dataset], [pinned_dataset_title, unified_dataset_title],
                   x_label, y_label,
                   figure_path=output_filename)


if __name__ == "__main__":
    show_cuda_pinned_unified()
