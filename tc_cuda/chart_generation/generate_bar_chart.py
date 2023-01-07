import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter


def draw_bar_chart(xtick_labels, datasets, dataset_titles,
                   x_label=None,
                   y_label=None,
                   title=None, figure_path=None, figure_width=10, figure_height=5):
    total_width = 0.9
    # if len(datasets[0]) > 7:
    #     total_width = 0.5

    single_width = total_width / len(datasets)

    # sort the labels and data by first dataset ascending order
    sorted_index = np.argsort(np.array(datasets[0]))
    xtick_labels = [xtick_labels[i] for i in sorted_index]
    for i, dataset in enumerate(datasets):
        datasets[i] = [dataset[j] for j in sorted_index]
    label_locations = np.arange(len(xtick_labels))

    fig, ax = plt.subplots(figsize=(figure_width, figure_height))
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
    if len(datasets[0]) > 7:
        ax.set_xticks(label_locations, xtick_labels, rotation=90)
    else:
        ax.set_xticks(label_locations, xtick_labels)
    ax.legend()

    plt.yscale("log")
    plt.gca().yaxis.set_major_formatter(ScalarFormatter())

    fig.tight_layout()
    if figure_path == None:
        figure_path = 'output/bar.png'
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


def show_cuda_souffle():
    dataset_labels = ['CA-HepTh', 'SF.cedge', 'p2p-Gnutella09', 'p2p-Gnutella04', 'cal.cedge', 'TG.cedge', 'OL.cedge',
                      'luxembourg_osm', 'wiki-Vote', 'wing', 'ego-Facebook', 'delaunay_n16',
                      'fe_sphere', 'loc-Brightkite', 'fe_ocean', 'cti', 'fe_body', 'usroads']
    cuda_dataset = [4.3180, 11.2749, 0.7202, 2.0920, 0.4894, 0.1989, 0.1481, 1.3222, 1.1372, 0.0857,
                    0.5442, 1.1374, 13.1590, 15.8805, 138.2379, 0.2953, 47.7587, 364.5549]
    cuda_dataset_title = "CUDA"
    souffle_dataset = [15.206, 17.073, 3.094, 7.537, 0.455, 0.219, 0.181, 2.548, 3.172, 0.193,
                       0.606, 1.612, 20.008, 29.184, 536.233, 1.496, 29.070, 222.761]
    souffle_dataset_title = "Souffle"
    x_label = "Datasets"
    y_label = "Execution Time (seconds in log scale)"
    output_filename = os.path.join("output", "cuda_vs_souffle.png")
    draw_bar_chart(dataset_labels, [cuda_dataset, souffle_dataset], [cuda_dataset_title, souffle_dataset_title],
                   x_label, y_label,
                   figure_path=output_filename, figure_width=12, figure_height=6)


if __name__ == "__main__":
    # show_cuda_pinned_unified()
    show_cuda_souffle()
