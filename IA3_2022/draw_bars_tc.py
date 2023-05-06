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

    fig, ax = plt.subplots(figsize=(8, 4))
    for i in range(len(datasets)):
        mid = len(datasets) / 2
        if i < mid:
            rect = ax.bar(label_locations - (mid - i) * single_width + (single_width/2),
                          datasets[i], single_width,
                          label=dataset_titles[i])
        else:
            rect = ax.bar(label_locations + (i - mid) * single_width + (single_width/2),
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
        figure_path = 'screenshots/tc.png'
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved in {figure_path}")


def show_cudf_pandas_tc():
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
    draw_bar_chart(dataset_labels,
                   [first_dataset, second_dataset],
                   [first_dataset_title, second_dataset_title, ],
                   x_label, y_label,
                   figure_path="screenshots/tc_cudf_pandas.png")


def show_cuda_cudf_pandas_tc():
    dataset_labels = ['SF.cedge', 'p2p-Gnutella09', 'p2p-Gnutella04',
                      'cal.cedge', 'TG.cedge', 'OL.cedge']
    cuda_dataset = [69.9199, 4.7001, 22.2563, 1.241, 0.3603,
                    0.3052]
    cuda_dataset_title = "CUDA"
    first_dataset = [64.235956, 3.881797, 14.104549, 3.883682, 1.191066,
                     0.557429]
    first_dataset_title = "cuDF"
    second_dataset = [4582.067635, 167.143291, 569.249333, 5.769209, 1.400392,
                      0.474801]
    second_dataset_title = "Pandas"
    x_label = "Datasets"
    y_label = "Execution Time (seconds in log scale)"
    draw_bar_chart(dataset_labels,
                   [cuda_dataset, first_dataset, second_dataset],
                   [cuda_dataset_title, first_dataset_title,
                    second_dataset_title],
                   x_label, y_label,
                   figure_path="screenshots/transitive_closure_3.png")

def show_cuda_cudf_pandas_triangle():
    dataset_labels = ['roadNet-CA', 'roadNet-TX', 'roadNet-PA', 'SF.cedge', 'p2p-Gnutella09']
    cuda_dataset = [2.3305, 0.6461, 0.5132, 0.0444, 0.0101]
    cuda_dataset_title = "CUDA"
    first_dataset = [0.1593, 0.1180, 0.1084, 0.0877, 0.0486]
    first_dataset_title = "cuDF"
    second_dataset = [3.0106, 1.9466, 1.4777, 0.1482, 0.0256]
    second_dataset_title = "Pandas"
    x_label = "Datasets"
    y_label = "Execution Time (seconds in log scale)"
    draw_bar_chart(dataset_labels,
                   [cuda_dataset, first_dataset, second_dataset],
                   [cuda_dataset_title, first_dataset_title,
                    second_dataset_title],
                   x_label, y_label,
                   figure_path="screenshots/triangle_counting_3.png")

if __name__ == "__main__":
    show_cuda_cudf_pandas_tc()
    show_cuda_cudf_pandas_triangle()
