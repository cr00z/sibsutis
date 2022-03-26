import numpy.linalg
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from typing import Any, Callable, List, Tuple

numpy.random.seed(42)

# Limits for _k_NN
LOW_K = 1
HIGH_K = 10
LOW_H = 0.1
HIGH_H = 1
STEP_H = 0.1
SIGMA = 1e-18


# Visualization

def show_data_sample(showed_data: np.ndarray) -> None:
    """Show distribution by pyplot"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(showed_data[:, 0], showed_data[:, 1], c=showed_data[:, 2])
    plt.xlabel("MrotInHour")
    plt.ylabel("Salary")
    plt.show()


def get_classes_counts_str(inst_data: np.ndarray) -> str:
    """Counts the number of instances of each class and form the report str"""
    return "Class 0: {}, class 1: {}".format(
        np.count_nonzero(inst_data[:, 2] == 0),
        np.count_nonzero(inst_data[:, 2] == 1)
    )


def show_results(banner: str, results: List[Tuple[int, float]]) -> None:
    print(f"\n--- {banner} ---")
    for k, metric in results:
        print("{:2} | {}".format(k, metric))


# Utils

def get_tqdm_pbar(iterable: np.ndarray, param: Any) -> tqdm:
    """Get progress bar iterator with description"""
    pbar = tqdm.tqdm(iterable, position=0, leave=False)
    pbar.set_description(f"Processing {param}")
    return pbar


# Sample preparation

def sample_split(dt: np.ndarray, train_size: float) \
        -> Tuple[np.ndarray, np.ndarray]:
    """Split data to train and test. The split tries to equalize the number
    of elements of each class in the train"""
    np.random.shuffle(dt)
    class0_idxs = np.where(dt[:, 2] == 0)[0]
    class1_idxs = np.where(dt[:, 2] == 1)[0]

    train_len = int(dt.shape[0] * train_size)
    class0_train_len = int(min(train_len / 2, class0_idxs.shape[0]))
    class1_train_len = int(min(train_len / 2, class1_idxs.shape[0]))
    if class0_train_len < class1_train_len:
        class1_train_len = train_len - class0_train_len
    if class1_train_len < class0_train_len:
        class0_train_len = train_len - class1_train_len

    train_idxs = np.concatenate(
        (class0_idxs[:class0_train_len], class1_idxs[:class1_train_len]))
    test_idxs = np.concatenate(
        (class0_idxs[class0_train_len:], class1_idxs[class1_train_len:]))
    np.random.shuffle(train_idxs)
    np.random.shuffle(test_idxs)

    return dt[train_idxs], dt[test_idxs]


def z_normalize(initial: np.ndarray,
                means: np.ndarray = None, stds: np.ndarray = None)\
        -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalization, formula: z_i = ( x_i - mu ) / sigma"""
    if means is None:
        means = np.mean(initial, axis=0)
    if stds is None:
        stds = np.std(initial, axis=0)
    return (initial - means) / stds, means, stds


# Algorithms

def get_sorted_distances(x_train: np.ndarray, y_train: np.ndarray,
                         x_test_sample: np.ndarray) -> np.ndarray:
    """Euclidean distance sorted from minimum to maximum"""
    distances = numpy.linalg.norm(x_train - x_test_sample, axis=1)
    dist_classes = np.concatenate((distances.reshape((-1, 1)), y_train), axis=1)
    return dist_classes[dist_classes[:, 0].argsort()]


def knn(x_train: np.ndarray, y_train: np.ndarray,
        x_test_sample: np.ndarray, k: int) -> int:
    """kNN realization"""
    dist_sorted = get_sorted_distances(x_train, y_train, x_test_sample)
    class0_count = np.count_nonzero(dist_sorted[:k, 1] == 0)
    return 0 if (class0_count > k / 2) else 1


def kwnn(x_train: np.ndarray, y_train: np.ndarray,
        x_test_sample: np.ndarray, k: int) -> int:
    """Weighted kNN realisation"""
    dist_sorted = get_sorted_distances(x_train, y_train, x_test_sample)
    classes = np.zeros(2, dtype=float)
    for i in range(1, k + 1):
        # classes[int(dist_sorted[i, 1])] += pow(q, i)
        classes[int(dist_sorted[i, 1])] += (k + 1 - i) / k
    return classes.argmax()


def quartic(r: float) -> float:
    """Quartic kernel"""
    return 15 / 16 * (1 - r ** 2) ** 2


def parsen_fixed_h(x_train: np.ndarray, y_train: np.ndarray,
                   x_test_sample: np.ndarray, h: float) -> int:
    """Parsen with fixed window"""
    dist_sorted = get_sorted_distances(x_train, y_train, x_test_sample)
    classes = np.array((0, 0), dtype=float)
    for i in range(dist_sorted.shape[0]):
        r = dist_sorted[i, 0] / h
        if r > 1:
            break
        classes[int(dist_sorted[i, 1])] += quartic(r)
    return classes.argmax()


def parsen_relative_h(x_train: np.ndarray, y_train: np.ndarray,
                      x_test_sample: np.ndarray, k: int) -> int:
    """Parsen with relative window"""
    dist_sorted = get_sorted_distances(x_train, y_train, x_test_sample)
    # r = distance / (h + sigma) [r <= 1]
    h = dist_sorted[k - 1, 0] + SIGMA
    classes = np.array((0, 0), dtype=float)
    for i in range(round(k)):
        r = dist_sorted[i, 0] / h
        classes[int(dist_sorted[i, 1])] += quartic(r)
    return classes.argmax()


# Metrics

def accuracy(y_test: np.ndarray, y_pred: np.ndarray) -> float:
    """Accuracy metric"""
    diff = np.sum(np.abs(y_test - y_pred))
    return 1 - diff / y_test.shape[0]


def loo(full_data: np.ndarray, knn_algo: Callable, param: Any) -> float:
    """LOO metric"""
    x_data_norm, _, _ = z_normalize(full_data[:, :2])
    y_data = full_data[:, 2:]
    y_pred = []
    pbar = get_tqdm_pbar(x_data_norm, param)
    for indx, test_sample in enumerate(pbar):
        x_wo_element = np.delete(x_data_norm, indx, axis=0)
        y_wo_element = np.delete(y_data, indx, axis=0)
        y_pred.append(knn_algo(x_wo_element, y_wo_element, test_sample, param))
    return accuracy(y_data.flatten(), np.array(y_pred))


# Process

def experiment(title: str,
               all_data: np.ndarray,
               x_train: np.ndarray,
               y_train: np.ndarray,
               x_test: np.ndarray,
               y_test: np.ndarray,
               knn_algo: Callable,
               start: Any,
               stop: Any,
               step: Any = 1) -> None:
    # LOO
    result = []
    for k in np.arange(start, stop, step):
        result.append((k, loo(all_data, knn_algo, k)))
    show_results(f"{title} LOO", result)
    # Predict
    k_max = sorted(result, reverse=True, key=lambda x: (x[1], x[0]))[0][0]
    pbar = get_tqdm_pbar(x_test, k_max)
    y_pred = [knn_algo(x_train, y_train, sample, k_max) for sample in pbar]
    accuracy_metric = accuracy(y_test.flatten(), np.array(y_pred))
    show_results(f"{title} predict", [(k_max, accuracy_metric)])


if __name__ == '__main__':
    data = pd.read_csv("data1.csv").to_numpy()
    show_data_sample(data[:1000])
    print("Distribution of classes in the data - ",
          get_classes_counts_str(data))

    train, test = sample_split(data, train_size=0.667)
    print("Distribution of classes in the train - ",
          get_classes_counts_str(train))
    y_train = train[:, 2:]
    y_test = test[:, 2:]

    # Normalization
    x_train_norm, train_means, train_stds = z_normalize(train[:, :2])
    x_test_norm, _, _ = z_normalize(test[:, :2], train_means, train_stds)

    experiment("kNN", data, x_train_norm, y_train,
               x_test_norm, y_test, knn, LOW_K, HIGH_K + 1)
    experiment("Weighted kNN", data, x_train_norm, y_train,
               x_test_norm, y_test, kwnn, LOW_K, HIGH_K + 1)
    experiment("Parsen window", data, x_train_norm, y_train,
               x_test_norm, y_test, parsen_fixed_h, LOW_H, HIGH_H + SIGMA, STEP_H)
    experiment("Parsen relative window", data, x_train_norm, y_train,
               x_test_norm, y_test, parsen_relative_h, LOW_K, HIGH_K + 1)
