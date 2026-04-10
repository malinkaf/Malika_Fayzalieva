import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def get_optimal_k(X, max_k):
    best_k = 2
    best_score = -1.0
    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X)
        score = silhouette_score(X, labels)
        if score > best_score:
            best_score = score
            best_k = k
    return best_k


def kmeans_from_scratch(X, k, pca, max_iter=30, tol=1e-4):
    np.random.seed(7)
    centroids = X[np.random.choice(X.shape[0], k, replace=False)].copy()

    X_2d = pca.transform(X)

    plt.ion()
    fig, ax = plt.subplots(figsize=(7, 5))

    for step in range(max_iter):

        labels = np.zeros(X.shape[0], dtype=int)
        for i in range(X.shape[0]):
            dists = np.sqrt(np.sum((centroids - X[i]) ** 2, axis=1))
            labels[i] = np.argmin(dists)

        ax.clear()
        sc = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='tab10',
                        s=35, alpha=0.75, edgecolor='k', linewidth=0.5)

        cent_2d = pca.transform(centroids)
        ax.scatter(cent_2d[:, 0], cent_2d[:, 1], c='red', marker='X',
                   s=180, linewidths=2, label='Центроиды')

        ax.set_title(f"Шаг {step + 1}")
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.legend(loc='upper right')
        ax.grid(True, linestyle=':', alpha=0.6)

        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(1.0)  

        new_centroids = np.zeros_like(centroids)
        for c in range(k):
            cluster_mask = (labels == c)
            if np.any(cluster_mask):
                new_centroids[c] = X[cluster_mask].mean(axis=0)
            else:
                new_centroids[c] = centroids[c]

        shift = np.sqrt(np.sum((new_centroids - centroids) ** 2))
        centroids = new_centroids.copy()

        if shift < tol:
            print(f"Сходимость достигнута на шаге {step + 1}. Сдвиг: {shift:.5f}")
            break
    else:
        print(f"Превышен лимит итераций ({max_iter}).")

    plt.ioff()
    plt.show()
    return labels, centroids


def print_label_match(pred, true, n_classes=3):
    """Выводит таблицу пересечения предсказанных и истинных меток"""
    table = np.zeros((n_classes, n_classes), dtype=int)
    for p, t in zip(pred, true):
        table[p, t] += 1
    print("\nМатрица соответствия кластеров и настоящих классов:")
    print("       Setosa  Versicolor  Virginica")
    for i in range(n_classes):
        print(f"Кластер {i}:  {table[i, 0]:>5}      {table[i, 1]:>5}       {table[i, 2]:>5}")


def main():
    X, y_true = load_iris(return_X_y=True)

    optimal_k = get_optimal_k(X,10) 
    print(f"Оптимальное количество кластеров: {optimal_k}")

    pca = PCA(n_components=2)
    pca.fit(X)

    print("\nЗапуск K-Means (реализация с нуля)...")
    y_pred, final_centroids = kmeans_from_scratch(X, optimal_k, pca)

    print_label_match(y_pred, y_true)

    print("\nИтоговые центроиды в исходном пространстве признаков:")
    print(final_centroids)


if __name__ == "__main__":
    main()