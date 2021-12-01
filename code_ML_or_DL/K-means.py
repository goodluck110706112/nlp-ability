"""从零开始实现一个k-means，参考：https://zhuanlan.zhihu.com/p/158776162
"""
import random
import numpy as np
import math
from typing import Dict, Iterable
import collections
from matplotlib import pyplot


def distance(point1, point2):
    # 计算两点欧氏距离
    return math.sqrt(sum((n1 - n2) ** 2 for n1, n2 in zip(point1, point2)))


class KMeans:
    def __init__(self, n_clusters: int, max_iters: int, tolerance: float):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tolerance = tolerance
        self.centers = {}
        self.clusters = collections.defaultdict(list)

    def fit(self, points):
        centers_sampled = random.sample(points, self.n_clusters)
        self.centers = {
            idx: center for idx, center in enumerate(centers_sampled)
        }

        for epoch in range(self.max_iters):
            # assign all points
            for point in points:
                distance_to_centers = [
                    distance(point, center) for center in self.centers.values()
                ]
                cluster_idx = distance_to_centers.index(
                    min(distance_to_centers)
                )
                self.clusters[cluster_idx].append(point)
            # record old center
            pre_centers = dict(self.centers)
            # assign new center

            for center_idx, all_point in self.clusters.items():
                self.centers[center_idx] = np.average(all_point, axis=0)
            # decide whether stop iter
            distance_between_pre_curr = [
                distance(pre, curr)
                for pre, curr in zip(
                    pre_centers.values(), self.centers.values()
                )
            ]
            if max(distance_between_pre_curr) < self.tolerance:
                break

    def predict(self, point):
        distances_to_centers = [
            distance(point, center) for center in self.centers.values()
        ]
        return distances_to_centers.index(min(distances_to_centers))


if __name__ == "__main__":
    x = [[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]]
    k_means = KMeans(n_clusters=2, max_iters=20, tolerance=1)
    k_means.fit(x)
    print(k_means.clusters)
    print(k_means.centers)
    for center in k_means.centers:
        pyplot.scatter(
            k_means.centers[center][0],
            k_means.centers[center][1],
            marker="*",
            s=150,
        )

    for cat in k_means.clusters:
        for point in k_means.clusters[cat]:
            pyplot.scatter(point[0], point[1], c=("r" if cat == 0 else "b"))

    samples = [[2, 1], [6, 9]]
    for sample in samples:
        cat = k_means.predict(sample)
        print(cat)
        pyplot.scatter(
            sample[0], sample[1], c=("r" if cat == 0 else "b"), marker="x"
        )

    pyplot.show()
