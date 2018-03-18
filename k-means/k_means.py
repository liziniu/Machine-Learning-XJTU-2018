from sklearn.cluster import KMeans
from sklearn.datasets import samples_generator
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.manifold import TSNE


x, y = samples_generator.make_classification(
    n_samples=1000,
    n_features=10,
    n_redundant=2,
    n_clusters_per_class=1,
    n_classes=3,
    random_state=1,
)

# print(x.shape)     (1000, 10)
# print(y.shape)     (1000, )


# description:
# the k_means implemented by sklearn module
def sklearn_k_means(x, n_clusters=3):
    k_means = KMeans(n_clusters)
    k_means.fit(x)
    y_predict = k_means.predict(x)

    return y_predict


# description:
# the k_means implemented by hand-crafted
class K_Means:
    # preprocessed
    def __init__(self):
        self.x = None
        self.n_cluster = None
        self.res = {}
        self.center_points = None
        self.max_train_counter = None
        self.dist_sum = 0
        self.train_counter = 1
        self.x_to_y = None
        self.img = False
        self.seed = None
        self.distances = []
        pass

    # input:    x, train data; shape:[None, n_features]
    #           n_cluster, the number of clusters; shape:[None, n_cluster]
    # output:   self.x_to_y, the cluster each x belongs to; np.array: [None, n_cluster]
    #           self.center_points: the center of cluster; dictionary {index: center point}
    def train(self, x, n_cluster, max_train_iteration=30, img=False, seed=1):
        self.img = img
        self.seed = seed
        self.x = self._preprocess(x)
        self.max_train_counter = max_train_iteration
        self.n_cluster = n_cluster
        self.center_points = self._get_initial_point()   # generate self.center_points
        self._re_arrange()      # generate self.res
        while True:
            if self.train_counter > self.max_train_counter:
                print("训练结束！")
                print("distances sum: ", self.dist_sum)
                # self._plot_distance()
                return self.x_to_y, self.center_points
            else:
                print("after {} epoch, the distances sum is {}".format(self.train_counter, self.dist_sum))
                # self.check()
                self.distances.append(self.dist_sum)
                self.train_counter += 1
                self.dist_sum = 0

                self._re_find_center()
                self.res = {}
                self._re_arrange()

    # description:
    # check out whether there are error
    def check(self):
        c = 0
        for xs in self.res.values():
            c += xs.shape[0]
        print("c: ", c)
        print("res.keys", self.res.keys())

    # description:
    # arrange all data to its nearest cluster
    def _re_arrange(self):
        self.x_to_y = []
        for data in self.x:
            idx, distance = self._get_cluster(data)
            self.dist_sum += distance
            if idx not in self.res.keys():
                self.res[idx] = np.array(data[np.newaxis, :])
            else:
                self.res[idx] = np.append(self.res[idx], data[np.newaxis, :], axis=0)
            self.x_to_y.append(idx)
        return self.res

    # description:
    # from all data to find n_clusters center points
    def _re_find_center(self):
        i = 0
        for xs in self.res.values():
            new_center = self._get_center(xs)
            self.center_points[i] = new_center
            i += 1

    # input: xs, a cluster of data
    # output: the center point of these data
    def _get_center(self, xs):
        center_point = np.mean(xs, axis=0, keepdims=True)
        return center_point

    # description:
    # randomly find n_cluster points from all data
    def _get_initial_point(self):
        np.random.seed(self.seed)
        order = np.random.choice(self.x.shape[0], self.n_cluster)
        value = self.x[order]
        key = list(range(self.n_cluster))
        center_point = dict(zip(key, value))
        return center_point

    # input: x, single data
    # output: the cluster of x belongs to;
    #         the distance between x and its center
    def _get_cluster(self, x):
        min_dist = 1e5
        cluster = None
        i = 0
        for center_point in self.center_points.values():
            dist = np.linalg.norm(center_point - x)
            if dist < min_dist:
                min_dist = dist
                cluster = i
            i += 1
        return cluster, min_dist

    # input:    x, test data; shape: [None, n_features]
    # output:   the cluster of each data; shape:[None, ]
    def predict(self, x):
        x = self._preprocess(x)     # normalize
        y = []
        for data in x:
            idx, distance = self._get_cluster(data)
            y.append(idx)
        return np.array(y)

    # description:
    # normalize the data
    def _preprocess(self, x):
        if self.img:
            return x / 255.0
        else:
            return preprocessing.normalize(x, axis=0)

    # description:
    # plot the distance change during train
    def _plot_distance(self):
        plt.figure()
        plt.plot(self.distances)
        plt.xlabel('epoch')
        plt.ylabel('distances')
        plt.title('seed={}'.format(self.seed))
        plt.show()


if __name__ == "__main__":
    y_sklearn = sklearn_k_means(x)
    y_myself = []
    for i in [0, 1]:
        k_means = K_Means()
        x_to_y, center_points = k_means.train(x, 3, seed=i)
        y_myself.append(x_to_y)


    # t-sne for visualizing high dimensional data
    t_sne = TSNE(n_components=2)
    y_sne = t_sne.fit_transform(x)

    # use the first two dimensional feature to visualize raw data
    plt.figure()
    plt.scatter(x[:, 0], x[:, 1], c=y)
    plt.title('raw data')
    plt.savefig('raw data.png')

    # compare the clustering results
    plt.subplot(2, 2, 1)
    plt.scatter(y_sne[:, 0], y_sne[:, 1], c=y)
    plt.title('raw data')
    plt.subplot(2, 2, 2)
    plt.scatter(y_sne[:, 0], y_sne[:, 1], c=y_sklearn)
    plt.title('k-means by scikit learn')
    plt.subplot(2, 2, 3)
    plt.scatter(y_sne[:, 0], y_sne[:, 1], c=y_myself[0])
    plt.title('k-means by myself(seed=0)')
    plt.subplot(2, 2, 4)
    plt.scatter(y_sne[:, 0], y_sne[:, 1], c=y_myself[1])
    plt.title('k-means by myself(seed=1)')
    plt.show()
    plt.savefig('k-means results.png')



