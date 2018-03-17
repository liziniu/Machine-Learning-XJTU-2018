from sklearn.cluster import KMeans
from sklearn.datasets import samples_generator
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing


x, y = samples_generator.make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=5,
    n_redundant=1,
    n_repeated=1,
    n_classes=10,
    random_state=1,
)

# print(x.shape)     (1000, 10)
# print(y.shape)     (1000, )


# description:
# the k_means implemented by sklearn module
def sklearn_k_means(x, n_clusters=10):
    k_means = KMeans(n_clusters)
    k_means.fit(x)
    y_predict = k_means.predict(x)

    accuracy = np.mean((y == y_predict).astype(np.int))
    print(accuracy)


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
                plt.figure()
                plt.plot(self.distances)
                plt.title("k = {}".format(self.n_cluster))
                plt.xlabel('Epoches')
                plt.ylabel('Distance')
                plt.savefig("image_segmentation_loss.png")
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

    # input: x, data to be visualized; shape:[None, n_features]
    #        dims, the dimension to be plotted; shape:[dim_1, dim_2]
    #        y(optimal), the cluster of each data; shape:[None, n_clusters]
    def visualize(self, x, dims, y=None):
        # plt.style.use('seaborn')
        if y is None:
            plt.figure()
            first_dim = dims[0]
            second_dim = dims[1]
            plt.scatter(x[:, first_dim], x[:, second_dim])
            plt.show()
        else:
            plt.figure()
            first_dim = dims[0]
            second_dim = dims[1]
            plt.scatter(x[:, first_dim], x[:, second_dim], c=y)
            plt.show()


if __name__ == "__main__":
    # print(sklearn_k_means(x))
    for i in range(10):
        k_means = K_Means()
        result, distance = k_means.train(x, 10, seed=i)
    




