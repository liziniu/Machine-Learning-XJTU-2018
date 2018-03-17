from scipy import misc
from k_means import K_Means
import numpy as np
import matplotlib.pyplot as plt

img_ori = misc.imread('face.jpg')  # 1080, 721, 3
h, w, c = img_ori.shape
print(h, w, c)
img = img_ori.reshape((h*w, c))
# plt.imshow(img_ori)
# plt.show()


def main():
    res = []
    for k in [2, 4, 6]:
        k_means = K_Means()
        x_to_y, center_points = k_means.train(img, n_cluster=k, img=True)
        y = np.ones((h*w, c))
        for i in range(y.shape[0]):
            y[i, :] = center_points[x_to_y[i]]

        y = y.reshape((h, w, c)) * 255
        res.append(y)

    plt.subplot(2, 2, 1)
    plt.imshow(img_ori)
    plt.title("origin image")
    plt.subplot(2, 2, 2)
    plt.imshow(res[0])
    plt.title("k = 2")
    plt.subplot(2, 2, 3)
    plt.imshow(res[1])
    plt.title("k = 4")
    plt.subplot(2, 2, 4)
    plt.imshow(res[2])
    plt.title("k = 6")
    plt.show()

if __name__ == "__main__":
    main()
