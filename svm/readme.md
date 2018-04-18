This part is the implementation of svm(support vector machine). I think it is a somewhat interpretable classification algorithm.

I use the [digit data](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html) from skleran module. After being reduced to 2 dimension by [t-sne](https://github.com/liziniu/machine_learning_2018_spring/blob/master/k-means/t-sne-tutorial.ipynb), the digit of 1 and 2 are visualized in the following.

<div align=center>
  <img src="https://github.com/liziniu/machine_learning_2018_spring/blob/master/svm/reduced_data.png" width="350" height="350">
</div>
Without kernel(i.e. it is a linear algorithm), the classification boundary are plotted in the following.

<div align=center>
  <img src="https://github.com/liziniu/machine_learning_2018_spring/blob/master/svm/svm_results.png" width="350" height="350">
</div>

With rbf kernel, the result is as follows:

<div align=center>
  <img src="https://github.com/liziniu/machine_learning_2018_spring/blob/master/svm/kernel_svm.png" width="350" height="350">
</div>
