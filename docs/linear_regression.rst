.. _linear_regression:

============================
Linear Regression（线性回归）
============================

.. contents::
    :local:
    :depth: 2


Introduction（引言）
============================

Linear Regression is a supervised machine learning algorithm where the predicted output is continuous and has a constant slope. It's used to predict values within a continuous range, (e.g. sales, price) rather than trying to classify them into categories (e.g. cat, dog). There are two main types:

线性回归属于预测结果连续且具有固定斜率的监督学习算法。线性回归用来在连续区间中预测数据，（例：销售，价格）而不是试图将他们分属到不同的类别（例：猫，狗）。他们有以下两种表现形式：


.. rubric:: Simple regression（简单回归）

Simple linear regression uses traditional slope-intercept form, where :math:`m` and :math:`b` are the variables our algorithm will try to "learn" to produce the most accurate predictions. :math:`x` represents our input data and :math:`y` represents our prediction.

简单线性回归通常使用斜截式的表现形式，其中 :math:`m` 和 :math:`b` 属于变量，我们的算法将尝试“训练”这些变量以产生最精准预测。:math:`x` 表示输入数据，:math:`y` 表示输出数据。

.. math::

  y = mx + b

.. rubric:: Multivariable regression（多变量性回归）

A more complex, multi-variable linear equation might look like this, where :math:`w` represents the coefficients, or weights, our model will try to learn.

一个更复杂的多变量线性方程可能如下所示，其中 :math:`w` 表示系数（权重），我们的模型将尝试“训练”这些变量。

.. math::

  f(x,y,z) = w_1 x + w_2 y + w_3 z

The variables :math:`x, y, z` represent the attributes, or distinct pieces of information, we have about each observation. For sales predictions, these attributes might include a company's advertising spend on radio, TV, and newspapers.

这些变量 :math:`x, y, z` 表示观测样本的属性（不同的特征信息）。对于销量预测，这些属性可能包含一个公司分别在电台，TV和报纸的广告投入。

.. math::

  Sales = w_1 Radio + w_2 TV + w_3 News


Simple regression（简单回归）
============================

Let’s say we are given a `dataset <http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv>`_ with the following columns (features): how much a company spends on Radio advertising each year and its annual Sales in terms of units sold. We are trying to develop an equation that will let us to predict units sold based on how much a company spends on radio advertising. The rows (observations) represent companies.

假设我们给定具有以下表头（特征）的 `数据集 <http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv>`_：一家公司每年在电台广告上的支出以及以销量为单位的年销售额。我们试图建一个方程式来根据公司的电台广告花费（Radio）预测销量（Sales）。

+--------------+---------------+-----------+
| **Company**  | **Radio ($)** | **Sales** |
+--------------+---------------+-----------+
| Amazon       | 37.8          | 22.1      |
+--------------+---------------+-----------+
| Google       | 39.3          | 10.4      |
+--------------+---------------+-----------+
| Facebook     | 45.9          | 18.3      |
+--------------+---------------+-----------+
| Apple        | 41.3          | 18.5      |
+--------------+---------------+-----------+


Making predictions（做出预测）
-----------------------------

Our prediction function outputs an estimate of sales given a company's radio advertising spend and our current values for *Weight* and *Bias*.

我们的预测函数根据当前的 *权重（斜率）* 和 *偏置（截距）* 值以及给定的电台广告支出输出一个销售额的估算结果

.. math::

  Sales = Weight \cdot Radio + Bias

Weight（权重）
  the coefficient for the Radio independent variable. In machine learning we call coefficients *weights*.

  作为自变量Radio的系数。在机器学习中我们称之为 *权重* 。

Radio（电台广告支出）
  the independent variable. In machine learning we call these variables *features*.

  自变量。在机器学习中我们称这些变量为 *特征* 。

Bias（偏置）
  the intercept where our line intercepts the y-axis. In machine learning we can call intercepts *bias*. Bias offsets all predictions that we make.

  直线与y轴相交处的截距。在机器学习中我们将截距称为 *偏置* 。偏置会对我们所做的所有预测产生一个偏差修正。

Our algorithm will try to *learn* the correct values for Weight and Bias. By the end of our training, our equation will approximate the *line of best fit*.

我们的算法将尝试训练权重和偏置的修正值。训练结束，我们的方程将贴近 *最佳拟合线* 。

.. image:: images/linear_regression_line_intro.png
    :align: center

.. rubric:: Code

::

  def predict_sales(radio, weight, bias):
      return weight*radio + bias


Cost function（成本函数）
------------------------

The prediction function is nice, but for our purposes we don't really need it. What we need is a :doc:`cost function <loss_functions>` so we can start optimizing our weights.

预测函数很完美，但就我们的目的而言，（现阶段）并非真正必需。我们需要的是一个 :doc:`成本函数 <损失函数>`，这样我们才能开始优化我们的权重。

Let's use :ref:`mse` as our cost function. MSE measures the average squared difference between an observation's actual and predicted values. The output is a single number representing the cost, or score, associated with our current set of weights. Our goal is to minimize MSE to improve the accuracy of our model.

我们使用 :ref:`mse` （均方误差）作为我们的成本函数。MSE（均方误差）衡量的是每个样本实际值与其预测值之间平方差的平均值。其输出是一个单一数字，代表与当前（预测函数）权值集合相关的成本或评分。我们的目标是尽可能减小MSE，以提高模型的准确性。

.. rubric:: Math

Given our simple linear equation :math:`y = mx + b`, we can calculate MSE as:

给定简单的线性方程 :math:`y = mx + b`, 可以使用以下方式计算MSE：

.. math::

  MSE =  \frac{1}{N} \sum_{i=1}^{n} (y_i - (m x_i + b))^2

.. note::

  - :math:`N` is the total number of observations (data points)
  - :math:`N` 表示样本数量（数据点集合）
  - :math:`\frac{1}{N} \sum_{i=1}^{n}` is the mean
  - :math:`\frac{1}{N} \sum_{i=1}^{n}` 表示均值
  - :math:`y_i` is the actual value of an observation and :math:`m x_i + b` is our prediction
  - :math:`y_i` 表示样本的真实值， :math:`m x_i + b` 表示我们的预测值

.. rubric:: Code

::

  def cost_function(radio, sales, weight, bias):
      companies = len(radio)
      total_error = 0.0
      for i in range(companies):
          total_error += (sales[i] - (weight*radio[i] + bias))**2
      return total_error / companies


Gradient descent（梯度下降）
---------------------------

To minimize MSE we use :doc:`gradient_descent` to calculate the gradient of our cost function. Gradient descent consists of looking at the error that our weight currently gives us, using the derivative of the cost function to find the gradient (The slope of the cost function using our current weight), and then changing our weight to move in the direction opposite of the gradient. We need to move in the opposite direction of the gradient since the gradient points up the slope instead of down it, so we move in the opposite direction to try to decrease our error. 

为了最小化MSE我们使用 :doc:`gradient_descent` 来计算损失函数的梯度。梯度下降的过程包括：观察当前权重所带来的误差，利用成本函数的导数找到梯度（即采用当前权重时成本函数的斜率），然后调整权重使其朝着梯度相反的方向移动。

.. rubric:: Math

There are two :ref:`parameters <glossary_parameters>` (coefficients) in our cost function we can control: weight :math:`m` and bias :math:`b`. Since we need to consider the impact each one has on the final prediction, we use partial derivatives. To find the partial derivatives, we use the :ref:`chain_rule`. We need the chain rule because :math:`(y - (mx + b))^2` is really 2 nested functions: the inner function :math:`y - (mx + b)` and the outer function :math:`x^2`.

在我们的成本函数中有两个可控参数（系数）：权重 :math:`m` 和偏置 :math:`b`。由于我们需要考虑这两个参数各自对最终预测结果产生的影响，所以我们采用偏导数来进行分析。

Returning to our cost function:

回到我们的成本函数：

.. math::

    f(m,b) =  \frac{1}{N} \sum_{i=1}^{n} (y_i - (mx_i + b))^2

Using the following:

使用如下等价公式（复合函数表示形式）：

.. math::

    (y_i - (mx_i + b))^2 = A(B(m,b))

We can split the derivative into

我们拆解复合函数导数为

.. math::

    A(x) = x^2

    \frac{df}{dx} = A'(x) = 2x

and

以及

.. math::

    B(m,b) = y_i - (mx_i + b) = y_i - mx_i - b

    \frac{dx}{dm} = B'(m) = 0 - x_i - 0 = -x_i

    \frac{dx}{db} = B'(b) = 0 - 0 - 1 = -1

And then using the :ref:`chain_rule` which states:

然后使用链式法则申明如下：

.. math::

    \frac{df}{dm} = \frac{df}{dx} \frac{dx}{dm}

    \frac{df}{db} = \frac{df}{dx} \frac{dx}{db}

We then plug in each of the parts to get the following derivatives

将前面求得的基本函数导数代入其中得到（成本函数关于权重 :math:`m` 的）导数：

.. math::

    \frac{df}{dm} = A'(B(m,f)) B'(m) = 2(y_i - (mx_i + b)) \cdot -x_i

    \frac{df}{db} = A'(B(m,f)) B'(b) = 2(y_i - (mx_i + b)) \cdot -1

We can calculate the gradient of this cost function as:

我们可以通过如下形式计算成本函数的梯度：

.. math::
  \begin{align}
  f'(m,b) =
    \begin{bmatrix}
      \frac{df}{dm}\\
      \frac{df}{db}\\
    \end{bmatrix}
  &=
    \begin{bmatrix}
      \frac{1}{N} \sum -x_i \cdot 2(y_i - (mx_i + b)) \\
      \frac{1}{N} \sum -1 \cdot 2(y_i - (mx_i + b)) \\
    \end{bmatrix}\\
  &=
    \begin{bmatrix}
       \frac{1}{N} \sum -2x_i(y_i - (mx_i + b)) \\
       \frac{1}{N} \sum -2(y_i - (mx_i + b)) \\
    \end{bmatrix}
  \end{align}

.. rubric:: Code

To solve for the gradient, we iterate through our data points using our new weight and bias values and take the average of the partial derivatives. The resulting gradient tells us the slope of our cost function at our current position (i.e. weight and bias) and the direction we should update to reduce our cost function (we move in the direction opposite the gradient). The size of our update is controlled by the :ref:`learning rate <glossary_learning_rate>`.

为了求解梯度，我们不断使用新的权重和偏差值遍历所有数据点（样本数据），并取偏导数的平均值。通过此时梯度结果可知成本函数在当前位置的斜率（权重和偏置）以及应该更新以减少成本函数的方向（我们朝梯度反方向移动）。（权重和偏置）更新的步进值由 :ref:`learning rate` （学习率）控制。

::

  def update_weights(radio, sales, weight, bias, learning_rate):
      weight_deriv = 0
      bias_deriv = 0
      companies = len(radio)

      for i in range(companies):
          # Calculate partial derivatives
          # -2x(y - (mx + b))
          weight_deriv += -2*radio[i] * (sales[i] - (weight*radio[i] + bias))

          # -2(y - (mx + b))
          bias_deriv += -2*(sales[i] - (weight*radio[i] + bias))

      # We subtract because the derivatives point in direction of steepest ascent
      weight -= (weight_deriv / companies) * learning_rate
      bias -= (bias_deriv / companies) * learning_rate

      return weight, bias


.. _simple_linear_regression_training:

Training（训练）
---------------

Training a model is the process of iteratively improving your prediction equation by looping through the dataset multiple times, each time updating the weight and bias values in the direction indicated by the slope of the cost function (gradient). Training is complete when we reach an acceptable error threshold, or when subsequent training iterations fail to reduce our cost.

训练模型的过程是指通过多次遍历整个数据集的方式，循环改进预测函数。每次循环中，都会根据成本函数（梯度）的斜率指示的方向更新权重和偏置值。训练完成的标志是我们达到可接受的误差阈值，或者后续训练迭代无法进一步降低我们的成本为止。

Before training we need to initialize our weights (set default values), set our :ref:`hyperparameters <glossary_hyperparameters>` (learning rate and number of iterations), and prepare to log our progress over each iteration.

开始训练之前我们需要初始化权重（设置默认值），设置:ref:`hyperparameters` 超参数（学习率和迭代次数），以及记录每次迭代的调整信息。

.. rubric:: Code

::

  def train(radio, sales, weight, bias, learning_rate, iters):
      cost_history = []

      for i in range(iters):
          weight,bias = update_weights(radio, sales, weight, bias, learning_rate)

          #Calculate cost for auditing purposes
          cost = cost_function(radio, sales, weight, bias)
          cost_history.append(cost)

          # Log Progress
          if i % 10 == 0:
              print "iter={:d}    weight={:.2f}    bias={:.4f}    cost={:.2}".format(i, weight, bias, cost)

      return weight, bias, cost_history


Model evaluation（模型评估）
---------------------------

If our model is working, we should see our cost decrease after every iteration.

模型开始训练时，我们需要观察每次迭代后成本减小情况。

.. rubric:: Logging

::

  iter=1     weight=.03    bias=.0014    cost=197.25
  iter=10    weight=.28    bias=.0116    cost=74.65
  iter=20    weight=.39    bias=.0177    cost=49.48
  iter=30    weight=.44    bias=.0219    cost=44.31
  iter=30    weight=.46    bias=.0249    cost=43.28

.. rubric:: Visualizing

.. image:: images/linear_regression_line_1.png
    :align: center

.. image:: images/linear_regression_line_2.png
    :align: center

.. image:: images/linear_regression_line_3.png
    :align: center

.. image:: images/linear_regression_line_4.png
    :align: center


.. rubric:: Cost history

.. image:: images/linear_regression_training_cost.png
    :align: center


Summary（总结）
--------------

By learning the best values for weight (.46) and bias (.25), we now have an equation that predicts future sales based on radio advertising investment.

通过学习到的权重（.46）和偏置（.25），我们拥有了一个可以通过电台广告投入来预测未来销售数的函数。

.. math::

  Sales = .46 Radio + .025

How would our model perform in the real world? I’ll let you think about it :)

我们的模型在现实世界中会有怎样的表现呢？这个问题我留给你思考一下 :)



Multivariable regression（多元回归）
===================================

Let’s say we are given `data <http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv>`_ on TV, radio, and newspaper advertising spend for a list of companies, and our goal is to predict sales in terms of units sold.

假定我们获得以下数据集 `data <http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv>`_ ，数据记录了一系列公司在TV（电视），radio（电台），newpaper（报纸）上的广告开支，我们的最终目标是预测以销售的Units为单位的销量。

+----------+-------+-------+------+-------+
| Company  | TV    | Radio | News | Units |
+----------+-------+-------+------+-------+
| Amazon   | 230.1 | 37.8  | 69.1 | 22.1  |
+----------+-------+-------+------+-------+
| Google   | 44.5  | 39.3  | 23.1 | 10.4  |
+----------+-------+-------+------+-------+
| Facebook | 17.2  | 45.9  | 34.7 | 18.3  |
+----------+-------+-------+------+-------+
| Apple    | 151.5 | 41.3  | 13.2 | 18.5  |
+----------+-------+-------+------+-------+


Growing complexity（复杂度增加）
-------------------------------
As the number of features grows, the complexity of our model increases and it becomes increasingly difficult to visualize, or even comprehend, our data.

随着特征数量的增长，模型的复杂性也随之增加，这就使得数据的可视化乃至理解变得越来越困难。

.. image:: images/linear_regression_3d_plane_mlr.png
    :align: center

One solution is to break the data apart and compare 1-2 features at a time. In this example we explore how Radio and TV investment impacts Sales.

一种解决方案是将数据拆分开来，一次对比1-2个特征。在这个例子中，我们将探讨电台和电视投资如何影响销量。

Normalization（归一化）
----------------------

As the number of features grows, calculating gradient takes longer to compute. We can speed this up by "normalizing" our input data to ensure all values are within the same range. This is especially important for datasets with high standard deviations or differences in the ranges of the attributes. Our goal now will be to normalize our features so they are all in the range -1 to 1.

随着特征数量的增多，计算梯度所需的时间也会变长。为了加快这一过程，我们可以通过对输入数据进行“归一化”来确保所有数值都在同一范围内。这对于具有高标准差或属性范围差异大的数据集尤为重要。我们现在的目标是将特征归一化，使其都落在-1到1的范围内。

.. rubric:: Code

::

  For each feature column {
      #1 Subtract the mean of the column (mean normalization)
      #1 减去列的均值（均值归一化）
      #2 Divide by the range of the column (feature scaling)
      #2 除以列的区间值（特征缩放）
  }

Our input is a 200 x 3 matrix containing TV, Radio, and Newspaper data. Our output is a normalized matrix of the same shape with all values between -1 and 1.

我们的输入是一个形状 200 * 3 的矩阵，包括电视，电台和报纸数据。我们的输出是一个形状相同且所有数据范围在[-1,1]的归一化矩阵。

::

  def normalize(features):
      **
      features     -   (200, 3)
      features.T   -   (3, 200)

      We transpose the input matrix, swapping
      cols and rows to make vector math easier
      我们对输入矩阵进行转置，交换
      列和行以简化向量运算
      **

      for feature in features.T:
          fmean = np.mean(feature)
          frange = np.amax(feature) - np.amin(feature)

          #Vector Subtraction
          feature -= fmean

          #Vector Division
          feature /= frange

      return features

.. note::

  **Matrix math**. Before we continue, it's important to understand basic :doc:`linear_algebra` concepts as well as numpy functions like `numpy.dot() <https://docs.scipy.org/doc/numpy/reference/generated/numpy.dot.html>`_.

  **矩阵数学**。了解 :doc:`线性代数` 的概念以及 numpy 库的函数例如`numpy.dot() <https://docs.scipy.org/doc/numpy/reference/generated/numpy.dot.html>`_ 对于后续学习课程非常重要。

.. _multiple_linear_regression_predict:

Making predictions（做出预测）
-----------------------------

Our predict function outputs an estimate of sales given our current weights (coefficients) and a company's TV, radio, and newspaper spend. Our model will try to identify weight values that most reduce our cost function.

我们的预测函数会根据权重（系数）和一个公司的电视，电台以及报纸投入输出一个销量的估算结果。我们的模型将会试图找到可以减少代价函数的权重值。

.. math::

  Sales = W_1 TV + W_2 Radio + W_3 Newspaper

::

  def predict(features, weights):
    **
    features - (200, 3)
    weights - (3, 1)
    predictions - (200,1)
    **
    predictions = np.dot(features, weights)
    return predictions


Initialize weights（初始化）
---------------------------

::

  W1 = 0.0
  W2 = 0.0
  W3 = 0.0
  weights = np.array([
      [W1],
      [W2],
      [W3]
  ])


Cost function（代价函数）
------------------------
Now we need a cost function to audit how our model is performing. The math is the same, except we swap the :math:`mx + b` expression for :math:`W_1 x_1 + W_2 x_2 + W_3 x_3`. We also divide the expression by 2 to make derivative calculations simpler.

现在我们需要代价函数来评估我们模型的表现。数学表达式相近，只是将 :math:`mx + b` 替换成 :math:`W_1 x_1 + W_2 x_2 + W_3 x_3`。我们还需要将表达式除以2来简化求导过程。

.. math::

  MSE =  \frac{1}{2N} \sum_{i=1}^{n} (y_i - (W_1 x_1 + W_2 x_2 + W_3 x_3))^2

::

  def cost_function(features, targets, weights):
      **
      features:(200,3)
      targets: (200,1)
      weights:(3,1)
      returns average squared error among predictions
      **
      N = len(targets)

      predictions = predict(features, weights)

      # Matrix math lets use do this without looping
      # 矩阵数学可以不使用循环完成平方差计算
      sq_error = (predictions - targets)**2

      # Return average squared error among predictions
      # 返回同预测结果间的均方误差
      return 1.0/(2*N) * sq_error.sum()


Gradient descent（梯度下降）
---------------------------

Again using the :ref:`chain_rule` we can compute the gradient--a vector of partial derivatives describing the slope of the cost function for each weight.

我们再次使用 :ref:`链式规则` 计算梯度--一个描述代价函数对于每个权重斜率的偏导数向量。

.. math::

  \begin{align}
  f'(W_1) = -x_1(y - (W_1 x_1 + W_2 x_2 + W_3 x_3)) \\
  f'(W_2) = -x_2(y - (W_1 x_1 + W_2 x_2 + W_3 x_3)) \\
  f'(W_3) = -x_3(y - (W_1 x_1 + W_2 x_2 + W_3 x_3))
  \end{align}

::

  def update_weights(features, targets, weights, lr):
      '''
      Features:(200, 3)
      Targets: (200, 1)
      Weights:(3, 1)
      '''
      predictions = predict(features, weights)

      #Extract our features
      #提取特征数据
      x1 = features[:,0]
      x2 = features[:,1]
      x3 = features[:,2]

      # Use dot product to calculate the derivative for each weight
      # 使用点积计算每个权重的导数
      d_w1 = -x1.dot(targets - predictions)
      d_w2 = -x2.dot(targets - predictions)
      d_w2 = -x2.dot(targets - predictions)

      # Multiply the mean derivative by the learning rate
      # 用学习率乘以导数的均值
      # and subtract from our weights (remember gradient points in direction of steepest ASCENT)
      # 并且用当前权重减去它（记住最陡峭上升方向的梯度点）
      weights[0][0] -= (lr * np.mean(d_w1))
      weights[1][0] -= (lr * np.mean(d_w2))
      weights[2][0] -= (lr * np.mean(d_w3))

      return weights

And that's it! Multivariate linear regression.

这就是多元线性回归！



Simplifying with matrices（简化矩阵）
------------------------------------

The gradient descent code above has a lot of duplication. Can we improve it somehow? One way to refactor would be to loop through our features and weights--allowing our function to handle any number of features. However there is another even better technique: *vectorized gradient descent*.

上面梯度下降代码中有许多重复计算。那么我们可以怎样优化呢？

.. rubric:: Math

We use the same formula as above, but instead of operating on a single feature at a time, we use matrix multiplication to operative on all features and weights simultaneously. We replace the :math:`x_i` terms with a single feature matrix :math:`X`.

我们采用与上述相同的公式，但是不再逐个操作单个特征，而是通过矩阵乘法同时对所有特征和权重进行运算。我们将单个特征项x_i替换为一个特征矩阵X

.. math::

  gradient = -X(targets - predictions)

.. rubric:: Code

::

  X = [
      [x1, x2, x3]
      [x1, x2, x3]
      .
      .
      .
      [x1, x2, x3]
  ]

  targets = [
      [1],
      [2],
      [3]
  ]

  def update_weights_vectorized(X, targets, weights, lr):
      **
      gradient = X.T * (predictions - targets) / N
      X: (200, 3)
      Targets: (200, 1)
      Weights: (3, 1)
      **
      companies = len(X)

      #1 - Get Predictions
      predictions = predict(X, weights)

      #2 - Calculate error/loss
      error = targets - predictions

      #3 Transpose features from (200, 3) to (3, 200)
      # So we can multiply w the (200,1)  error matrix.
      # 因此我们可以计算权重矩阵和(200,1)误差矩阵的点积
      # Returns a (3,1) matrix holding 3 partial derivatives --
      # 返回一个包含3个偏导的(3,1)矩阵：(3,200)@(200,1)->(3,1)
      # one for each feature -- representing the aggregate
      # 每个偏导数对应一个特征 -- 代表所有样本数据在代价函数上的整体斜率
      # slope of the cost function across all observations
      gradient = np.dot(-X.T,  error)

      #4 Take the average error derivative for each feature
      gradient /= companies

      #5 - Multiply the gradient by our learning rate
      gradient *= lr

      #6 - Subtract from our weights to minimize cost
      weights -= gradient

      return weights


Bias term（偏置项）
------------------

Our train function is the same as for simple linear regression, however we're going to make one final tweak before running: add a :ref:`bias term <glossary_bias_term>` to our feature matrix.

我们的训练函数同简单线性回归相同，但在训练之前我们还需要做一些细微调整：添加 :ref:`偏置项` 到我们的特征矩阵。

In our example, it's very unlikely that sales would be zero if companies stopped advertising. Possible reasons for this might include past advertising, existing customer relationships, retail locations, and salespeople. A bias term will help us capture this base case.

在我们的案列中，即使公司停止广告投入对应的销量也不太可能为零。可能原因包括过去的广告活动、已有的客户关系、
零售地点以及销售人员等。偏置项可以帮助我们覆盖这些基础场景。

.. rubric:: Code

Below we add a constant 1 to our features matrix. By setting this value to 1, it turns our bias term into a constant.

下面我们添加常量1到我们的特征矩阵。通过将这个值设置为1，我们的偏置项变成了一个常数

::

  bias = np.ones(shape=(len(features),1))
  features = np.append(bias, features, axis=1)


Model evaluation（模型评估）
---------------------------

After training our model through 1000 iterations with a learning rate of .0005, we finally arrive at a set of weights we can use to make predictions:

通过学习率.0005和1000次迭代训练模型，最终我们得到了可以用来预测的权重集合：

.. math::

  Sales = 4.7TV + 3.5Radio + .81Newspaper + 13.9

Our MSE cost dropped from 110.86 to 6.25.

.. image:: images/multiple_regression_error_history.png
    :align: center


.. rubric:: References

.. [1] https://en.wikipedia.org/wiki/Linear_regression
.. [2] http://www.holehouse.org/mlclass/04_Linear_Regression_with_multiple_variables.html
.. [3] http://machinelearningmastery.com/simple-linear-regression-tutorial-for-machine-learning
.. [4] http://people.duke.edu/~rnau/regintro.htm
.. [5] https://spin.atomicobject.com/2014/06/24/gradient-descent-linear-regression
.. [6] https://www.analyticsvidhya.com/blog/2015/08/common-machine-learning-algorithms
