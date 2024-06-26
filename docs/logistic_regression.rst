.. _logistic_regression:

===============================
Logistic Regression（逻辑回归）
===============================

.. contents:: :local:

Introduction（引言）
====================

Logistic regression is a classification algorithm used to assign observations to a discrete set of classes. Unlike linear regression which outputs continuous number values, logistic regression transforms its output using the logistic sigmoid function to return a probability value which can then be mapped to two or more discrete classes.

逻辑回归是用于将观测样本标识到一组离散类别的分类算法。有别于输出连续数值的线性回归，归逻辑回归使用逻辑曲线函将结果映射输出到两个或多个离散类别的概率值。

Comparison to linear regression（对比线性回归）
---------------------------------------------

Given data on time spent studying and exam scores. :doc:`linear_regression` and logistic regression can predict different things:

给定一组学习时间和考试成绩的数据集， :doc:`线性回归` 和逻辑回归可以预测不同的内容:

  - **Linear Regression** could help us predict the student's test score on a scale of 0 - 100. Linear regression predictions are continuous (numbers in a range).

  - **线性回归** 可以帮助我们预测分数范围0-100的学生测试成绩。线性回归预测结果是连续的（区间数值）。

  - **Logistic Regression** could help use predict whether the student passed or failed. Logistic regression predictions are discrete (only specific values or categories are allowed). We can also view probability scores underlying the model's classifications.

  - **逻辑回归** 可以帮助我们预测学生考试是否通过。逻辑回归预测结果是离散的（仅允许指定数值或类别）。我们也可以查看模型分类类别的概率分数。

Types of logistic regression（逻辑回归的分类）
------------------------------------------

  - Binary (Pass/Fail)
  - 二元逻辑回归（通过/未通过）
  - Multi (Cats, Dogs, Sheep)
  - 多元逻辑回归（猫，狗，羊）
  - Ordinal (Low, Medium, High)
  - 有序逻辑回归（低，中，高）



Binary logistic regression（二元逻辑回归）
========================================

Say we're given `data <http://scilab.io/wp-content/uploads/2016/07/data_classification.csv>`_ on student exam results and our goal is to predict whether a student will pass or fail based on number of hours slept and hours spent studying. We have two features (hours slept, hours studied) and two classes: passed (1) and failed (0).

假设给定学生考试结果的 `数据集 <http://scilab.io/wp-content/uploads/2016/07/data_classification.csv>`_，我们的目标是通过睡眠时间和学习时间来预测他们是否通过考试。我们有两种特征（睡眠时间，学习时间）和两种类别：通过（1）和未通过（0）。


+--------------+-------------+-------------+
| **Studied**  | **Slept**   | **Passed**  |
+--------------+-------------+-------------+
| 4.85         | 9.63        | 1           |
+--------------+-------------+-------------+
| 8.62         | 3.23        | 0           |
+--------------+-------------+-------------+
| 5.43         | 8.23        | 1           |
+--------------+-------------+-------------+
| 9.21         | 6.34        | 0           |
+--------------+-------------+-------------+

Graphically we could represent our data with a scatter plot.

我们可以用散点图的形式呈现我们的数据。

.. image:: images/logistic_regression_exam_scores_scatter.png
    :align: center


Sigmoid activation（Sigmoid激活）
---------------------------------

In order to map predicted values to probabilities, we use the :ref:`sigmoid <activation_sigmoid>` function. The function maps any real value into another value between 0 and 1. In machine learning, we use sigmoid to map predictions to probabilities.

为了完成从预测值到概率的映射，我们使用 :ref:`sigmoid <activation_sigmoid>` 函数。该函数可以将任意真实值映射到0-1的区间。在机器学习中，我们使用sigmoid完成预测值到概率的映射。

.. rubric:: Math

.. math::

  S(z) = \frac{1} {1 + e^{-z}}

.. note::

  - :math:`s(z)` = output between 0 and 1 (probability estimate)
  - :math:`s(z)` = 输出0-1的值（概率估算）
  - :math:`z` = input to the function (your algorithm's prediction e.g. mx + b)
  - :math:`z` = 函数输入（你的函数预测 例：mx + b）
  - :math:`e` = base of natural log
  - :math:`e` = 自然对数的底数

.. rubric:: Graph

.. image:: images/sigmoid.png
    :align: center

.. rubric:: Code

.. literalinclude:: ../code/activation_functions.py
    :language: python
    :pyobject: sigmoid

.. could potentially link to another file.. http://docutils.sourceforge.net/docs/ref/rst/directives.html#include


Decision boundary（决策边界）
----------------------------

Our current prediction function returns a probability score between 0 and 1. In order to map this to a discrete class (true/false, cat/dog), we select a threshold value or tipping point above which we will classify values into class 1 and below which we classify values into class 2.

我们的预测函数返回一个0-1的概率值。为了完成该值到离散类别的映射（真/假，猫/狗），我们选择一个阈值或临界点，高于该值时我们将数值分类为类别1；低于该值时则将数值分类为类别2。

.. math::

  p \geq 0.5, class=1 \\
  p < 0.5, class=0

For example, if our threshold was .5 and our prediction function returned .7, we would classify this observation as positive. If our prediction was .2 we would classify the observation as negative. For logistic regression with multiple classes we could select the class with the highest predicted probability.

例如：如果我们的阈值是0.5并且预测函数返回0.7，我们应该将样本归类到正例。如果我们的预测值是0.2我们应该将样本归类到反例。针对多类别的逻辑回归我们应该选择预测概率最高的类别。

.. image:: images/logistic_regression_sigmoid_w_threshold.png
    :align: center


Making predictions（预测）
-------------------------

Using our knowledge of sigmoid functions and decision boundaries, we can now write a prediction function. A prediction function in logistic regression returns the probability of our observation being positive, True, or "Yes". We call this class 1 and its notation is :math:`P(class=1)`. As the probability gets closer to 1, our model is more confident that the observation is in class 1.

利用sigmoid函数和决策边界，我们可以写出预测函数。逻辑回归的预测函数返回的是样本为正，真或是的概率。我们称为类别1，符号 :math:`P(class=1)`。预测概率越接近1，我们的模型就越认为样本属于类别1.

.. rubric:: Math

Let's use the same :ref:`multiple linear regression <multiple_linear_regression_predict>` equation from our linear regression tutorial.

让我们使用线性回归章节中相同的多元线性回归方程 :ref:`multiple linear regression <multiple_linear_regression_predict>` 

.. math::

  z = W_0 + W_1 Studied + W_2 Slept

This time however we will transform the output using the sigmoid function to return a probability value between 0 and 1.

同时我们将使用sigmoid函数来转换（多元线性回归）输出值，使其返回一个介于0和1之间的概率值。

.. math::

  P(class=1) = \frac{1} {1 + e^{-z}}

If the model returns .4 it believes there is only a 40% chance of passing. If our decision boundary was .5, we would categorize this observation as "Fail.""

如果模型返回0.4表示仅有40%的通过几率。如果我们的决策边界是0.5，我们将样本归类到"Fail.""

.. rubric:: Code

We wrap the sigmoid function over the same prediction function we used in :ref:`multiple linear regression <multiple_linear_regression_predict>`

我们将sigmoid函数包装在多元线性回归章节中相同的预测函数中 :ref:`multiple linear regression <multiple_linear_regression_predict>` 

.. literalinclude:: ../code/logistic_regression.py
    :language: python
    :pyobject: predict


Cost function（成本函数）
------------------------

Unfortunately we can't (or at least shouldn't) use the same cost function :ref:`mse` as we did for linear regression. Why? There is a great math explanation in chapter 3 of Michael Neilson's deep learning book [5]_, but for now I'll simply say it's because our prediction function is non-linear (due to sigmoid transform). Squaring this prediction as we do in MSE results in a non-convex function with many local minimums. If our cost function has many local minimums, gradient descent may not find the optimal global minimum.

不幸的是我们不能（至少不应该）使用同线性回归相同的成本函数 :ref:`mse` 。为何？ Michael Neilson在其深度学习书籍第3章中从数学角度很好的解释了这个问题 [5]_ ，这里我做一些简要说明，因为我们的预测函数是非线性的（受sigmoid变换的影响）。如果我们像在MSE中那样对这种预测结果进行平方处理会得到一个存在许多局部最小值的非凸函数（预测值趋近0或1时导数都趋于0）。如果我们的成本函数包含多个局部最小值，那么梯度下降法可能无法找到最优的全局最小值。

.. rubric:: Math

Instead of Mean Squared Error, we use a cost function called :ref:`loss_cross_entropy`, also known as Log Loss. Cross-entropy loss can be divided into two separate cost functions: one for :math:`y=1` and one for :math:`y=0`.

我们不使用均方误差，而是采用交叉熵损失 :ref:`loss_cross_entropy` 的成本函数，也被称为对数损失。交叉熵损失可以被分解为两个独立的成本函数：一个是针对 :math:`y=1` 的情况，另一个则是针对 :math:`y=0` 的情况。

.. image:: images/ng_cost_function_logistic.png
    :align: center

The benefits of taking the logarithm reveal themselves when you look at the cost function graphs for y=1 and y=0. These smooth monotonic functions [7]_ (always increasing or always decreasing) make it easy to calculate the gradient and minimize cost. Image from Andrew Ng's slides on logistic regression [1]_.

当您查看 y=1 和 y=0 的成本函数图时，取对数的好处就会显现出来。利用平滑单调函数 [7]_ （始终单调递增或单调递减）很容易计算梯度和最小损失。图片来自Andrew Ng's有关逻辑回归的幻灯片 [1]_。

.. image:: images/y1andy2_logistic_function.png
    :align: center

The key thing to note is the cost function penalizes confident and wrong predictions more than it rewards confident and right predictions! The corollary is increasing prediction accuracy (closer to 0 or 1) has diminishing returns on reducing cost due to the logistic nature of our cost function.

关键在于损失函数对模型做出错误预测的惩罚远大于正确预测的奖励！由此推论，由于成本函数的对数性质，提高预测精度（更接近 0 或 1）对降低成本的收益递减。

.. rubric:: Above functions compressed into one（上述函数合二为一）

.. image:: images/logistic_cost_function_joined.png
    :align: center

Multiplying by :math:`y` and :math:`(1-y)` in the above equation is a sneaky trick that let's us use the same equation to solve for both y=1 and y=0 cases. If y=0, the first side cancels out. If y=1, the second side cancels out. In both cases we only perform the operation we need to perform.

上式中乘以 :math:`y` 和 :math:`(1-y)` 是一个小技巧，让我们可以用同一个等式求解 y=1 和 y=0 的情况。当y=0时，左边被抵消。当y=1时，右边被抵消。这两种场景下我们只需执行必要的运算。
。
.. rubric:: Vectorized cost function（损失函数向量）

.. image:: images/logistic_cost_function_vectorized.png
    :align: center

.. rubric:: Code

.. literalinclude:: ../code/logistic_regression.py
    :language: python
    :pyobject: cost_function


Gradient descent（梯度下降）
---------------------------

To minimize our cost, we use :doc:`gradient_descent` just like before in :doc:`linear_regression`. There are other more sophisticated optimization algorithms out there such as conjugate gradient like :ref:`optimizers_lbfgs`, but you don't have to worry about these. Machine learning libraries like Scikit-learn hide their implementations so you can focus on more interesting things!

为了最小化损失，我们使用同线性回归相同 :doc:`linear_regression` 的梯度下降法 :doc:`gradient_descent` 。还有其他更复杂的优化算法，如共轭梯度算法 :ref:`optimizers_lbfgs`，但你无需担心。类似Scikit-learn的机器学习库隐藏了具体的实现逻辑，你可以更专注感兴趣的事情。

.. rubric:: Math

One of the neat properties of the sigmoid function is its derivative is easy to calculate. If you're curious, there is a good walk-through derivation on stack overflow [6]_. Michael Neilson also covers the topic in chapter 3 of his book.

sigmoid函数的一个显著特征是很容易计算他的导数。如果你实在好奇，那么在stack overflow上可以找到很好的推演过程 [6]_。Michael Neilson也在他书的第3章中解释过该问题。

.. math::

  \begin{align}
  s'(z) & = s(z)(1 - s(z))
  \end{align}

Which leads to an equally beautiful and convenient cost function derivative:

由此推导出同样漂亮简洁的代价函数导数（链式法则 J'(θ) = dJ/dp ⋅ dp/dz ⋅ dz/dθ，其中dJ/dp = y/p - (1-y)/(1-p) = (y - p)/p(1 - p)；dp/dz = p(1 - p)；dz/dθ = x；最终-1/m(y - p) ⋅ x）：

.. math::

  C' = x(s(z) - y)

.. note::

  - :math:`C'` is the derivative of cost with respect to weights
  - :math:`C'` 是成本相对权重的导数
  - :math:`y` is the actual class label (0 or 1)
  - :math:`y` 是真实类别标签（0或1）
  - :math:`s(z)` is your model's prediction
  - :math:`s(z)` 是你模型的预测结果
  - :math:`x` is your feature or feature vector.
  - :math:`x` 是你的特征或特征向量

Notice how this gradient is the same as the :ref:`mse` gradient, the only difference is the hypothesis function.

注意这个梯度与 :ref:`mse` 梯度相同，唯一不同的是假设函数。

.. rubric:: Pseudocode

::

  Repeat {

    1. Calculate gradient average
    2. Multiply by learning rate
    3. Subtract from weights

  }

.. rubric:: Code

.. literalinclude:: ../code/logistic_regression.py
    :language: python
    :pyobject: update_weights


Mapping probabilities to classes（概率类别映射）
-----------------------------------------------

The final step is assign class labels (0 or 1) to our predicted probabilities.

最终步骤是将我们的预测概率划分类别标签（0或1）。

.. rubric:: Decision boundary（决策边界）

.. literalinclude:: ../code/logistic_regression.py
    :language: python
    :pyobject: decision_boundary

.. rubric:: Convert probabilities to classes（概率转换为类别）


.. literalinclude:: ../code/logistic_regression.py
    :language: python
    :pyobject: classify

.. rubric:: Example output

::

  Probabilities = [ 0.967, 0.448, 0.015, 0.780, 0.978, 0.004]
  Classifications = [1, 0, 0, 1, 1, 0]


Training（训练）
----------------

Our training code is the same as we used for :ref:`linear regression <simple_linear_regression_training>`.

我们的训练代码同线性回归一样 :ref:`linear regression <simple_linear_regression_training>`。

.. literalinclude:: ../code/logistic_regression.py
    :language: python
    :pyobject: train


Model evaluation（模型评估）
----------------------------

If our model is working, we should see our cost decrease after every iteration.

如果模型开始工作，我们可以看见随着每轮迭代成本减少。

::

  iter: 0 cost: 0.635
  iter: 1000 cost: 0.302
  iter: 2000 cost: 0.264

**Final cost:**  0.2487.  **Final weights:** [-8.197, .921, .738]

.. rubric:: Cost history

.. image:: images/logistic_regression_loss_history.png
    :align: center

.. rubric:: Accuracy（精度）

:ref:`Accuracy <glossary_accuracy>` measures how correct our predictions were. In this case we simply compare predicted labels to true labels and divide by the total.

:ref:`Accuracy <glossary_accuracy>`估量我们预测的准确程度。在该案例中我们简单地将预测标签和真实标签进行比较，然后除以总数。

.. literalinclude:: ../code/logistic_regression.py
    :language: python
    :pyobject: accuracy


.. rubric:: Decision boundary（决策边界）

Another helpful technique is to plot the decision boundary on top of our predictions to see how our labels compare to the actual labels. This involves plotting our predicted probabilities and coloring them with their true labels.

另一个有用的技巧是在我们的预测结果之上绘制决策边界，以此观察比较预测标签和真实标签。这包括绘制我们预测概率，并用真实标签为其着色。

.. image:: images/logistic_regression_final_decision_boundary.png
    :align: center

.. rubric:: Code to plot the decision boundary

.. literalinclude:: ../code/logistic_regression.py
    :language: python
    :pyobject: plot_decision_boundary


Multiclass logistic regression
==============================

Instead of :math:`y = {0,1}` we will expand our definition so that :math:`y = {0,1...n}`. Basically we re-run binary classification multiple times, once for each class.

Procedure
---------

  #. Divide the problem into n+1 binary classification problems (+1 because the index starts at 0?).
  #. For each class...
  #. Predict the probability the observations are in that single class.
  #. prediction = <math>max(probability of the classes)

For each sub-problem, we select one class (YES) and lump all the others into a second class (NO). Then we take the class with the highest predicted value.


Softmax activation
------------------

The softmax function (softargmax or normalized exponential function) is a function that takes as input a vector of K real numbers, and normalizes it into a probability distribution consisting of K probabilities proportional to the exponentials of the input numbers. That is, prior to applying softmax, some vector components could be negative, or greater than one; and might not sum to 1; but after applying softmax, each component will be in the interval [ 0 , 1 ] , and the components will add up to 1, so that they can be interpreted as probabilities.
The standard (unit) softmax function is defined by the formula 

.. math::

  \begin{align}
   σ(z_i) = \frac{e^{z_{(i)}}}{\sum_{j=1}^K e^{z_{(j)}}}\ \ \ for\ i=1,.,.,.,K\ and\ z=z_1,.,.,.,z_K
  \end{align}

In words: we apply the standard exponential function to each element :math:`z_i` of the input vector :math:`z` and normalize these values by dividing by the sum of all these exponentials; this normalization ensures that the sum of the components of the output vector :math:`σ(z)` is 1. [9]_


Scikit-Learn example
-------------

Let's compare our performance to the ``LogisticRegression`` model provided by scikit-learn [8]_.

.. literalinclude:: ../code/logistic_regression_scipy.py


**Scikit score:**  0.88. **Our score:** 0.89


.. rubric:: References

.. [1] http://www.holehouse.org/mlclass/06_Logistic_Regression.html
.. [2] http://machinelearningmastery.com/logistic-regression-tutorial-for-machine-learning
.. [3] https://scilab.io/machine-learning-logistic-regression-tutorial/
.. [4] https://github.com/perborgen/LogisticRegression/blob/master/logistic.py
.. [5] http://neuralnetworksanddeeplearning.com/chap3.html
.. [6] http://math.stackexchange.com/questions/78575/derivative-of-sigmoid-function-sigma-x-frac11e-x
.. [7] https://en.wikipedia.org/wiki/Monotoniconotonic_function
.. [8] http://scikit-learn.org/stable/modules/linear_model.html#logistic-regression>
.. [9] https://en.wikipedia.org/wiki/Softmax_function
