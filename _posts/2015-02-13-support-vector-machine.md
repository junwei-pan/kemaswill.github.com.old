---
layout: post
title: "Support Vector Machine"
description: ""
category: 
tags: []
---
{% include JB/setup %}

# 最大间隔分类器

假设我们仍然使用线性模型来建模:

\\(y(\mathbf{x})=\mathbf{w}^T\phi(\mathbf{x})+b\\)

假设数据的标签为\\(\{-1, +1\}\\), 我们根据分类器的输出\\(y(\mathbf{x})\\)的符号进行分类. 假设数据是线性可分的, 那么对于标签为1的样本, 我们希望得到\\(y(\mathbf{x})>0\\), 对于标签为-1的样本, 我们希望得到\\(y(\mathbf{x})<1\\). 所以对于所有样本, 我们希望得到\\(t_ny(\mathbf{x}_n)>0\\).

满足以上条件的分类器会有很多(亦即得到的满足条件的参数会有很多), 我们希望得到的分类器能够将正样本和负样本尽可能的分开. 亦即希望最大化所有样本到分类平面的最小距离.

分类超平面定义为\\(y(\mathbf{x})=\mathbf{w}^T\phi(\mathbf{x})+b=0\\), 某点到该平面的距离为\\(\frac{\|y(\mathbf{x})\|}{\|\mathbf{w}\|}\\)

\\(arg\hspace{2 pt}\max_{\mathbf{w}, b}\{\frac{1}{\|\mathbf{w}\|}\min_{n}[t_n(\mathbf{w}^T\phi(\mathbf{x})+b)] \} \tag{1}\\)


\\(\max_i\\)


直接求解以上问题是非常困难的, 我们可以将其转换成一个更容易求解的问题.

如果我们将\\(\mathbf{w}\\)和\\(b\\)进行等比例的伸缩:\\(\mathbf{w}\to\kappa\mathbf{w}, b \to \kappa b\\), 则某点到分类超平面的距离是不变的. 所以我们可以利用这个特性将距离分类超平面最近的点到分类超平面的距离设置为\\(1\\), 那么所有的其他点会满足

\\(t_ny(\mathbf{w})=t_n(\mathbf{w}^T\phi(\mathbf{x})+b)\ge1, n=1,...,N\\)

所以\\((1)\\)式可以转化为以下等价问题

\\(arg\hspace{2 pt}\min_{\mathbf{w}}\frac{1}{2}\|\mathbf{w}\|^2 \tag{2}\\)

\\(s.t.t_ny(\mathbf{w})=t_n(\mathbf{w}^T\phi(\mathbf{x})+b)\ge1, n=1,...,N\\) 

# 对偶问题

可以使用朗格朗日乘子法来求解问题\\((2)\\), 构造如下朗格朗日函数

\\(L(\mathbf{w}, b, \mathbf{a})=\frac{1}{2}\|\mathbf{w}\|^2 - \sum_{n=1}^Na_n\{t_n(\phi(\mathbf{x})+b)-1\}\tag{3}\\)

将上式分别对\\(\mathbf{w}\\)和\\(b\\)求导得:

\\(\mathbf{w}=\sum_{n=1}^Na_nt_n\phi(\mathbf{x}_n)\\)

\\(0=\sum_{n=1}^Na_nt_n\\) 

将上述两式带入\\((3)\\)得到我们需要最大化的对偶问题:

\\(\tilde{L}(\mathbf{a})=\sum_{n=1}^Na_n=\frac{1}{2}\sum_{n=1}^N\sum_{m=1}^Na_na_mt_nt_mk(\mathbf{x}_n, \mathbf{x}_m)\tag{4}\\)

限制条件为:

\\(a_n\ge0, n=1,...,N\\)

\\(\sum_{n=1}^Na_nt_n=0\\)

上述问题可以用常见的二次规划软件来求解, 也可以使用SMO算法来求解.

在对新样本进行预测时, 公式为

\\(y(\mathbf{x})=\sum_{n=1}^Na_nt_nk(\mathbf{x}, \mathbf{x}_n)+b\tag{5}\\)

所有\\(a_n\\)等于0的点都不会出现在\\((5)\\)式中, 而那些\\(a_n\\)不等于\\(0\\)的点成为支持向量, 其距离分类超平面的距离最小(等于1). 

在求解了问题\\((4)\\)之后, 我们可以利用\\((5)\\)式求解\\(b\\). 因为支持向量\\(\mathbf{x}_n\\)距离分类超平面的距离为1, 所以

\\(t_n\left(\sum_{m\in S}a_mt_mk(\mathbf{x}_n, \mathbf{x}_m) +b\right)=1\\)

我们可以基于任意一个支持向量来求得\\(b\\), 也可以基于所有支持向量来求平均:

\\(b=\frac{1}{N_S}\sum_{n\in S}\left(t_n-\sum_{m\in S}a_mt_mk(\mathbf{x}_n,\mathbf{x}_m)\right)\\)

# 非线性可分情况

对于非线性可分情况, 我们需要引入一个松弛变量, 使得所有的样本满足

\\(t_ny(\mathbf{x}_n)\ge 1-\xi_n, n=1,..., N, \xi_n\ge 0\\)

如果\\(\xi_n=0\\), 则该样本被正确分类, 并且位于margin的正确一面.

如果\\(0<\xi_n\le1\\), 则该样本被正确分类, 但是位于margin的里面.

如果\\(\xi_n>1\\), 则该样本被误分类.

引入松弛变量之后, 优化目标变为最小化以下式子

\\(C\sum_{n=1}^N\xi_n + \frac{1}{2}\|\mathbf{w}\|^2\\)

限制条件为

\\(t_ny(\mathbf{x}_n)\ge 1-\xi_n, n=1,..., N \\)

\\( \xi_n\ge 0\\)

其对应的拉格朗日函数为

\\(L(\mathbf{w}, b, \mathbf{a})=\frac{1}{2}\|\mathbf{w}\|^2+C\sum_{n=1}^N\xi_n - \sum_{n=1}^Na_n\{t_ny(\mathbf{x})n)-1 | \xi_n\} -\sum_{n=1}^N\mu_n\xi_n\\)

将上式对于\\(\mathbf{w}, b, \xi_n\\)求导得

\\(\mathbf{w}=\sum_{n=1}^Na_nt_n\phi(\mathbf{x}_n)\\)

\\(\sum_{n=1}^Na_nt_n=0\\)

\\(a_n=C-\mu_n\\)

则最终得到对偶形式为

\\(\tilde{L}(\mathbf{a})=\sum_{n=1}^Na_n-\frac{1}{2}\sum_{n=1}^N\sum_{m=1}^Na_na_mt_nt_mk(\mathbf{x}_n, \mathbf{x}_m)\tag{6}\\)

我们需要在满足以下条件的情况下最小化上式

\\(0\le a_n\le C\tag{7}\\)

\\(\sum_{n=1}^Na_nt_n=0\\)

如果\\(a_n<C\\), 则\\(\mu_n>0, \xi_n=0\\), 则该点是支持向量.

\\(a_n=C\\)则表示该点可能被正确分类但是落在margin之内\\(\xi_n\le1\\), 也会被错误分类\\(\xi_n>1\\)

我们可以同样的根据支持向量(\\(0<a_n<C\\))来计算\\(b\\).

# SMO算法

Platt提出了一种快速的优化\\((6)\\)的算法, SMO.

传统的Corodinate Ascent每次在固定所有其他参数的基础上优化其中一个参数. 但是在优化\\((6)\\)时该方法不凑效, 因为固定所有其他参数, 则剩下的这个参数是固定的:

\\(a_1y_1=-\sum+{i=2}^Na_iy_i\\)

SMO每次优化两个参数, 假设为\\(a_1, a_2\\), 固定所有其他参数, 则

\\(a_1y_1+a_2y_2=\sum_{i=3}a_iy_i=\xi\\)

根据限制条件\\((7)\\), 我们知道\\(a_1, a_2\\)都必须落在\\([0, C] \times [0, C]\\)之内, 而再加上上面的条件, \\(L\le a_2 \le H\\)

\\(a_1\\)可以写作\\(a_2\\)的函数

\\(a_1=(\xi-a_2y_2)y_1\\)

则\\(L(\mathbf{w}, b, (a_1, a_2,...,a_n))\\)可以写成\\(L(\mathbf{w}, b, ((\xi-a_2y_2)y_1, a_2,...,a_n))\\)

易知上式可以写为为\\(a_2\\)的二项式, 很容易求得最优解\\(a_2^{new, unclipped}\\)

在加上之前的限制条件, 可以得到

\\(a_2^{new}=\begin{cases}H & a_2^{new, unclipped} > H \\ a_2^{new, unclipped} & if L \le a_2^{new, unclipped} \le H \\ L & a_2^{new, unclipped} < L\end{cases}\\)
