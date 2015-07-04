---
layout: post
title: "Introduction to Coordinate Descent"
description: ""
category: 
tags: []
---
{% include JB/setup %}

Coordinate Descent
===

常见的优化算法有
: 一阶算法
:  牛顿方法
:  对偶方法
: Interior-point方法

Coordinate Descent(CD)则另辟蹊径, 不再是给定全部或者一些随机样本的情况下更新所有的参数, 而是每次只更新一个参数. 所以问题在于, 这种方法会收敛么? 考虑以下的情况:

1. 给定可微凸函数$f:\mathbb{R}^n \to \mathbb{R}$,如果某点$x$在各个坐标上都是最小的, 那么$x$是否是全局最小点? 答案是肯定的, 因为$$\nabla f(x)=(\frac{\partial f}{\partial x_1}(x),..., \frac{\partial f}\partial x_n(x))=0$$
2. 如果$f$是不可微的凸函数呢? 如下图所示,  两个红色虚线的交点处, 该点无论沿着$x_1$还是$x_2$都是最小的, 但是该点并不是全局最小点.
![enter image description here](https://lh3.googleusercontent.com/-iGRSMhxBU20/VZcpEs6IVgI/AAAAAAAAACI/7p2n3SbC82E/s0/Screen+Shot+2015-07-03+at+%25E4%25B8%258B%25E5%258D%258805.27.06.png "Screen Shot 2015-07-03 at 下午05.27.06.png")
3. 如果$f(x)=g(x)+\sum_{i=1}^nh_i(x_i)$, 其中$g$是可微凸函数, $h_i$是凸函数. 答案也是肯定的. 对于任意$y$:$$f(y)-f(x)\geq \nabla g(x)^T(y-x)+\sum_{i=1}^n[h_i(y_i)-h_i(x_i)]=\sum_{i=1}^n[\nabla_i g(x)(y_i-x_i)+h_i(x_i)] \geq 0$$

Coordinate Descent
---
Coordinate Descent算法本身很简单, 从初始值$x^{(0)}$开始, 对于$k=1,2,3,...$
$$x_1^{(k)}=argmin_{x_1}f(x_1, x_2^{(k-1)}, x_3^{(k-1)},...,x_n^{(k-1)})$$
$$x_2^{(k)}=argmin_{x_1}f(x_1^{k}, x_2, x_3^{(k-1)},...,x_n^{(k-1)})$$
$$x_3^{(k)}=argmin_{x_1}f(x_1^{k}, x_2^{(k)}, x_3,...,x_n^{(k-1)})$$
$$...$$
$$x_n^{(k)}=argmin_{x_1}f(x_1^{k}, x_2^{(k)}, x_3^{(k)},...,x_n)$$

需要注意以下几点
1. 每次循环时, 左边的顺序是任意的, 可以使用${1,2,...,n}$的任意排列.
2. 每次只能更新一个坐标, 如果每次更新所有的, 不一定会收敛[^footnote].
[^footnote]: 但是后来Shai Shalev-Shwartz等人提出了stochastic Coordinate descent, Eric P. Xing等人提出了Parallel Coordinate Descent等变种. 详见下文

Coordinate Descent for Linear Regression

考虑 $f(x)=\frac{1}{2}\|y-Ax\|^2$, 其中$y\in\mathbb{R}^n, A\in \mathbb{R}^{n\times p}$, 其中$A_1, ..., A_p$是$A$的列. 固定住$x_i, j\neq i$, 对于$x_i$最小化$f(x)$:$$0=\nabla_i f(x)=A^T_i(Ax-y)=A^T_i(A_ix_i+A_{-i}x_{-i}-y)$$得到$$x_i=\frac{A_i^T(y-A_{-i}x_{-i})}{A_i^TA_i}$$
Coordinate Descent按照$i=1,2,...,p,1,2,...
的顺序重复以上步骤.
Coordinate Descent for Lasso

$$f(x)=\frac{1}{2}\|y-Ax\|^2+\lambda \|x\|_i$$
固定住$x_i, j\neq i$, 对于$x_i$最小化上式:$$0=A_i^TA_ix_i+A_i^T(A_{-i}x_{-i}-y) + \lambda s_i$$
其中$s_i=\partial|x_i|$
最后$$x_i=S_{\lambda / \|A_i\|^2}(\frac{A_i^T(y-A_{-ix_{-i}})}{A_i^TA_i})$$
