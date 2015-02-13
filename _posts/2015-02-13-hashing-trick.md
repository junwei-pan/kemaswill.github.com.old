---
layout: post
title: "Hashing Trick"
description: "Notes based on PRML and Andrew Ng's Lectures"
category: Machine Learning
tags: ['Machine Learning, SVM']
---
{% include JB/setup %}

本博客已经迁往[http://kemaswill.github.io](http://kemaswill.github.io/), 博客园这边也会继续更新, 欢迎关注~

在机器学习领域, kernel trick是一种非常有效的比较两个样本(对象)的方法. 给定两个对象$$ x_i, x_j \in \mathcal{X} $$, 用$$ k(x_i, x_j) :=\left <\phi(x_i), \phi(x_j)\right> $$来比较两个对象的特征$$ \phi(x_i), \phi(x_j) $$. kernel trick通过定义一个半正定核矩阵$$ k $$, 可以在不得到$$ \phi(x_i) $$的情况下隐式的得到高维向量$$ \phi(x_i) $$和$$ \phi(x_j) $$的内积, 大大减少计算时间. 但是Weinberger[1]等人提出, 在实际中, 尤其是文本分类领域, 原始的输入空间几乎是线性可分的, 但是, 训练集太大, 特征维度太高. 在这种情况下, 没必要把输入向量映射到一个高维的特征空间. 相反的, 有限的内存可能存不下核矩阵. 为此, Langford[2], Qinfeng Shi[3]等人提出了hashing trick, 把高维的输入向量哈希到一个低维的特征空间$$ \mathbb{R}^m $$.

# 1. Hashing Trick

最简单的hashing trick是将原始的每个特征名(或者特征索引)hash到一个低维向量的索引上, 然后将该特征的值累加到该低维向量的索引上:

$$ \bar{\phi}_j(x) = \sum_{i\in \mathcal{J}; h(i) = j}\phi_i(x) $$

其中$$ \phi(x) \in \mathbb{R}^{\mathcal{J}} $$为原始的输入向量, $$ h: \mathcal{J} \to {1,..,n} $$为哈希函数. 算法伪代码为:

{% highlight python linenos %}
function hashing_vectorizer(features : array of string, N : integer):
    x := new vector[N]
    for f in features:
    h := hash(f)     # f 是特征名, 也可以是特征的索引
    x[h mod N] += 1  # 此处累加的是1, 也可以是特征的值
    return x
{% endhighlight %}
# 2. Signed Hash Trick

Weinberger等人提出了一个新的变种, 可以称作signed hash trick.  做法是累加的值不再是固定的1或者特征值, 而是由另外一个哈希函数确定: $$ \xi : \mathbb{N} \to {\pm 1} $$, 这样的好处是可以得到一个无偏的估计.

$$ \bar{\phi}_j(x) = \sum_{i\in \mathcal{J}; h(i) = j}\xi(i)\phi_i(x) $$

算法伪代码为:

{% highlight python linenos %}
function hashing_vectorizer(features : array of string, N : integer):
    x := new vector[N]
    for f in features:
    h := hash(f)
    idx := h mod N
           if ξ(f) == 1:
           x[idx] += 1  # 此处累加的是1, 也可以是特征值
           else:
           x[idx] -= 1  # 此处累加的是-1, 也可以是特征值 * -1
           return x
{% endhighlight %}


# 3. Multiple Hashing

为了防止哈希冲突(亦即不同的特征被哈希到了相同的索引上)带来的负面影响, 可以对那些特征值比较大的特征哈希多次, 如果哈希$$ c $$次, 则每个索引需要累加的值为$$ \frac{1}{\sqrt{c}}\phi_i(x) $$[1].



参考文献:

[1]. Feature Hashing for Large Scale Multitask Learning. K. Weinberger, A. Dasgupta, J. Attenberg, J. Longford, A.Smola. ICML, 2010.

[2]. Vow- pal wabbit online learning project (Technical Report). http://hunch.net/?p=309. Langford, J., Li, L., & Strehl, A. (2007).

[3]. Hash kernels. AISTATS 12. Shi, Q., Petterson, J., Dror, G., Langford, J., Smola, A., Strehl, A., & Vishwanathan, V. (2009).

[4]. Wikipedia: Feature Hashing

[5]. Mahout in Action, page 261. Section 14.3.1 Representing data as a vector: Feature Hashing.
