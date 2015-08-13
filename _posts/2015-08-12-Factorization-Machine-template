<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>2015-08-10-from-matrix-factotization-to-factorization-machine-to-field-aware-factorization-machine</title>
<link rel="stylesheet" href="https://stackedit.io/res-min/themes/base.css" />
<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS_HTML"></script>
</head>
<body><div class="container"><p>{% include JB/setup %}</p>



<h2 id="matrix-factorization">Matrix Factorization</h2>

<p>In Recommender System, <strong>Matrix Factorization</strong> maps both the users and items to a joint latent factor space of dimension <script type="math/tex" id="MathJax-Element-26312">k</script>, such that the user-item interaction can be modeled as the inner product in this space. That is to say, we map each user <script type="math/tex" id="MathJax-Element-26313">i</script> to a vector <script type="math/tex" id="MathJax-Element-26314">p_i\in \mathbb{R}^k</script>, and each item <script type="math/tex" id="MathJax-Element-26315">j</script> to a vector <script type="math/tex" id="MathJax-Element-26316">q_j \in \mathbb{R}^k</script>. For movie recommendation, each dimension of the latent factor space can be explained as a topic, say comedy v.s. drama, or other features such as amount of action, orientation to children and so on. <br>
Given the latent vector for user <script type="math/tex" id="MathJax-Element-26317">u</script> and item <script type="math/tex" id="MathJax-Element-26318">i</script>, we can predict the interaction between them as <script type="math/tex; mode=display" id="MathJax-Element-26319">\hat{r_{ui}}=q_i^Tp_u\tag{1}</script>The major challenge is to compute the latent vector for each user and item. This is quite similar with <strong>Singular Value Decomposition(SVD)</strong>, which …. However, the matrix <script type="math/tex" id="MathJax-Element-26320">M</script> is needed to be complete when using SVD to decompose it. One method is to rely on imputation to fill in missing ratings to make the matrix dense. However, this will significantly increase the amount of data, and the inaccurate imputation might distort the data. <br>
Matrix Factorization is a method which focus only on the observed ratings only, while avoid overfitting by regularization. Here is the cost function for matrix factorization<script type="math/tex; mode=display" id="MathJax-Element-26321">\min_{q^*, p^*}\sum_{(u, i) \in \kappa}(r_ui - q_i^Tp_u)^2 + \lambda (||q_i||^2 + ||p_u||^2)\tag{2}</script>Here <script type="math/tex" id="MathJax-Element-26322">\kappa</script> is the set of <script type="math/tex" id="MathJax-Element-26323">(u, i)</script>pairs for which <script type="math/tex" id="MathJax-Element-26324">r_{ui}</script> is known in the training set.</p>

<h3 id="optimization-by-sgd">Optimization by SGD</h3>

<p>The above cost function <script type="math/tex" id="MathJax-Element-26340">(2)</script> works as following</p>

<blockquote>
  <p>Stochastic Gradient Descent for Matrix Factorization</p>
  
  <blockquote>
    <p>Util termination(Iterate the training data N times, or when the cost function converges)</p>
    
    <blockquote>
      <p>For each rating <script type="math/tex" id="MathJax-Element-26341">r_{ui}</script> in the training set</p>
      
      <blockquote>
        <p><script type="math/tex" id="MathJax-Element-26342">e_{ui} \stackrel{\text{def}}{=} r_{ui} - q_i^Tp_u</script></p>
        
        <p><script type="math/tex" id="MathJax-Element-26343">q_i \gets q_i + \gamma(e_{ui} p_u -\lambda q_i)</script></p>
        
        <p><script type="math/tex" id="MathJax-Element-26344">p_u \gets p_u + \gamma (e_{ui}q_i -\lambda p_u)</script></p>
      </blockquote>
    </blockquote>
  </blockquote>
</blockquote>

<h3 id="adding-biases">Adding Biases</h3>

<p>The <script type="math/tex" id="MathJax-Element-26218">(2)</script> only interpret the rating <script type="math/tex" id="MathJax-Element-26219">r_{ui}</script> as an interaction between the user <script type="math/tex" id="MathJax-Element-26220">u</script> and item <script type="math/tex" id="MathJax-Element-26221">i</script>, but in the fact,  the rating values can also due to effects associated with either users or items. For example, in the recommender system, some user tend to give higher rating than others, or some items is in general better than others. To consider all such effects, we can add a bias term to the <script type="math/tex" id="MathJax-Element-26222">(1)</script> <br>
<script type="math/tex; mode=display" id="MathJax-Element-26223">\hat{r_{ui}} = \mu + b_i + b_u + q_i^Tp_u\tag{3}</script> <br>
 and the corresponding cost function is <script type="math/tex; mode=display" id="MathJax-Element-26224">\min_{p^*,q^*, b^*}\sum_{(u, i)\in \kappa}(r_{ui}-\mu-b_u-b_i-p_u^Tq_i)^2 + \lambda(||p_u||^2 + ||q_i||^2 + b_u^2 + b_i^2)\tag{4}</script></p>



<h2 id="factorization-machines">Factorization Machines</h2>

<p>Original Matrix Factorization use only the rating information. What if we can get more features about the user and item? Such as the gender and age information of user, or the category or sale information about the item. Koren has mentioned that we can also use matrix factorization when adding more informations. For example, if we also have the implicit feedback such as the purchase or browsing history, as well as some user attributes. <br>
We denote <script type="math/tex" id="MathJax-Element-26225">N(u)</script> as the sets of items for which user <script type="math/tex" id="MathJax-Element-26226">u</script> has expressed an implicit feedback,  where each item <script type="math/tex" id="MathJax-Element-26227">i</script> is associated with <script type="math/tex" id="MathJax-Element-26228">x_i \in \mathbb{R}^f</script>. So a user who showed a preference for items in <script type="math/tex" id="MathJax-Element-26229">N(u)</script> is characterized by <script type="math/tex; mode=display" id="MathJax-Element-26230">|N(u)|^{-0.5}\sum_{i\in N(u)}x_i\tag{4}</script></p>

<p>Denote <script type="math/tex" id="MathJax-Element-26231">A(u)</script> as the set of attributes of a user <script type="math/tex" id="MathJax-Element-26232">u</script>, and a factor vector <script type="math/tex" id="MathJax-Element-26233">y_a \in \mathbb{R}^f</script> corresponds to each attribu to describe a user through the set of user-associated attributes:<script type="math/tex; mode=display" id="MathJax-Element-26234">\sum_{a\in A(u)}y^a\tag{5}</script></p>

<p>Then the predicted rating can be modeled by <br>
<script type="math/tex; mode=display" id="MathJax-Element-26235">\hat{r}_{ui}=\mu + b_i + b_u + q_i^T[p_u + |N(u)|^{-0.5}\sum_{i\in N(u)}x_i + \sum_{a\in A(u)}y^a]\tag{6}</script></p>

<p>Although Matrix Factorization can model such kind of implicit feedback and user or item attribute features, <script type="math/tex" id="MathJax-Element-26236">Factorization Machine</script> can handle such kind of features more directly. Assume that the user <script type="math/tex" id="MathJax-Element-26237">u</script> and item <script type="math/tex" id="MathJax-Element-26238">i</script> have feature vectors<script type="math/tex" id="MathJax-Element-26239">f_u</script> and <script type="math/tex" id="MathJax-Element-26240">g_i</script>, we can formulate the following regression cost function:<script type="math/tex; mode=display" id="MathJax-Element-26241">\min_w\sum_{u, i \in R}(R_{u, i} - w^T\begin{bmatrix}f_u \\ g_i\end{bmatrix})\tag{7}</script></p>

<p>The following cost function is a only linear combination of user and item features, which doesn’t consider the interaction between them. We can use the degree-2 polynomial mapping to hander such interaction:<script type="math/tex; mode=display" id="MathJax-Element-26242">\min_{w_{t, s}\forall t, s}\sum_{u, v \in R}(r_{u, i}-\sum_{t'=1}^U\sum_{s'=1}^Vw_{t', s'}(f_u)_{t'}(g_i)_{s'})^2\tag{8}</script></p>

<p>This is equivalent to <script type="math/tex; mode=display" id="MathJax-Element-26243">\min_W\sum_{u, i\in R}(r_{u, i}-f_u^TWg_i)^2\tag{9}</script> <br>
However, this setting fails for extreme sparse features.  <br>
Consider the most extreme situation where the <script type="math/tex" id="MathJax-Element-26244">f_u</script> and <script type="math/tex" id="MathJax-Element-26245">g_i</script> is the user ID and item ID features, then the optimal solution is <script type="math/tex; mode=display" id="MathJax-Element-26246">w_{u, i}=\begin{cases}r_{u, i}  & \text{if } u, i \in R \\ 0, & \text{if } u, i \notin r\end{cases}\tag{10}</script> <br>
So that we can never predict <script type="math/tex; mode=display" id="MathJax-Element-26247">r_{u, i}, u, i \notin R</script> <br>
The reason is that overfitting occurs, since the number of variables is much more than the number of instances:<script type="math/tex; mode=display" id="MathJax-Element-26248">\text{#variables} = mn \gg \text{#instances} = |R|</script> <br>
We can avoid this by letting<script type="math/tex; mode=display" id="MathJax-Element-26249">W \approx P^TQ</script> <br>
, where <script type="math/tex" id="MathJax-Element-26250">P</script> and <script type="math/tex" id="MathJax-Element-26251">Q</script> are both low-rank matrices. So it becomes matrix factorization. <br>
So we can reformulate <script type="math/tex" id="MathJax-Element-26252">(9)</script> as <script type="math/tex; mode=display" id="MathJax-Element-26253">\min_{u, i \in R}(R_{u, i} - f_u^TP^TQg_i)^2</script> <br>
We can think <script type="math/tex; mode=display" id="MathJax-Element-26254">Pf_u \text{ and }Qg_i</script> <br>
 as the latent representation of user <script type="math/tex" id="MathJax-Element-26255">u</script> and item <script type="math/tex" id="MathJax-Element-26256">i</script> in the latent space respectively. This is <strong>Factorization Machine</strong>.</p>



<h2 id="field-aware-factorization-machine">Field-Aware Factorization Machine</h2>

<p>Factorization Machine can effectively model the interaction between user and item, as well as the user side and item side features. But what if there are more than 3 dimension? For example, in the CTR prediction for computational advertising, we may have User, Advertisement as well as Publisher. There is interaction between the User and Advertisement, as well as interaction between User and Publisher. The Field-Aware Factorization Machine can handle all such interactions. <br>
The formulation of FFM is<script type="math/tex; mode=display" id="MathJax-Element-26257">\min_w \sum_{i=1}^L(log(1 + exp(-y_I \phi(w, x_i)))  + \frac{\lambda}{2}||w||^2</script></p>

<p>where</p>



<p><script type="math/tex; mode=display" id="MathJax-Element-26258">\phi(w, x) = \sum_{j_1, j_2 \in C_2}\langle w_{j_1, f_2}, w_{j_2, f_1} \rangle x_{j_1}x_{j_2}</script></p>

<p>where <script type="math/tex" id="MathJax-Element-26259">f_1, f_2</script> are respectively the field of <script type="math/tex" id="MathJax-Element-26260">j_1</script> and <script type="math/tex" id="MathJax-Element-26261">j_2</script>, and <script type="math/tex" id="MathJax-Element-26262">w_{j_1, f_2}</script> and <script type="math/tex" id="MathJax-Element-26263">w_{j_2, f_1}</script> are two vectors with length <script type="math/tex" id="MathJax-Element-26264">k</script>. Here the field means the dimension, for the recommender system, typically there are 2 dimensions: User and Item. For CTR prediction, there are typically 3 dimensions: User, Advertisement and Publisher. But actually we can build one dimension for each ID features, or even category features.  <br>
YuChin Juan has showed the formulation from linear model, to degree-2 polynomial model, to factorization machine and finally field-aware factorization machine models. <br>
The formulation of <strong>linear model</strong> is</p>



<p><script type="math/tex; mode=display" id="MathJax-Element-26265">\phi(w, x) = w^Tx = \sum_{j\in C_1}w_jx_J</script> <br>
where <script type="math/tex" id="MathJax-Element-26266">C_1</script> is the non-zero elements in <script type="math/tex" id="MathJax-Element-26267">x</script>. <br>
The formulation of <strong>Poly-2</strong> is <script type="math/tex; mode=display" id="MathJax-Element-26268">\phi(w, x) = \sum_{j_1, j_2 \in C_2}w_{j_1, j_2}x_{j_1}x_{j_2}</script> <br>
where <script type="math/tex" id="MathJax-Element-26269">C_2</script> is the 2-combination of non-zero elements in <script type="math/tex" id="MathJax-Element-26270">x</script>.</p>

<p>The formulation of <strong>factorization machine</strong> is </p>



<p><script type="math/tex; mode=display" id="MathJax-Element-26271">\phi(w, x) = \sum_{j_1, j_2 \in C_2}\langle w_{j_1}, w_{j_2}x_{j_1}x_{j_2}\rangle</script></p>

<p>where <script type="math/tex" id="MathJax-Element-26272">w_{j_1}</script> and <script type="math/tex" id="MathJax-Element-26273">w_{j_2}</script> are two vectors with length <script type="math/tex" id="MathJax-Element-26274">k</script>, and <script type="math/tex" id="MathJax-Element-26275">k</script> is a used-defined parameter.</p>

<p>The formulation of <strong>Field-Aware Factorization Machine</strong> is </p>



<p><script type="math/tex; mode=display" id="MathJax-Element-26276">\phi(w, x)=\sum_{j_1, j_2 \in C_2}\langle w_{j_1, f_2}, w_{j_2, f_1}\rangle x_{j_1}x_{j_2}</script></p>

<p>where <script type="math/tex" id="MathJax-Element-26277">f_1</script> and <script type="math/tex" id="MathJax-Element-26278">f_2</script> are respectively the fields of <script type="math/tex" id="MathJax-Element-26279">j_1</script> and <script type="math/tex" id="MathJax-Element-26280">j_2</script>. <br>
Here is a concrete example, say there is a sample:</p>

<table>
<thead>
<tr>
  <th>User(Us)</th>
  <th>Movie(Mo)</th>
  <th>Gender(Ge)</th>
  <th>Price(Pr)</th>
</tr>
</thead>
<tbody><tr>
  <td>YuChin(YC)</td>
  <td>3Idiots(3I)</td>
  <td>Comedy, Drama(Co, Dr)</td>
  <td>$9.99</td>
</tr>
</tbody></table>


<p>The <script type="math/tex" id="MathJax-Element-26281">\phi(w, x)</script> in FFM is </p>



<p><script type="math/tex; mode=display" id="MathJax-Element-26282">\langle w_{Us-Yu ,Mo} ,w_{ Mo-3I, Us}\rangle x_{Us-Yu} x_{Mo-3I}+ \langle w_{Us-Yu ,Ge} ,w_{ Ge-Co, Us}\rangle x_{Us-Yu} x_{Ge-Co} + \langle w_{ Us-Yu,Ge}, w_{Ge-Dr , Us}\rangle x_{Us-Yu} x_{Ge-Dr}  + \langle w_{ Us-Yu, Pr}, w_{ Pr, Us}\rangle x_{Us-Yu} x_{Pr} </script> <br>
<script type="math/tex; mode=display" id="MathJax-Element-26283">+ \langle w_{Mo-3I , Ge} ,w_{ Ge-Co, Mo}\rangle x_{Mo-3I} x_{Ge-Co}+ \langle w_{Mo-3I ,Ge} ,w_{Ge-Dr , Mo}\rangle x_{Mo-3I} x_{Ge-Dr}+ \langle w_{Mo-3I ,Pr} ,w_{ Pr, Mo}\rangle x_{Mo-3I} x_{Pr} </script> <br>
<script type="math/tex; mode=display" id="MathJax-Element-26284">+  \langle w_{Ge-Co ,Ge} ,w_{Ge-Dr , Ge}\rangle x_{Ge-Co} x_{Ge-Dr}+ \langle w_{Ge-Co ,Pr} ,w_{ Pr, Ge}\rangle x_{Ge-Co} x_{Pr}</script> <br>
<script type="math/tex; mode=display" id="MathJax-Element-26285"> + \langle w_{Ge-Dr ,Pr} ,w_{Pr , Ge}\rangle x_{Ge-Dr} x_{Pr}</script></p>

<p>Reference: <br>
[1]. Matrix Factorization Techniques for Recommender System. Y. Koren, et. at. <br>
[2]. Matrix Factorization and Factorization Machine for Recommender Systems[Slides]. Chih-Jen Lin. <br>
[3]. Factorization Machines. Steffen Rendle. <br>
[4]. 3 Idiots’ Approach for Display Advertising Challenge[Slides]. Yu-Chin Juan, Yong Zhuang, and Wei-Sheng Chin. <br>
[5]. Field-aware Factorization Machine[Slides]. Yu-Chin Juan, Yong Zhuang, and Wei-Sheng Chin. <br>
[6]. Pairwise interaction tensor factorization for personalized tag recommendation. Steffen Rendle, Lars Schmidt-Thieme.</p></div></body>
</html>