> originally posted on
> [giuliostarace.com/posts/dlml-tutorial/](https://www.giuliostarace.com/osts/dlml-tutorial/)
> (recommended for better math rendering)

# Discretized Logistic Mixture Likelihood - The Why, What and How

In this post I will explain what the Discretized Logistic Mixture Likelihood
(DLML)[^1] is. This is a modeling method particularly relevant to my MSc thesis,
where I use it to model the continuous action my imitation learning agent should
make. While there are already some great posts explaining the concept, the
information is scattered, which can make understanding the concept a bit
painful. I will first start with motivating _why_ we need DLML. I will then
present _what_ DLML is. Finally I will outline _how_ we can implement DLML in
[PyTorch](https://pytorch.org/).

<!-- vim-markdown-toc GFM -->

* [The Why](#the-why)
* [The What (and some more why)](#the-what-and-some-more-why)
   * [Some more why](#some-more-why)
* [The How](#the-how)
   * [Training](#training)
   * [Sampling](#sampling)
* [Closing words](#closing-words)
* [More Resources](#more-resources)

<!-- vim-markdown-toc -->

## The Why

Suppose you wish to predict some variable that happens to be continuous,
conditional on some other quantity. For example, you are interested in
predicting the value of a given pixel in an image, given the values of
neighbouring pixels.

With a bit of domain knowledge, this class of problem can typically be
reformulated by discretizing the target variable and modeling the resulting
(conditional) probability distribution. The prediction task can then be posed as
a classification one over the discretized bins: apply a
[softmax](https://en.wikipedia.org/wiki/Softmax_function) and train using
[cross-entropy loss](https://en.wikipedia.org/wiki/Cross_entropy).

This is (in part) what the authors of
[PixelCNN](https://arxiv.org/abs/1606.05328) did: for the task of conditional
image generation, they model each (sub)pixel of an image with a softmax over a
256-dimensional vector, where each dimension represents an 8-bit intensity value
that the pixel may take. There are more details, particularly around the
conditioning, but that's all you need to know for now for the premise of this
tutorial.

Immediately, we face a number of limitations:

1. Softmax can be
   [computationally expensive](https://en.wikipedia.org/wiki/Softmax_function#Computational_complexity_and_remedies)
   and [unstable](https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/).
   This is particularly problematic for high-dimensional inputs, which are
   usually the case when dealing with (discretized) continuous variables. This
   is particularly problematic if you plan to repeat the computation on several
   output variables (e.g. several pixels of an output image, several dimensions
   of robotic arm rotation, etc.), which is usually the case when interested in
   conditional generation of continuous variables.

2. [Softmax can lead to sparse gradients](https://arxiv.org/abs/1811.10779),
   especially at the beginning of training, which can slow down learning. This
   is also especially the case with high-dimensional input.

3. Softmax does not model any sort of ordinality in the random variable that is
   being considered: every single dimension in the input vector is considered
   independently. There is no notion that a value of 127 is close to 128. This
   ordinality is typically present when dealing with (discretized) continuous
   variables by virtue of their nature. Rather than relying on some inductive
   bias, the model has to devote more training time to learn this aspect of the
   data, leading to slower training.

4. Softmax fails to properly model values that are never observed, assigning
   probabilities of 0 to values that may otherwise be more likely to occur.

These issues are at least some of the motivations for using DLML, which I will
introduce in the next section.

## The What (and some more why)

In DLML, for a given output variable $y$ we do the following:

1. We assume that there is a latent value $v$ with a continuous distribution.
2. We take $y$ to come from a discretization of this continuous distribution of
   $v$. We do this discretization in some arbitrary way, but usually by rounding
   to the nearest 8-bit representation. What this means is that if e.g. $v$ can
   be any value between 0 and 255, then $y$ will be any _integer_ between those
   two numbers.
3. We model $v$ using a simple continuous distribution - e.g. the
   [logistic distribution](https://en.wikipedia.org/wiki/Logistic_distribution).
4. We then take a further step, choosing to model $v$ as a mixture of $K$
   logistic distributions:

   $$ v \sim \sum_i^K \pi_i \text{logistic}(\mu_i, s_i), $$

   (equation 1) where $\pi_i$ is some coefficient weighing the likelihood of the
   $i$th distribution while $\mu_i$ and $s_i$ are the mean and scale
   parametrizing it.

5. To compute the likelihood of $y$, we sum its (weighted) probability masses
   over the $K$ mixtures. We can obtain the probability masses by computing the
   difference between consecutive cumulative density function (CDF) values of
   equation (1). Note that the
   [CDF of the logistic distribution is a sigmoid function](https://en.wikipedia.org/wiki/Logistic_distribution#Cumulative_distribution_function).
   We therefore write:

   $$
   p(y | \mathbf{\pi}, \mathbf{\mu}, \mathbf{s} )  = \sum_{i=1}^K \pi_i
   \left[\sigma\left(\frac{y + 0.5 - \mu_i}{s_i}\right) -
   \sigma\left(\frac{y - 0.5 - \mu_i}{s_i}\right)\right],
   $$

   (equation 2) where $\sigma$ is the logistic sigmoid. The 0.5 value comes from
   the fact that we have discretized $v$ into $y$ through rounding, and
   therefore successive values of our discrete random variable $y$ are found at
   this boundary.

6. We can additionally model edge cases, replacing $y - 0.5$ with $-\infty$ when
   $y=0$ and $y + 0.5$ with $+\infty$ when $y = 2^8 = 255$.

This is nothing more than a likelihood, so we can use it in a
[maximum likelihood estimation (MLE)](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation)
process to estimate our parameters. In the case of Deep Learning, we use the
negative log likelihood as our loss function.
[This comment on GitHub](https://github.com/Rayhane-mamah/Tacotron-2/issues/155#issuecomment-413364857)
provides a different perspective to what's going on.

### Some more why

This approach provides a number of advantages, many of which address the
shortcomings of the softmax approach described in
[the previous section](#the-why). In particular:

1. It avoids assigning probability mass outside the valid range of [0, 255] by
   explicitly modeling the rounding and edge cases.

2. Edge values are naturally assigned higher probability values, which tends to
   align with what is observed when dealing with this nature of data.

3. We rely on the simple sigmoid function, which is less computationally
   expensive than its multi-class cousin the softmax. This addresses limitation
   1 from above.

4. Because we are now making use of the logistic distribution to model the
   (latent) value of $y$, we are implicitly also modelling ordinality when
   discretizing, since the logistic distribution is continuous. This addresses
   limitation 3 from above.

5. Our reliance on a continuous distribution similarly addresses limitation 4
   from above, as we will no longer assign non-zero probability prematurely.

6. Empirically it has been found that only a small number of mixtures,
   $\le
   10$, is enough. What this means is that we can work with much lower
   dimensionality network outputs (3 parameters: $\mu$, $s$ and $\pi$ for each
   mixture element), leading to denser gradients. This addresses limitation 2
   from above.

7. Because we make use of a mixture, we can more easily model multi-modal data.
   This can be desirable when learning skills from imitation, where the same
   skill can be shown to be completed through different action sequences. It is
   exactly for this reason that
   [Lynch et al. 2020](https://proceedings.mlr.press/v100/lynch20a.html) and
   [Mees et al. 2022](https://arxiv.org/abs/2204.06252) make use of DLML in
   their action decoders.

## The How

So how do we actually go about implementing this? This is one of those
techniques where we do slightly different things depending on whether we are
training or whether we just want outputs from our model (sampling). For
completeness, I provide a full-reference to the code below on my
[GitHub](https://github.com/thesofakillers/dlml-tutorial).

### Training

Earlier, we defined a likelihood for $y$. We can use this to train our model to
output the appropriate parameters $\mu$, $s$, and $\pi$ for a given input using
MLE. In practice we will calculate the likelihood and then minimize the negative
log likelihood.

For a given output variable $y$, using $K$ mixture elements, our model should
therefore output $K$ means, scales and mixture logits:

```python {linenos=table}
# each of these have shape (B x K)
means, log_scales, mixture_logits = model(**batch['inputs'])
inv_scales = torch.exp(-log_scales)
```

We treat the predicted scales as $\log(s)$, which we can then follow with an
exponential to recover $s$. This is to enforce positive values of $s$ and for
numerical stability. In practice we take an exponential of the negative
`log_scales` to obtain `inv_scales` i.e. $\frac{1}{s}$, since $s$ is always used
in the denominator in our formulas.

We can then start computing the rest of the terms for our likelihood from
equation (2).

```python {linenos=table}
y = batch['targets']
# explained in text
epsilon = (0.5*y_range) / (num_y_vals - 1)
# convenience variable
centered_y = y - means
# inputs to our sigmoid functions
upper_bound_in = inv_scales * (centered_y + epsilon)
lower_bound_in = inv_scales * (centered_y - epsilon)
# remember: cdf of logistic distr is sigmoid of above input format
upper_cdf = torch.sigmoid(upper_bound_in)
lower_cdf = torch.sigmoid(lower_bound_in)
# finally, the probability mass and equivalent log prob
prob_mass = upper_cdf - lower_cdf
vanilla_log_prob = torch.log(torch.clamp(prob_mass, min=1e-12))
```

Before I go on, you may be asking - "what is this epsilon? Weren't we
adding/subtracting $0.5$ instead?". Indeed, this is actually still the case.
However, here we are operating on the assumption that we have scaled our $y$'s
to be in the range [-1, 1]. For 8-bit data, that is equivalent to scaling by
$\frac{2}{2^8 - 1}$. We have to apply this same scaling to our $0.5$ boundaries
for consistency. Note that `y_range` is simply $1 - (-1) = 2$, the $2$ in the
numerator, and that `num_classes` for $y$ is $ 2^8 = 256 $. We now move on to
the edge cases described in step 6 of [the what](#the-what-and-some-more-why).

```python {linenos=table}
# edges
# log probability for edge case of 0 (before scaling)
low_bound_log_prob = upper_bound_in - F.softplus(upper_bound_in)
# log probability for edge case of 255 (before scaling)
upp_bound_log_prob = - F.softplus(lower_bound_in)
# middle
mid_in = inv_scales * centered_y
log_pdf_mid = mid_in - log_scales - 2.0 * F.softplus(mid_in)
log_prob_mid = log_pdf_mid - np.log((num_classes - 1) / 2)
```

"Woah, what is going on here?" you may ask. Here we are making use of the
[softplus-sigmoid relations](https://github.com/openai/pixel-cnn/issues/23), in
particular that

$$
\zeta(x) = \int_{-\infty}^x \sigma(y)dy,
$$

(equation 3) where $\zeta$ is the
[softplus function](https://jiafulow.github.io/blog/2019/07/11/softplus-and-softminus/).
We also approximate the log probability at the center of the bin, based on the
assumption that the log-density is constant in the bin of the observed value.
This is used as a backup in cases where calculated probabilities are below 1e-5,
which could happen due to numerical instability. This case is extremely rare and
I would not dedicate too much thought to it, it is just there as a (rarely-used)
backup.

We can now put all these terms together into a single log likelihood tensor:

```python {linenos=table}
# Create a tensor with the same shape as 'y', filled with zeros
log_probs = torch.zeros_like(y)
# conditions for filling in tensor
is_near_min = y < output_min_bound + 1e-3
is_near_max = y > output_max_bound - 1e-3
is_prob_mass_sufficient = prob_mass > 1e-5
# And then fill it in accordingly
# lower edge
log_probs[is_near_min] = low_bound_log_prob[is_near_min]
# upper edge
log_probs[is_near_max] = upp_bound_log_prob[is_near_max]
# vanilla case
log_probs[
    ~is_near_min & ~is_near_max & is_prob_mass_sufficient
] = vanilla_log_prob[
    ~is_near_min & ~is_near_max & is_prob_mass_sufficient
]
# extreme case where prob mass is too small
log_probs[
    ~is_near_min & ~is_near_max & ~is_prob_mass_sufficient
] = log_prob_mid[
    ~is_near_min & ~is_near_max & ~is_prob_mass_sufficient
]
```

We are almost done, but there is one last piece. So far we have computed the
terms to minimize for learning the distribution(s), i.e. learning $\mu$ and $s$.
We also need to learn which mixture distribution to sample from, i.e. we have to
learn $\pi$. This is very simple, and consists in adding a log of the softmax
over the logits (the $\pi_i$) outputted by our model:

```python {linenos=table}
# modeling which mixture to sample from
log_probs = log_probs + F.log_softmax(mixture_logits, dim=-1)
```

We add a log of the softmax because we are after
$\log(\text{softmax}(\pi) \cdot \text{likelihood})$ which, by applying log
properties, is equivalent to
$\log(\text{softmax}(\pi)) + \log(\text{likelihood})$, which is what we get.

All that's left to do now is summing over our mixtures. We do this after
applying the
[Log-Sum-Exp trick for numerical stability](https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/)

```python {linenos=table}
log_likelihood = torch.sum(log_sum_exp(log_probs), dim=-1)
```

Our loss is the negative log likelihood, which we can choose to reduce across
the batch or return unreduced

```python {linenos=table}
loss = - log_likelihood

if reduction == 'mean'
   loss = torch.mean(loss)
elif reduction =='sum'
   loss =  torch.sum(loss)
```

And that's it for training. Once you have your loss, you can run
loss.backwards() and all the cool stuff torch provides with autodiff.

### Sampling

Sampling is fortunately a bit easier, and some people start their explanation
from here. Here, we first sample a distribution from our mixture, and then
sample a value from the sampled distribution. We have logits for each
distribution in our mixture, so we can sample from a softmax over this
distribution. In practice, we make use of the
[Gumbel-Max trick](https://timvieira.github.io/blog/post/2014/07/31/gumbel-max-trick/),
to keep things differentiable.

```python {linenos=table}
# each of these have shape (B x K)
means, log_scales, mixture_logits = model(**batch['inputs'])

# gumbel-max sampling
r1, r2 = 1e-5, 1.0 - 1e-5
temp = (r1 - r2) * torch.rand(means.shape, device=means.device) + r2
temp = mixture_logits - torch.log(-torch.log(temp))
argmax = torch.argmax(temp, -1)
```

`argmax` is the index of our sampled distribution. We can use it to get the
distribution's mean and scale from our model's outputs:

```python {linenos=table}
# (K dimensional vector, e.g. [0 0 0 1 0 0 0 0] for k=8, argmax=3
dist_one_hot = torch.eye(k)[argmax]

# use it to sample, and aggregate over the batch
sampled_log_scale = (dist_one_hot * log_scales).sum(dim=-1)
sampled_mean = (dist_one_hot * means).sum(dim=-1)
```

We can then sample from our logistic distribution using
[inverse sampling](https://www.statisticshowto.com/inverse-sampling/). For a
logistic distribution, this consists in

$$
X = \mu + s \log \left(\frac{y}{1-y} \right),
$$

(equation 4) where we select y from a random uniform distribution. In code:

```python {linenos=table}
# scale the (0,1) uniform distribution and re-center it
y = (r1 - r2) * torch.rand(means.shape, device=means.device) + r2

sampled_output = sampled_mean + torch.exp(sampled_log_scale) * (
    torch.log(y) - torch.log(1 - y)
)
```

And just like that, we have a way of sampling from our model.

## Closing words

I hope this post was helpful. I came across DLML during my MSc thesis on
language-enabled imitation learning and while there are several high quality
posts online, I couldn't find a single one that summarized the process in its
entirety, from motivation to implementation, so I decided to write it myself,
also as a way to help me understand the concept. As a reminder, the complete
code accompanying this post is available on
[my GitHub profile](https://github.com/thesofakillers/dlml-tutorial).

## More Resources

- [Great comment on Tacotron GitHub](https://github.com/Rayhane-mamah/Tacotron-2/issues/155#issuecomment-413364857)
- [Somewhat outdated Google Colab](https://colab.research.google.com/github/tensorchiefs/dl_book/blob/master/chapter_06/nb_ch06_01.ipynb)
- [_PixelCNN++: Improving the PixelCNN with Discretized Logistic Mixture Likelihood and Other Modifications_, Salimans et al., 2017.](https://openreview.net/forum?id=BJrFC6ceg)
- [_Learning Latent Plans from Play_, Lynch et al., 2020.](https://proceedings.mlr.press/v100/lynch20a.html)
- [_What Matters in Language Conditioned Robotic Imitation Learning over Unstructured Data_, Mees et al., 2022.](https://arxiv.org/abs/2204.06252)

[^1]:
    Also known as "Discretized Mixture of Logistics (DMoL)", "Discretized
    Logistic Mixture (DLM)", "Mixture of Discretized Logistics (MDL)"
