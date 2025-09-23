---
title: Probability Calculations for Means (Class)
format:
  html:
    self-contained: true
    embed-resources: true
execute: 
  keep-md: TRUE
---







# The Central Limit Theorem

The Central Limit Theorem states that for a large enough sample size ($n>30$) the distribution of _sample means_ is approximately normal with mean, $\mu_{\bar{x}} = \mu$ and $\sigma_{\bar{x}} = \frac{\sigma}{\sqrt{n}}$ regardless of the distribution of the population.  

We can assume the distribution of sample means is approximately normal if:

1. The population is normally distributed
2. n > 30

Don't forget, that if the population is normally distributed, so is the distribution of sample means __regardless of sample size__. 

# Probability Calculations for the Sampling Distribution of $\bar{x}$

When we are confident that the sampling distribution of $\bar{x}$ is normal, we can use the `pnorm()` function just as we did before.  We simply input the mean, $\mu_{\bar{x}} = \mu$ and $\sigma_{\bar{x}} = \frac{\sigma}{\sqrt{n}}$ into the Z-score formula as follows:  



::: {.cell}

```{.r .cell-code}
xbar <- 13.5
mu <- 20
sigma <- 15
n <- 33
sigma_xbar <- sigma / sqrt(n)

z <- (xbar - mu) / (sigma_xbar)
z
```

::: {.cell-output .cell-output-stdout}

```
[1] -2.48931
```


:::

```{.r .cell-code}
# Left tail [Prob(some sample mean LESS THAN xbar)]
pnorm(z)
```

::: {.cell-output .cell-output-stdout}

```
[1] 0.006399557
```


:::

```{.r .cell-code}
# Right tail [Prob(some sample mean GREATER THAN xbar)]
1-pnorm(z)
```

::: {.cell-output .cell-output-stdout}

```
[1] 0.9936004
```


:::

```{.r .cell-code}
# The probability of a sample mean being between A and B

A <- 9
B <- 15
pnorm(B, mu, sigma_xbar) - pnorm(A, mu, sigma_xbar)
```

::: {.cell-output .cell-output-stdout}

```
[1] 0.02774293
```


:::

```{.r .cell-code}
# Calculating Percentiles: What is the sample mean below which a certain percent is found? 

qnorm(.25, mu, sigma_xbar)
```

::: {.cell-output .cell-output-stdout}

```
[1] 18.2388
```


:::
:::



# One Z Formula to Rule Them All

We do not have to worry about choosing between two different Z-score formulas (one for individuals and a separate one for means).  

If we are calculating probabilities for a normal population and we want to calculate the probability of a single individual, we can simply set $n=1$.  

_WARNING_:  Using sample size of 1 only works if the population is already normally distributed.  But if we confident that it is, we can use the following calculations:  



::: {.cell}

```{.r .cell-code}
xbar <- 13.5
mu <- 20
sigma <- 15
sigma_xbar <- sigma / sqrt(n)
n <- 1

z <- (xbar - mu) / (sigma_xbar)
z
```

::: {.cell-output .cell-output-stdout}

```
[1] -2.48931
```


:::

```{.r .cell-code}
# Left tail [Prob(some sample mean LESS THAN xbar)]
pnorm(z)
```

::: {.cell-output .cell-output-stdout}

```
[1] 0.006399557
```


:::

```{.r .cell-code}
# Right tail [Prob(some sample mean GREATER THAN xbar)]
1-pnorm(z)
```

::: {.cell-output .cell-output-stdout}

```
[1] 0.9936004
```


:::
:::




# Practice Together

Use the following calculator to answer the questions below.  

__REMEMBER__: Whenever you update information, it is useful to run the whole code chunk.  









## GPA's

Suppose the mean GPA of BYU-Idaho students is 3.5 and the standard deviation is 0.7. It is well known that this distribution is left-skewed. A random sample of n = 45 students will be drawn.

__Question__: What is the mean of the distribution of the sample means (sampling distribution) for all possible samples of size 45 that could be drawn from the parent population of GPAs?  
__Answer__:  3.5

__Question__:  What is the standard deviation of the distribution of the sample means (sampling distribution) for all possible samples of size 45 that could be drawn from the parent population?  
__Answer__:  0.7


__Question__:  What is the probability that the mean GPA for 45 randomly selected BYU-Idaho students will be less than 3.3?  
__Answer__:  10.435%


::: {.cell}

```{.r .cell-code}
sigma_xbar <- sigma / sqrt(n)
sigma_xbar * 100
```

::: {.cell-output .cell-output-stdout}

```
[1] 1500
```


:::
:::




__Question__:  What is the shape of the distribution of sample means, $\bar{x}$, when 45 students are selected?  
__Answer__:  the distribution is approximatelly normal.


## GRE Scores

Scores on the quantitative portion of the GRE are approximately normally distributed with mean, $\mu=150.8$,
 and standard deviation $\sigma = 8.8$.
 


::: {.cell}

```{.r .cell-code}
mu <- 150.8
sigma <- 8.8
x <- 160

z <- (x-mu) / sigma
z
```

::: {.cell-output .cell-output-stdout}

```
[1] 1.045455
```


:::
:::


 
 
__Question__:  Dianne earned a score of 160 on the quantitative portion of the GRE. What is the z-score corresponding to Dianne’s score?  
__Answer__:  1.0455


__Question__:  What is the probability that a randomly selected student will score above 160 on the quantitative portion of the GRE?  
__Answer__:  14.791%


::: {.cell}

```{.r .cell-code}
(1-pnorm(160, mu, sigma)) * 100
```

::: {.cell-output .cell-output-stdout}

```
[1] 14.79065
```


:::
:::





__Question__: What is the probability that a randomly selected *group* of 18 students will have an *average* less than 160 on the quantitative portion of the GRE?  
__Answer__:  99.99%


::: {.cell}

```{.r .cell-code}
sigma_xbar <- sigma / sqrt(18)
z <- (160-mu) / sigma_xbar
pnorm(z)
```

::: {.cell-output .cell-output-stdout}

```
[1] 0.9999954
```


:::
:::




__Question__: What is the probability that a randomly selected group of 18 students will have an *average* between 145 and 160 on the quantitative portion of the GRE?  
__Answer__: 99.74%


::: {.cell}

```{.r .cell-code}
sigma_xbar <- sigma / sqrt(18)
z1 <- (160-mu) / sigma_xbar
z2 <- (145-mu) / sigma_xbar

pnorm(z1) - pnorm(z2)
```

::: {.cell-output .cell-output-stdout}

```
[1] 0.9974107
```


:::
:::
