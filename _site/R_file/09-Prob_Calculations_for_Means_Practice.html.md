---
title: Probability Calculations for Means (Practice)
subtitle: Applying the CLT
format:
  html:
    self-contained: true
    embed-resources: true
execute: 
  keep-md: TRUE
---







# Instructions

Answer the following questions, render the document and submit the `.html` report.

# Questions

## GPA

Suppose the mean GPA of BYU-Idaho students is 3.5 and the standard deviation is 0.7. It is well known that this distribution is left-skewed. A random sample of n = 81 students will be drawn.

Use the following R code to answer the questions below:



::: {.cell}

```{.r .cell-code}
xbar <- 
mu <- 3.5
sigma <- 0.7
n <- 81
sigma_xbar <- 
sigma_xbar
```

::: {.cell-output .cell-output-error}

```
Error: object 'sigma_xbar' not found
```


:::

```{.r .cell-code}
z <- 
z
```

::: {.cell-output .cell-output-error}

```
Error: object 'z' not found
```


:::

```{.r .cell-code}
# Area to the left:
  
# Area to the right:
```
:::



__Question__: What is the mean of the distribution of the sample means ($\mu_{\bar{x}}$) for all possible samples of size 81 that could be drawn from the parent population of GPAs?  
__Answer__:  3.5

__Question__: What is the standard deviation  of the distribution of the sample means ($\sigma_{\bar{x}}$) for all possible samples of size 81 that could be drawn from the parent population of GPAs?  
__Answer__:  0.0778


::: {.cell}

```{.r .cell-code}
s <- 0.7/ sqrt(81)
s
```

::: {.cell-output .cell-output-stdout}

```
[1] 0.07777778
```


:::
:::




__Question__: What is the shape of the distribution of the sample means for all possible samples of size 81 that could be drawn from the parent population of GPAs?  
__Answer__:  approximatelly normal


__Question__: What is the probability that the mean GPA for 81 randomly selected BYU-Idaho students will be less than 3.3?  
__Answer__:  The answer is 0.51%



::: {.cell}

```{.r .cell-code}
z <- (3.3-3.5) / (0.7/sqrt(81))
pnorm(z) * 100
```

::: {.cell-output .cell-output-stdout}

```
[1] 0.5063995
```


:::
:::



## GRE

Scores on the quantitative portion of the GRE are approximately normally distributed with mean, $\mu = 150.8$ and standard deviation, $\sigma = 8.8$.
 
Use the following R code to answer the questions below:



::: {.cell}

```{.r .cell-code}
x<- 160
mu <- 150.8
sigma <- 8.8
n <- 
sigma_xbar <- 
sigma_xbar
```

::: {.cell-output .cell-output-error}

```
Error: object 'sigma_xbar' not found
```


:::

```{.r .cell-code}
z <- 
z

# Area to the left:
  
# Area to the right:

# Percentile (qnorm())
```
:::



__Question__: Dianne earned a score of 160 on the quantitative portion of the GRE. What is the z-score corresponding to Dianne’s score?  
__Answer__:   1.045


::: {.cell}

```{.r .cell-code}
z <- (160-150.8) / 8.8
z
```

::: {.cell-output .cell-output-stdout}

```
[1] 1.045455
```


:::
:::




__Question__: What is the probability that a randomly selected student will score *above* 160 on the quantitative portion of the GRE?  
__Answer__: 14.79%


::: {.cell}

```{.r .cell-code}
1-pnorm(z) 
```

::: {.cell-output .cell-output-stdout}

```
[1] 0.1479065
```


:::
:::




__Question__: What GRE score (n=1) corresponds to the 95th percentile?  
__Answer__: 165.1


::: {.cell}

```{.r .cell-code}
qnorm(.95, 150.8, 8.8)
```

::: {.cell-output .cell-output-stdout}

```
[1] 165.2747
```


:::
:::




__Question__:  What is the probability that the *average* GRE score of 5 randomly selected students will be above 160?  
__Answer__: 0.97%



::: {.cell}

```{.r .cell-code}
z <- (160-150.8) / (8.8/sqrt(5))
1-pnorm(z)
```

::: {.cell-output .cell-output-stdout}

```
[1] 0.009701215
```


:::
:::



## Coast Guard

The United States Coast Guard assumes the mean weight of passengers in commercial boats is 185 pounds. The previous value was lower, but was raised after a tragic boating accident. The standard deviation of passenger weights is 26.7 pounds. The weights of a random sample of 48 commercial boat passengers were recorded. The sample mean was determined to be 177.6 pounds. 

Use the following R code to answer the questions below:



::: {.cell}

```{.r .cell-code}
xbar <- 177.6
mu <- 185
sigma <- 26.7
n <- 48
sigma_xbar <- 26.7 / sqrt(n)


z <- (xbar-mu) / sigma_xbar


# Area to the left:
pnorm(z)
```

::: {.cell-output .cell-output-stdout}

```
[1] 0.02741782
```


:::

```{.r .cell-code}
# Area to the right:
1-pnorm(z)
```

::: {.cell-output .cell-output-stdout}

```
[1] 0.9725822
```


:::
:::




__Question__: Find the probability that a random sample of passengers will have a mean weight that is as extreme or more extreme (either above or below the mean) than was observed in this sample.  
__Answer__:  5.48%



::: {.cell}

```{.r .cell-code}
pnorm(z) * 2
```

::: {.cell-output .cell-output-stdout}

```
[1] 0.05483565
```


:::
:::



__HINT__:  To get a value "as extreme" means it could be higher or lower.  Because the normal distribution is symmetric, if the $z<0$, we can take the area to the *left* of $z$ and multiply by 2.  Conversely, if $z>0$ we can take the area to the right and multiply by 2.  


## Tankers

Tanker trucks are designed to carry huge quantities of gasoline from refineries to filling stations. A factory that manufactures the tank of the trucks claims to manufacture tanks with a capacity of 8550 gallons of gasoline. The actual capacity of the tanks is normally distributed with mean, $\mu = 8544$ gallons, and standard deviation, $\sigma=12$ gallons.

Use the following R code to answer the questions below:



::: {.cell}

```{.r .cell-code}
xbar <- 
mu <- 8544
sigma <- 12
n <- 1
sigma_xbar <- sigma / sqrt(n)
sigma_xbar
```

::: {.cell-output .cell-output-stdout}

```
[1] 12
```


:::

```{.r .cell-code}
z <- (xbar - mu) / sigma_xbar
z
```

::: {.cell-output .cell-output-stdout}

```
[1] 0
```


:::

```{.r .cell-code}
# Area to the left:
pnorm(z)
```

::: {.cell-output .cell-output-stdout}

```
[1] 0.5
```


:::

```{.r .cell-code}
#Area to the right
1-pnorm(z)
```

::: {.cell-output .cell-output-stdout}

```
[1] 0.5
```


:::

```{.r .cell-code}
## Area between A and B
A <- 
B <- 

pnorm(B, mu, sigma_xbar) - pnorm(A, mu, sigma_xbar)
```

::: {.cell-output .cell-output-error}

```
Error: object 'B' not found
```


:::
:::




__Question__: Find the **z-score** corresponding to a single tank ($n=1$) with a capacity of 8550 gallons. Round your answer to one decimal place.  
__Answer__:  50%



::: {.cell}

```{.r .cell-code}
z <- (8550-8544) / 12
z
```

::: {.cell-output .cell-output-stdout}

```
[1] 0.5
```


:::
:::




__Question__: What is the probability that a randomly selected tank will have a capacity of **less** than 8550 gallons?  
__Answer__: 69.15%



::: {.cell}

```{.r .cell-code}
pnorm(8550, 8544, 12)
```

::: {.cell-output .cell-output-stdout}

```
[1] 0.6914625
```


:::
:::




__Question__: A simple random sample of $n = 20$ tanks was selected. Find the z-score corresponding to a sample mean capacity for 20 tanks of 8550.  
__Answer__: 2.2361



::: {.cell}

```{.r .cell-code}
z <- (8550 - 8544) / (12 / sqrt(20))
z
```

::: {.cell-output .cell-output-stdout}

```
[1] 2.236068
```


:::
:::




__Question__: What is the probability that the sample mean of $n=20$ randomly selected tanks will be between 8541 and 8547?  
__Answer__: 73.64%



::: {.cell}

```{.r .cell-code}
z1 <- (8541-8544) / (12 / sqrt(20))
z2 <- (8547-8544) / (12 / sqrt(20))

pnorm(z2) - pnorm(z1)
```

::: {.cell-output .cell-output-stdout}

```
[1] 0.7364475
```


:::
:::
