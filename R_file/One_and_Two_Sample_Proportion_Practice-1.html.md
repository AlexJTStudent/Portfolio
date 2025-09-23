---
title: "One and Two Sample Proportion Practice"
execute:
  keep-md: TRUE
---







# One-Sample Test of Proportions

## Teenage Smoking Habits

A study was conducted to determine the proportion of American teenagers between 13 and 17 who smoke. A survey from 10 years ago claimed that 15% percent of all teenagers smoke. 

A new Gallup survey interviewed a nationally representative sample of 785 teenagers aged 13 to 17. Seventy-one (71) teenagers in the survey acknowledged having smoked at least once in the past week. 

We want to see if the new study shows a *decrease* in the  percentage of teenagers who smoke from the 15% reported a decade ago.  

Perform the appropriate hypothesis test and create a confidence interval for the true proportion of teenagers who smoke.

### Hypothesis Test

State the null and alternative hypotheses:  

$$H_0 = 15$$  
$$H_a<15 $$
Choose your confidence level:

$$\alpha = 0.05$$  

Perform the appropriate analysis:  



::: {.cell}

```{.r .cell-code}
prop.test(x = 71, n = 785, alternative = "less")
```

::: {.cell-output .cell-output-stdout}

```

	1-sample proportions test with continuity correction

data:  71 out of 785, null probability 0.5
X-squared = 525.05, df = 1, p-value < 2.2e-16
alternative hypothesis: true p is less than 0.5
95 percent confidence interval:
 0.0000000 0.1094084
sample estimates:
         p 
0.09044586 
```


:::
:::



__Question__:  What is the P-value?  
__Answer__:  2.2e-16

__Question__:  Based on your chosen $\alpha$ and P-value, what is your conclusion?  
__Answer__:  the p-value is smaller than alpha of 0.05. Therefore, there is enough evidence to reject the null that smoking proportions haven't change.


#### Requirements

Recall that we must check that we have a big enough sample size to trust our p-value.  To do this, we check that there are at least 10 expected "success" and "failures" for a given sample size, n:  

Check:  

$$np \ge 10$$

$$n(1-p) \ge 10$$


::: {.cell}

```{.r .cell-code}
# Fill in n and the hypothesized p
n <- 785
p <- .15

n*p >= 10
```

::: {.cell-output .cell-output-stdout}

```
[1] TRUE
```


:::

```{.r .cell-code}
n*(1-p) >= 10
```

::: {.cell-output .cell-output-stdout}

```
[1] TRUE
```


:::
:::




__QUESTION__:  Are the requirements for a hypothesis test satisfied?  
__ANSWER__:  We have enough data to assume the probability to be normal.


### Confidence Interval

Create a 99\% confidence interval for the true population proportion of teenagers who smoke?  



::: {.cell}

```{.r .cell-code}
prop.test(x = 71, n = 785, conf.level = .99)
```

::: {.cell-output .cell-output-stdout}

```

	1-sample proportions test with continuity correction

data:  71 out of 785, null probability 0.5
X-squared = 525.05, df = 1, p-value < 2.2e-16
alternative hypothesis: true p is not equal to 0.5
99 percent confidence interval:
 0.06684739 0.12107323
sample estimates:
         p 
0.09044586 
```


:::
:::




__QUESTION__:  Interpret the confidence interval in context of the research question:  
__ANSWER__:  I have 99% percent confidence that the true smoking proportion is within 6.684739% and 12.107323%.


#### Check Requirements

Recall that Confidence Intervals do not depend on a hypothesized proportion, so the requirements are a little different.  For Confidence Intervals we check:  

$$n\hat{p} \ge 10$$

$$n(1-\hat{p}) \ge 10$$



::: {.cell}

```{.r .cell-code}
# Fill in X and N and check that there are enough "successes" and "failures"

x <- 71
n <- 785
phat <- x/n

n*phat >= 10
```

::: {.cell-output .cell-output-stdout}

```
[1] TRUE
```


:::

```{.r .cell-code}
n*(1-phat) >= 10
```

::: {.cell-output .cell-output-stdout}

```
[1] TRUE
```


:::
:::





# 2-Sample Proportion Tests

## A Penny Saved?

A random sample of BYU-Idaho students was surveyed and asked if they were in favor of retaining the penny as a form of currency in the United States. Out of the 116 women surveyed, 80 said that they were in favor of retaining the penny as a form of currency. Of the 137 men surveyed, 91 said that they were in favor of retaining the penny. 

For these questions, let group 1 represent women and group 2 represent men. 


### Hypothesis Test

Test to see if there is a *difference* between the proportion of women who want to keep the penny and the proportion of men who want to keep the penny. Use a level of significance of $\alpha = 0.05$.

State your null and alternative hypotheses (replace the question marks with the appropriate symbols):  

$$H_0: p_{female} =  p_{male}$$  
$$H_a: p_{female} != p_{male}$$  

Perform the appropriate test:  


::: {.cell}

```{.r .cell-code}
prop.test(x = c(91,80), n = c(137,116), alternative = "two.sided")
```

::: {.cell-output .cell-output-stdout}

```

	2-sample test for equality of proportions with continuity correction

data:  c(91, 80) out of c(137, 116)
X-squared = 0.087429, df = 1, p-value = 0.7675
alternative hypothesis: two.sided
95 percent confidence interval:
 -0.14888701  0.09804382
sample estimates:
   prop 1    prop 2 
0.6642336 0.6896552 
```


:::
:::




__Question__:  What is the P-value?  
__Answer__:  0.7675


__Question__:  Based on $\alpha = 0.05$, state your conclusion in context of the research question:  
__Answer__:  The P-value is not smaller than the alpha value, therefore there isn't enough evidence to reject the null and believe there is a significant difference between male and female proportions.



### Confidence Interval

Create a 95\% confidence interval for the *difference* in the proportion of females to males who prefer to keep the penny:



::: {.cell}

```{.r .cell-code}
prop.test(x = c(91,80), n = c(137,116), conf.level = .95)
```

::: {.cell-output .cell-output-stdout}

```

	2-sample test for equality of proportions with continuity correction

data:  c(91, 80) out of c(137, 116)
X-squared = 0.087429, df = 1, p-value = 0.7675
alternative hypothesis: two.sided
95 percent confidence interval:
 -0.14888701  0.09804382
sample estimates:
   prop 1    prop 2 
0.6642336 0.6896552 
```


:::
:::



__Question__:  Interpret the confidence interval in context of the research question:  
__Answer__:  The confidence interval of 95% contains a 0 within  -0.14888701 and 0.09804382, suggesting there is no significant difference between the two proportions. 


### Test Requirements

Recall that the requirements for Hypothesis Testing and Confidence Intervals for 2-sample proportions are the same.  We must check that there are more than 10 "successes" and "failures" in *both* samples:  

$$ n_1\hat{p}_1 \ge 10$$
$$n_1(1-\hat{p}_1) \ge 10$$
$$ n_2\hat{p}_2 \ge 10$$
$$n_2(1-\hat{p}_2) \ge 10$$


Use the following calculator to check the above requirements:  



::: {.cell}

```{.r .cell-code}
# All must be true:
x1 <- 91
n1 <- 137
phat1 <- x1/n1

n1*phat1 >= 10
```

::: {.cell-output .cell-output-stdout}

```
[1] TRUE
```


:::

```{.r .cell-code}
n1*(1-phat1) >=10
```

::: {.cell-output .cell-output-stdout}

```
[1] TRUE
```


:::

```{.r .cell-code}
x2 <- 80
n2 <- 116
phat2 <- x2 / n2

n2*phat2 >= 10
```

::: {.cell-output .cell-output-stdout}

```
[1] TRUE
```


:::

```{.r .cell-code}
n2*(1-phat2) >=10
```

::: {.cell-output .cell-output-stdout}

```
[1] TRUE
```


:::
:::




__Question__:  Are the requirements satisfied for assuming normality?  
__Answer__:  Yes, We have enough data to assume the probability to be normal.


# Couples Retreat?

A marriage counselor conducted a study of couples, categorizing each of the couples as "communicative" or "non-communicative". Among other things, the counselor wanted to see whether the percentage of communicative couples whose marriage ended in separation or divorce was *less than* the percentage of non-communicative couples whose marriage ended in separation or divorce. 

Of the 97 couples classified as "communicative", 17 ended in separation or divorce.  Of the 56 couples classified as "non-communicative", 13 ended in separation or divorce.  

## Hypothesis Test

Construct a null and alternative hypothesis for the study:

$$H_0: pcomm = pnotcomm $$  
$$H_a:pcomm < pnotcomm  $$  

Perform the appropriate analysis:  



::: {.cell}

```{.r .cell-code}
prop.test(x = c(17,13), n = c(97,56), alternative = "less")
```

::: {.cell-output .cell-output-stdout}

```

	2-sample test for equality of proportions with continuity correction

data:  c(17, 13) out of c(97, 56)
X-squared = 0.41262, df = 1, p-value = 0.2603
alternative hypothesis: less
95 percent confidence interval:
 -1.00000000  0.06964172
sample estimates:
   prop 1    prop 2 
0.1752577 0.2321429 
```


:::
:::




__Question__:  What is the P-value?  
__Answer__:  0.2603


__Question__:  Based on your decision rule, state your conclusion in context of the research question:   
__Answer__:  The p-value is not smaller than a significance level of 0.05. Therefore, there isn't enough evidence to reject the null.


## Confidence Interval

Create a 95\% confidence interval for the difference in the proportion of divorces between communicative and non-communicative couples:



::: {.cell}

```{.r .cell-code}
prop.test(x = c(17,13), n = c(97,56), conf.level = .95)
```

::: {.cell-output .cell-output-stdout}

```

	2-sample test for equality of proportions with continuity correction

data:  c(17, 13) out of c(97, 56)
X-squared = 0.41262, df = 1, p-value = 0.5206
alternative hypothesis: two.sided
95 percent confidence interval:
 -0.20495320  0.09118295
sample estimates:
   prop 1    prop 2 
0.1752577 0.2321429 
```


:::
:::



__Question__:  Interpret the confidence interval in context of the question:  
__Answer__:  with a 95% cofidence, there is a 0 within the interval -0.20495320  0.09118295, impliying there isn't a significant difference between the proprortion of communicative and non-communicative.


### Test Requirements

Use the following calculator to check the requirements for the hypothesis test and confidence interval:  



::: {.cell}

```{.r .cell-code}
# All must be true:
x1 <- 17
n1 <- 97
phat1 <- x1/n1

n1*phat1 >= 10
```

::: {.cell-output .cell-output-stdout}

```
[1] TRUE
```


:::

```{.r .cell-code}
n1*(1-phat1) >=10
```

::: {.cell-output .cell-output-stdout}

```
[1] TRUE
```


:::

```{.r .cell-code}
x2 <- 13
n2 <- 56
phat2 <- x2 / n2

n2*phat2 >= 10
```

::: {.cell-output .cell-output-stdout}

```
[1] TRUE
```


:::

```{.r .cell-code}
n2*(1-phat2) >=10
```

::: {.cell-output .cell-output-stdout}

```
[1] TRUE
```


:::
:::





__QUESTION__:  Are the requirements for the hypothesis test and confidence interval satisfied?  
__ANSWER__:  Yes, We have enough data to assume the probability to be normal.

