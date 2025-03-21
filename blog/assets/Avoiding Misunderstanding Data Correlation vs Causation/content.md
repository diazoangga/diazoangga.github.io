# Avoiding Misunderstanding Data: Correlation vs Causation

Published Date: March 18, 2025
Tags: Data Analytics, Featured

The quote from Richard Hamming (1962), *“The purpose of computing is insight, not numbers,”* describes the importance of understanding what the data represents and the story behind the numbers. In his book, he stated that it is not about how the data is read and used, instead, he advised to understand when the results are obtained — meaning that the data should be inbound with the source of problems to be addressed. A clear goal, of course, is a must to have a clear framework in order to acquire insights in data analytics. Another common mistake that people tend to have is confusing correlation and causation.

Correlation and causation are the metrics that data analysts or decision-makers use to make decisions. Correlation, by definition, describes the degree to which two variables move in relation to each other. For example, if you have a dataset between daylight period and temperature, we can conclude, intuitively, that daylight period and temperature are positively correlated. On the contrary, causation indicates that one event is the result of the occurrence of the other event. One important thing to remember is that just because two data are correlated to each other, does not mean the two data have causal relationship. One of the famous research by Prinz (2020) highlights the danger of misinterpreting correlation as causation. He discussed evaluating a report about the positive correlation between chocolate consumption and Nobel laureates per capita (as claimed by Messerli, 2012). It stated that even though there was a positive correlation between those variables, no mechanism was found between them. Prinz concludes that either there are hidden variables that are correlated with them or the correlations are accidental or coincidental. This case is a perfect example of thinking fallacy: Cum Hoc Ergo Propter Hoc (with this, therefore because of this) and Post Hoc Ergo Propter Hoc (after this, therefore because of this). It can be falsely assumed that eating more chocolate improves cognitive abilities, leading to Novel-worthy achievements easily.

This leads to answer broader questions, what the tasks of data science are. In principle, data science has 3 tasks: defining the data, predicting what would happen, and what would happen if there is intervened data. To put into broader contexts, in the real-world, let’s say that we have an imaginary causal model on how X, Y, and Z interact. And assume for X, Y and Z, there are probability distribution on how X, Y, Z correlated. That’s how we have the data we have in real world. However, most of the time we only know the data without knowing the probability distribution and the causal models. That’s where the task of data scientist live. Extracting the probability distribution from data is the description prediction task. and getting the structural causal model is the causal inference task.

## Measuring Correlations

The most common way to measure correlation is using Pearson’s correlation coefficient, which has been normalized ranging from -1 to +1, with this equation:

$$
r = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n} (x_i - \bar{x})^2 \sum_{i=1}^{n} (y_i - \bar{y})^2}}
$$

Where:

- r = correlation coefficient
- x_i and y_i = individual sample points
- \bar{x} and \bar{y} = sample means
- n = sample size

The correlation coefficient indicates:

- +1: Perfect positive correlation
- 0: No correlation
- -1: Perfect negative correlation

Here are visual representations of different types of correlations:

## Causal Analysis

The claim of the relation between two correlated variables does not mean they have causal relationships is not 100% true. Reichenbach published his Common Cause Principle, saying that if two variables — A and B — are dependent, then either A causes B, B causes A, or something else causes both A and B; meaning that if two variables are correlated, the variables might have direct causal relationships, or if there aren’t, some other variables causes both variables, this is called as confounding variables. Causal analysis can be approached from two methods: Causal Inference and Causal Discovery. 

The very practical questions on causal inference are “did the treatment directly help those who took it?”, “was it the marketing campaign that lead to increased sales this month or the holiday?”, “how big of an effect would increasing wages have on productivity?”. These questions root the same intuition, wanting to know the effect of a treatment or intervention on an outcome.  There are two main frameworks for causal estimation: Potential outcomes and structural causal models.

### Potential Outcomes

One of the methods of experiment methods is Randomized Controlled Trials (RCTs). Let’s do some experiments: say that we have a group of people who got sick. Let’s divide them equally and the first group, called Treatment (T or T=1), was given a medicine and the second group, called Control (C or T=0), was not given a medicine. We can check that 

Causal Discovery examines the data and deducing how variables are causally linked to each other.