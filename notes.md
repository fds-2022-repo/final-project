# Naive Bayes Classifiers

## Vanilla NBC

The multinomial naive Bayes classifier is suitable for classification with discrete features (e.g., word counts for text classification). The multinomial distribution normally requires integer feature counts. The conditional probability of a feature $x_j$ given a class $C_k$ is given by:

$$
P(x_j=1|C_k) = \frac{N_{jk} + \alpha}{N_k + \alpha d} = \phi_{jk}
$$

$$
P(x_j=l|C_k) = (\phi_{jk})^l
$$

where $N_k$ is the total number of features for class $C_k$, $N_{jk}$ is the number of occurrences of feature $x_j$ for class $C_k$, $\alpha$ is the smoothing parameter, and $d$ is the number of features/words. We then predict the class $C_k$ with the highest probability:

$$
\hat{y_i} = \{\underset{k}{\operatorname{argmax}} \ P(C_k) \prod_{j=1}^{D} P(x_{i,j}|C_k)\}
$$

where $D$ is the number of features/words.

## Weighted NBC

The weighted naive Bayes classifier is an extension of the vanilla NBC that weights the words by their importance. Weight $w_j$ is given by:

$$
w_j = \sqrt{\sum_{k=1}^K\left(P(y = k) - P(y = k|x_j = 1)\right)^2}
$$

where $K$ is the number of classes. The the prediction is given by:

$$
\hat{y_i} = \{\underset{k}{\operatorname{argmax}} \ P(C_k) \prod_{j=1}^{D_i} \phi_{jk}^{w_j}\}
$$

## NBC with L2 Regularization

The L2 regularized naive Bayes classifier is another extension of the vanilla NBC that adds a penalty term to the likelihood function. The likelihood function is given by:

$$
L(\Phi) = \prod_{i=1}^d P(C_k)\prod_{j=1}^{D_i} \phi_{jk}^{x_{i,j}}  - \lambda \sum_{j=1}^{D} \sum_{k=1}^K \phi_{jk}^2
$$

The gradient of the log-likelihood function is given by:

$$
\frac{\partial l(\Phi)}{\partial \phi_{jk}} = \sum_{i=1}^d \frac{x_{i,j}}{\phi_{jk}} - 2\lambda \phi_{jk}
$$

where $\theta_{kj}$ is the weight of feature $x_j$ for class $C_k$, and $\lambda$ is the regularization parameter. The algorithm is given by:

1. Initialize $\theta_{kj} = 0$ for all $k$ and $j$.