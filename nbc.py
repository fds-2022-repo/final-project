import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.sparse import csr_matrix as csr

class ClassicalNaiveBayes:
    def __init__(self, alpha=1):
        """
            alpha: laplace smoothing parameter
        """
        self.alpha = alpha
        self.phi = None
        self.prior = None
        self.classes = None
        self.u = Utils()

    def fit(self, X: csr, y):
        """
        Fits the model by calculating the prior and the conditional probabilities
            X: sparse matrix of shape [n_samples, n_features]
            y: labels of shape [n_samples]
        """
        self.classes = np.unique(y)
        self.prior = self.u.label2onehot(y).sum(axis=0) / self.u.label2onehot(y).sum()
        counts = X.T.dot(self.u.label2onehot(y)) + self.alpha
        self.phi = counts / counts.sum(axis=0)

    def fit_L2(self, X: csr, y, alpha = 0.000001, l2 = 1000, weight = False, plot = False):
        """
        Updates the model parameters by penalizing the L2 norm of the conditional probabilities
            X: sparse matrix of shape [n_samples, n_features]
            y: labels of shape [n_samples]
            alpha: learning rate
            l2: L2 regularization parameter
            weight: whether to weight the conditional probabilities
            plot: whether to plot the log likelihood
        """
        # Fit the model
        self.fit(X, y)

        print(f'Starting gradient ascent with alpha = {alpha} and l2 = {l2}')
        phi_l2, likelihood = self.u.gradient_ascent(
            X = X, y = self.u.label2onehot(y), 
            phi = self.phi, 
            alpha = alpha, 
            l2 = l2, 
            max_iter = 100, 
            tol = 1e-4)

        self.phi = phi_l2

        # Modifications
        self.fit_weighted(X, y)      if weight else None

        # Plotting for inspection
        pd.Series(likelihood).plot() if plot else None

        
    def fit_weighted(self):
        """
        Fits the model by calculating the prior and the conditional probabilities
            X: sparse matrix of shape [n_samples, n_features]
            y: labels of shape [n_samples]
        """
        if self.phi is None:
            print('No prior phi found, first fit the model')
            return
        
        weights = np.power(self.phi - self.prior, 2).sum(axis=1)
        self.phi = self.phi * weights[:, np.newaxis]


    def predict(self, X: csr) -> np.ndarray:
        """
        Predicts the class of each sample in X
            X: sparse matrix of shape [n_samples, n_features]
        """

        log_posterior = np.log(self.prior) + X.dot(np.log(self.phi))
        return np.argmax(log_posterior, axis=1)

    def score(self, X: csr, y) -> float:
        """
        Calculates the accuracy of the model
            X: sparse matrix of shape [n_samples, n_features]
            y: labels of shape [n_samples]
        """
        predicted = self.predict(X)
        return np.mean(predicted == self.u.label2num(y))


class Utils:

    @staticmethod
    def label2num(label):
        classes = np.unique(label)
        num = np.zeros(len(label))
        for i, c in enumerate(classes):
            # print(f'Class {c} is {i}th in num')
            num[label == c] = i

        return num

    @staticmethod
    def label2onehot(label):
        classes = np.unique(label)
        onehot = np.zeros((len(label), len(classes)))
        for i, c in enumerate(classes):
            # print(f'Class {c} is {i}th in onehot')
            onehot[label == c, i] = 1

        onehot
        return onehot

    @staticmethod
    def log_likelihood(X, y_onehot, phi, l2):
        """
        X: sparse feature matrix of shape (n_samples, n_features)
        y: target array of shape (n_samples,)
        phi: parameter array of shape (n_features, n_classes)
        prior: prior probability array of shape (n_classes,)
        alpha: learning rate
        l2: L2 regularization parameter
        """

        log_l = (X.dot(np.log(phi))[y_onehot.astype(bool)].sum() - l2 * np.power(phi, 2).sum()) / X.shape[0]

        return log_l

    @staticmethod
    def update_phi(X, y, phi, alpha, l2):
        """
        X: sparse feature matrix of shape (n_samples, n_features)
        y: target array of shape (n_samples,)
        phi: parameter array of shape (n_features, n_classes)
        prior: prior probability array of shape (n_classes,)
        alpha: learning rate
        l1: L1 regularization parameter
        """

        gradient = np.zeros(phi.shape)

        for i in range(phi.shape[1]):
            gradient[:,i] = X[y[:,i].astype(bool),:].sum(axis=0) / phi[:,i] - 2 * l2 * phi[:,i]

        gradient[gradient < 0] = 0

        phi = phi + alpha * gradient
        phi = phi / phi.sum(axis=0)
        
        return phi


    def gradient_ascent(self, X, y, phi, alpha=10^-6, l2=1000, max_iter=100, tol=1e-4):
        """
        X: sparse matrix of shape (n_samples, n_features)
        y: array of shape (n_samples,)
        phi: array of shape (n_features, n_classes)
        prior: array of shape (n_classes,)
        alpha: float
        l1: L1 regularization parameter
        max_iter: Maximum number of iterations
        tol: Tolerance in absolute difference of log likelihoods
        """

        log_likelihoods = []
        for i in range(max_iter):
            phi = self.update_phi(X, y, phi, alpha, l2)
            log_l = self.log_likelihood(X, y, phi, l2)
            log_likelihoods.append(log_l)

            if i > 0 and np.abs(log_l - log_likelihoods[-2]) < tol:
                break

        return phi, log_likelihoods


if __name__ == '__main__':
    # Data from Bancolombia, no null values
    bancolombia = pd.read_csv('data/bancolombia.csv')
    bancolombia.head(1)

    import pickle
    # we will import CountVectorizer to vectorize our text
    from sklearn.feature_extraction.text import CountVectorizer

    # we will import train_test_split to split our data
    from sklearn.model_selection import train_test_split

    # load stopwords
    with open('stopwords/spanish', 'rb') as f:
        spanish_stopwords = pickle.load(f)
        
    vectorizer = CountVectorizer(min_df=5, stop_words=spanish_stopwords)
    X = vectorizer.fit_transform(bancolombia['news'])

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, bancolombia['Type'], test_size=0.2, random_state=42)

    # create an instance of the model
    cnbc = ClassicalNaiveBayes()
    cnbc.fit(X_train, y_train)

    # predict
    print('For Vanilla Naive Bayes')
    print('Train accuracy: \t', cnbc.score(X_train, y_train))
    print('Test accuracy: \t', cnbc.score(X_test, y_test))

    cnbc.fit_L2(X_train, y_train, plot=True)

    # predict
    print('For L2 Regularized Naive Bayes')
    print('Train accuracy: \t', cnbc.score(X_train, y_train))
    print('Test accuracy: \t\t', cnbc.score(X_test, y_test))