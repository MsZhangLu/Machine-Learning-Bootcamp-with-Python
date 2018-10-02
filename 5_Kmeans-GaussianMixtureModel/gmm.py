import numpy as np
from kmeans import KMeans


class GMM():
    '''
        Fits a Gausian Mixture model to the data.

        attrs:
            n_cluster : Number of mixtures
            e : error tolerance
            max_iter : maximum number of updates
            init : initialization of means and variance
                Can be 'random' or 'kmeans'
            means : means of gaussian mixtures
            variances : variance of gaussian mixtures
            pi_k : mixture probabilities of different component
    '''

    def __init__(self, n_cluster, init='k_means', max_iter=100, e=0.0001):
        self.n_cluster = n_cluster
        self.e = e
        self.max_iter = max_iter
        self.init = init
        self.means = None
        self.variances = None
        self.pi_k = None

    def fit(self, x):
        '''
            Fits a GMM to x.

            x: is a NXD size numpy array
            updates:
                self.means
                self.variances
                self.pi_k
        '''
        assert len(x.shape) == 2, 'x can only be 2 dimensional'

        np.random.seed(42)
        N, D = x.shape

        if (self.init == 'k_means'):
            # TODO
            # - comment/remove the exception
            # - initialize means using k-means clustering
            # - compute variance and pi_k

            # DONOT MODIFY CODE ABOVE THIS LINE
            # raise Exception(
                # 'Implement initialization of variances, means, pi_k using k-means')

            kmeans = KMeans(n_cluster=self.n_cluster, max_iter=self.max_iter, e=1e-8)
            self.means, membership, i = kmeans.fit(x)

            gamma = np.zeros((N, self.n_cluster))
            tmp_matrix = np.zeros((self.n_cluster, N, D))
            for m, i in zip(membership, range(len(membership))):
                gamma[i][m] = 1 
                tmp_matrix[m][i] = x[i]

            N_k = np.sum(gamma, axis = 0)

            self.pi_k = N_k / N

            self.variances = np.zeros((self.n_cluster, D, D))
            for k in range(self.n_cluster):
                for i in range(N):
                    self.variances[k] +=  gamma[i][k] * np.dot(np.transpose([x[i] - self.means[k]]), [x[i] - self.means[k]])
                self.variances[k] = self.variances[k] / N_k[k]
            
            # DONOT MODIFY CODE BELOW THIS LINE

        elif (self.init == 'random'):
            # TODO
            # - comment/remove the exception
            # - initialize means randomly
            # - compute variance and pi_k

            # DONOT MODIFY CODE ABOVE THIS LINE
            # raise Exception(
            #     'Implement initialization of variances, means, pi_k randomly')

            self.means = np.random.random_sample((self.n_cluster, D))

            self.variances = np.array([np.identity(D)] * self.n_cluster)

            self.pi_k = np.array([1/self.n_cluster] * self.n_cluster)

            # DONOT MODIFY CODE BELOW THIS LINE

        else:
            raise Exception('Invalid initialization provided')

        # TODO
        # - comment/remove the exception
        # - find the optimal means, variances, and pi_k and assign it to self
        # - return number of updates done to reach the optimal values.
        # Hint: Try to seperate E & M step for clarity

        # DONOT MODIFY CODE ABOVE THIS LINE
        # raise Exception('Implement fit function (filename: gmm.py)')

        # compute log-likelihood 
        l = self.compute_log_likelihood(x)

        self.iterations = self.max_iter - 1

        for it in range(self.max_iter):
            # E-step: compute reponsibilities 
            gamma_num = np.zeros((N, self.n_cluster))
            for k in range(self.n_cluster):
                var = self.variances[k]
                while np.linalg.matrix_rank(var) != var.shape[0]:
                    var = var + np.identity(D) * 1e-3
                gamma_num[:, k] = self.pi_k[k] * (1/np.sqrt(np.power(2*np.pi, D) * np.linalg.det(var))) * \
                    np.exp(np.diag(-0.5 * np.matrix(x - self.means[k]).dot(np.linalg.inv(var)).dot(np.transpose(np.matrix(x - self.means[k])))))
            gamma_dom = np.sum(gamma_num, axis = 1)
            gamma = gamma_num / gamma_dom[:, None]

            # M-step: 
            N_k = np.sum(gamma, axis = 0)

            self.means = np.matmul(gamma.T, x) / N_k[:, None]

            self.variances = np.zeros((self.n_cluster, D, D))
            for k in range(self.n_cluster):
                for i in range(N):
                    self.variances[k] +=  gamma[i][k] * np.dot(np.transpose([x[i] - self.means[k]]), [x[i] - self.means[k]])
                self.variances[k] = self.variances[k] / N_k[k]

            self.pi_k = N_k / N

            l_new = self.compute_log_likelihood(x)

            if np.abs(l - l_new) <= self.e:
                self.iterations = it
                break
            l = l_new
        return self.iterations


        # DONOT MODIFY CODE BELOW THIS LINE

    def sample(self, N):
        '''
        sample from the GMM model

        N is a positive integer
        return : NXD array of samples

        '''
        assert type(N) == int and N > 0, 'N should be a positive integer'
        np.random.seed(42)
        if (self.means is None):
            raise Exception('Train GMM before sampling')

        # TODO
        # - comment/remove the exception
        # - generate samples from the GMM
        # - return the samples

        # DONOT MODIFY CODE ABOVE THIS LINE
        # raise Exception('Implement sample function in gmm.py')

        K = self.means.shape[0]
        D = self.means.shape[1]

        sample_k = np.random.choice(np.arange(K), N, p = self.pi_k)
        sample = np.zeros((N, D))
        for i, k in enumerate(sample_k):
            sample[i] = np.random.multivariate_normal(self.means[k], self.variances[k], 1)
        return sample

        # DONOT MODIFY CODE BELOW THIS LINE


    def compute_log_likelihood(self, x):
        '''
            Return log-likelihood for the data

            x is a NXD matrix
            return : a float number which is the log-likelihood of data
        '''
        assert len(x.shape) == 2,  'x can only be 2 dimensional'
        # TODO
        # - comment/remove the exception
        # - calculate log-likelihood using means, variances and pi_k attr in self
        # - return the log-likelihood
        # Note: you can call this function in fit function (if required)
        # DONOT MODIFY CODE ABOVE THIS LINE

        N, D = x.shape

        p_xk = np.zeros((N, self.n_cluster))
        for k in range(self.n_cluster):
            var = self.variances[k]
            while np.linalg.matrix_rank(var) != var.shape[0]:
                var = var + np.identity(x.shape[1]) * 1e-3
            p_xk[:, k] = self.pi_k[k] * (1/np.sqrt(np.power(2*np.pi, D) * np.linalg.det(var))) * np.exp(
                    np.diag(-0.5 * np.matrix(x - self.means[k]).dot(np.linalg.inv(var)).dot(np.transpose(np.matrix(x - self.means[k])))))

        t = np.sum(np.log(np.sum(p_xk, axis = 1)))

        return float(t)

        # DONOT MODIFY CODE BELOW THIS LINE
