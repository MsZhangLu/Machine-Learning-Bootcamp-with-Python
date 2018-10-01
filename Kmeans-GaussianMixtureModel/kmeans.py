import numpy as np


class KMeans():

    '''
        Class KMeans:
        Attr:
            n_cluster - Number of cluster for kmeans clustering
            max_iter - maximum updates for kmeans clustering
            e - error tolerance
    '''

    def __init__(self, n_cluster, max_iter=100, e=0.0001):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e

    def fit(self, x):
        '''
            Finds n_cluster in the data x
            params:
                x - N X D numpy array
            returns:
                A tuple
                (centroids or means, membership, number_of_updates )
            Note: Number of iterations is the number of time you update means other than initialization
        '''
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
        np.random.seed(42)
        N, D = x.shape

        # TODO:
        # - comment/remove the exception.
        # - Initialize means by picking self.n_cluster from N data points
        # - Update means and membership until convergence or until you have made self.max_iter updates.
        # - return (means, membership, number_of_updates)

        # DONOT CHANGE CODE ABOVE THIS LINE
        # raise Exception(
        #     'Implement fit function in KMeans class (filename: kmeans.py')

        # initialize
        # centers = x[np.random.choice(N, self.n_cluster)]
        # index = np.random.permutation(list(range(N)))[:self.n_cluster]

        idx = np.arange(len(x))
        centers = x[np.random.choice(idx, self.n_cluster, replace=False)]

        # centers = x[index]

        J = np.iinfo(np.int32).max/N
        self.updates = self.max_iter - 1
        
        # repeat
        for i in range(self.max_iter):
            r = np.zeros((N, self.n_cluster))

            distances = []
            for center in centers:
                distances.append(np.linalg.norm(x - center, axis = 1)**2)
            distances = np.transpose(np.array(distances))

            J_new = np.sum(np.min(distances, axis = 1)) / N

            r[np.arange(N), np.argmin(distances, axis = 1)] = 1

            if np.abs(J - J_new) <= self.e:
                self.updates = i 
                break

            J = J_new

            new_centers = []

            for k in range(self.n_cluster):
                d = np.sum(r[:, k], axis=0)
                n = np.matmul(np.transpose(r[:, k]), x)
                new_centers.append(n/d)

            centers = new_centers

        membership = np.array([np.argmax(member) for member in r])

        return (np.array(centers), membership, self.updates)

        # DONOT CHANGE CODE BELOW THIS LINE


class KMeansClassifier():

    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of cluster for kmeans clustering
            max_iter - maximum updates for kmeans clustering
            e - error tolerance
    '''

    def __init__(self, n_cluster, max_iter=100, e=1e-6):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e

    def fit(self, x, y):
        '''
            Train the classifier
            params:
                x - N X D size  numpy array
                y - N size numpy array of labels
            returns:
                None
            Stores following attributes:
                self.centroids : centroids obtained by kmeans clustering
                self.centroid_labels : labels of each centroid obtained by
                    majority voting
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        np.random.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the classifier
        # - assign means to centroids
        # - assign labels to centroid_labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        # raise Exception(
        #     'Implement fit function in KMeansClassifier class (filename: kmeans.py')

        k_means = KMeans(n_cluster=self.n_cluster, max_iter=self.max_iter, e=self.e)
        centroids, membership, i = k_means.fit(x)

        vote = {k: [] for k in range(len(centroids))}

        for m, yy in zip(membership, y):
            vote[int(m)].append(yy)

        centroid_labels = []

        for v in vote:
            counts = np.bincount(np.array(vote[v]))
            centroid_labels.append(np.argmax(counts))

        centroid_labels = np.array(centroid_labels)

        # DONOT CHANGE CODE BELOW THIS LINE

        self.centroid_labels = centroid_labels
        self.centroids = centroids

        assert self.centroid_labels.shape == (self.n_cluster,), 'centroid_labels should be a vector of shape {}'.format(
            self.n_cluster)

        assert self.centroids.shape == (self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(
            self.n_cluster, D)

    def predict(self, x):
        '''
            Predict function

            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        np.random.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the prediction algorithm
        # - return labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        # raise Exception(
            # 'Implement predict function in KMeansClassifier class (filename: kmeans.py')

        predict_labels = []

        for pt in x:
            distances = [np.sum((np.array(pt) - np.array(center)) ** 2) for center in self.centroids]
            predict_labels.append(self.centroid_labels[np.argmin(distances)])

        return np.array(predict_labels)


        # DONOT CHANGE CODE BELOW THIS LINE
