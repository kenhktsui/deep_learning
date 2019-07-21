import numpy as np

def binary_loss(pred, Y):
    return (-Y * np.log(pred + 1e-8) - (1 - Y) * np.log(1 - pred + 1e-8)).mean()


def y2vec(Y, K):
    N = len(Y)
    ind = np.zeros((N, K))
    for i in range(N):
        ind[i, int(Y[i])] = 1
    return ind

def vec2y(vec):
    return np.argmax(vec, axis=1)


def cross_entropy(pred, Y):
    return (-Y * np.log(pred + 1e-10)).sum()


def error_rate(pred, Y):
    return (vec2y(pred) != vec2y(Y)).sum() / len(Y)

def data_shuffle(X, Y):
    assert X.shape[0] == Y.shape[0]
    p = np.random.permutation(len(X))
    return X[p], Y[p]

class Hidden_Layer ():
    def __init__(self, N, M, id_layer, OL=False):
        self.N = N
        self.M = M
        self.id_layer = id_layer
        self.weight = np.float32(np.random.uniform(low=-np.sqrt(12 / (self.N + self.M)), high=np.sqrt(12 / (self.N + self.M)), size=(N, M)))
        self.bias = np.float32(np.full(M, 0.01))
        self.OL = OL
    #
    def forward(self, X):
        self.z = X.dot(self.weight) + self.bias
        if self.OL:
            if self.M == 1:
                return 1 / (1 + np.exp(-self.z))
            if self.M > 1:
                exps = np.exp(self.z - np.max(self.z))
                return exps / np.sum(exps, axis=1, keepdims=True)
        return self.z * (self.z > 0)


class DNN ():
    def __init__(self, architecture):
        self.architecture = architecture
        self.container = []
        for (i, j) in enumerate(architecture):
            if i == 0:
                prev_j = j
            elif i == (len(architecture) - 1):
                hl = Hidden_Layer(prev_j, j, i, OL=True)
                self.container.append(hl)
            else:
                hl = Hidden_Layer(prev_j, j, i)
                self.container.append(hl)
                prev_j = j
    #
    def predict(self, X, OL=True):
        z_list = []
        for i in self.container:
            z = i.forward(X)
            z_list.append(z)
            X = z
        if OL is not True:
            return z_list
        return z_list[-1]
    #
    def fit(self, X, Y, epoch=10000, learning_rate=0.001, verbose=True, beta1=0.9, beta2=0.999, batch_size=1, clipping=5):
        X = np.float32(np.array(X))
        Y = np.float32(np.array(Y))
        assert X.shape[0] == Y.shape[0]        
        nround = X.shape[0]//batch_size # drop the last batch
        for i in range(1, epoch + 1):
            X, Y = data_shuffle(X,Y)
            for k in range(0,nround):
                Xs = X[k*batch_size:(k+1)*batch_size,:]
                Ys = Y[k*batch_size:(k+1)*batch_size,:]
                e = (self.predict(Xs) - Ys)
                def grad_loop(i):
                    if i == 0:
                        return e
                    else:
                        return grad_loop(i - 1).dot(self.container[-i].weight.T) * (self.container[-i - 1].z > 0)
                #
                def grad_weight(i):
                    if i < (len(self.architecture) - 2):
                        return np.matmul(self.container[-i - 2].z.T, grad_loop(i)) / batch_size
                    else:
                        return np.matmul(Xs.T, grad_loop(i)) / batch_size
                #
                def grad_bias(i):
                    return grad_loop(i).mean(axis=0)
                #   
                m_gw_t1 = []
                m_gb_t1 = []
                v_gw_t1 = []
                v_gb_t1 = []
                for j in range(len(self.architecture) - 1):
                    gw = np.clip(grad_weight(len(self.architecture) - 2 - j), -clipping, clipping)
                    gb = np.clip(grad_bias(len(self.architecture) - 2 - j), -clipping, clipping)
                    if (i == 1) & (k == 0):
                        m_gw = (1 - beta1) * gw
                        m_gb = (1 - beta1) * gb
                        v_gw = (1 - beta2) * gw ** 2
                        v_gb = (1 - beta2) * gb ** 2
                    else:
                        m_gw = beta1 * memory1[j] + (1 - beta1) * gw
                        m_gb = beta1 * memory2[j] + (1 - beta1) * gb
                        v_gw = beta2 * memory3[j] + (1 - beta2) * gw ** 2
                        v_gb = beta2 * memory4[j] + (1 - beta2) * gb ** 2
                    m_gw_t1.append(m_gw)
                    m_gb_t1.append(m_gb)
                    v_gw_t1.append(v_gw)
                    v_gb_t1.append(v_gb)
                    m_gw = m_gw / (1 - beta1**(i))
                    m_gb = m_gb / (1 - beta1**(i))
                    v_gw = v_gw / (1 - beta2**(i))
                    v_gb = v_gb / (1 - beta2**(i))
                    self.container[j].weight -= learning_rate * m_gw / (np.sqrt(v_gw) + 1e-8)
                    self.container[j].bias -= learning_rate * m_gb / (np.sqrt(v_gb) + 1e-8)
                memory1, memory2, memory3, memory4 = m_gw_t1, m_gb_t1, v_gw_t1, v_gb_t1
                if verbose:
                    if k % 1000 == 0:
                        print('epoch:', i, 'batch:', k, 'cost:', cross_entropy(self.predict(Xs), Ys), 'error rate:', error_rate(self.predict(Xs), Ys))


if __name__ == '__main__':
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    X = mnist.train.images
    Y = mnist.train.labels
    dnn = DNN([X.shape[1], 32, 32, 10])
    dnn.fit(X, Y, epoch=10, batch_size=32, learning_rate=0.001)
