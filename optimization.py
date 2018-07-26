import numpy as np

from sklearn.datasets import make_circles, make_moons


def get_spiral():
    radius = np.linspace(1, 10, 100)
    thetas = np.empty((6, 100))
    for i in range(6):
        start_angle = np.pi * i / 3.0
        end_angle = start_angle + np.pi / 2
        points = np.linspace(start_angle, end_angle, 100)
        thetas[i] = points
    x1 = np.empty((6, 100))
    x2 = np.empty((6, 100))
    for i in range(6):
        x1[i] = radius * np.cos(thetas[i])
        x2[i] = radius * np.sin(thetas[i])
    X = np.empty((600, 2))
    X[:, 0] = x1.flatten()
    X[:, 1] = x2.flatten()
    X += np.random.rand(600, 2) * 0.5
    Y = np.array([0] * 100 + [1] * 100 + [0] * 100 + [1] * 100 + [0] * 100 + [1] * 100)
    return X, Y


def binary_loss(pred, Y):
    return (-Y * np.log(pred + 1e-8) - (1 - Y) * np.log(1 - pred + 1e-8)).mean()


def error_rate(pred, Y):
    return ((pred > 0.5) != Y).sum() / len(Y)


class Hidden_Layer ():
    def __init__(self, N, M, id_layer, OL=False):
        self.N = N
        self.M = M
        self.id_layer = id_layer
        self.weight = np.float32(np.random.uniform(low=-np.sqrt(12 / (self.N + self.M)), high=np.sqrt(12 / (self.N + self.M)), size=(N, M)))
        self.bias = np.float32(np.full(M, 0.01))
        self.OL = OL

    def forward(self, X):
        self.z = X.dot(self.weight) + self.bias
        if self.OL:
            return 1 / (1 + np.exp(-self.z))
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

    def predict(self, X, OL=True):
        z_list = []
        for i in self.container:
            z = i.forward(X)
            z_list.append(z)
            X = z
        if OL is not True:
            return z_list
        return z_list[-1]

    def fit(self, X, Y, epoch=10000, learning_rate=0.001, verbose=True, beta1=0.9, beta2=0.999, batch_size=1):
        X = np.float32(np.array(X))
        Y = np.float32(np.array(Y))
        Y.shape = (Y.shape[0], 1)
        sample = np.random.randint(0, len(X), batch_size)
        X = X[sample]
        Y = Y[sample]
        for i in range(1, epoch + 1):
            e = (self.predict(X) - Y)

            def grad_loop(i):
                if i == 0:
                    return e
                else:
                    return grad_loop(i - 1).dot(self.container[-i].weight.T) * (self.container[-i - 1].z > 0)

            def grad_weight(i):
                if i < (len(self.architecture) - 2):
                    return np.matmul(self.container[-i - 2].z.T, grad_loop(i)) / batch_size
                else:
                    return np.matmul(X.T, grad_loop(i)) / batch_size

            def grad_bias(i):
                return grad_loop(i).mean(axis=0)
            m_gw_t1 = []
            m_gb_t1 = []
            v_gw_t1 = []
            v_gb_t1 = []
            for j in range(len(self.architecture) - 1):
                gw = grad_weight(len(self.architecture) - 2 - j)
                gb = grad_bias(len(self.architecture) - 2 - j)
                if i == 1:
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
                if i % 100 == 0:
                    print('epoch:', i, 'cost:', binary_loss(self.predict(X), Y), 'error rate:', error_rate(self.predict(X), Y))


if __name__ == '__main__':
    # X, Y = make_circles(10000)
    #X, Y = make_moons(10000)
    X, Y = get_spiral()
    dnn = DNN([2, 32, 32, 32, 32, 32, 32, 1])
    dnn.fit(X, Y, epoch=10000, batch_size=32, learning_rate=0.001)
