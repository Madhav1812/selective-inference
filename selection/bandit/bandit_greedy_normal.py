import numpy as np
_cov_cache = {}

def _design(n, p, rho, equicorrelated):
    """
    Create an equicorrelated or AR(1) design.
    """
    if equicorrelated:
        X = (np.sqrt(1 - rho) * np.random.standard_normal((n, p)) +
             np.sqrt(rho) * np.random.standard_normal(n)[:, None])
        def equi(rho, p):
            if ('equi', p, rho) not in _cov_cache:
                sigmaX = (1 - rho) * np.identity(p) + rho * np.ones((p, p))
                cholX = np.linalg.cholesky(sigmaX)
                _cov_cache[('equi', p, rho)] = sigmaX, cholX
            return _cov_cache[('equi', p, rho)]
        sigmaX, cholX = equi(rho=rho, p=p)
    else:
        def AR1(rho, p):
            if ('AR1', p, rho) not in _cov_cache:
                idx = np.arange(p)
                cov = rho ** np.abs(np.subtract.outer(idx, idx))
                _cov_cache[('AR1', p, rho)] = cov, np.linalg.cholesky(cov)
            cov, chol = _cov_cache[('AR1', p, rho)]
            return cov, chol
        sigmaX, cholX = AR1(rho=rho, p=p)
        X = np.random.standard_normal((n, p)).dot(cholX.T)
    return X, sigmaX, cholX

def generate_data(beta,
                   X,
                   K_ch,
                   n,
                   sigma=1.):
    dimX = X.ndim
    if dimX == 2:
        _, d = X.shape
    elif dimX==1:
        d= X.shape[0]
    else:
        raise ValueError("invalid entry for X")

    p = beta.shape[0]
    if p-d>0:
        D = np.zeros((n, p-d))
        D[:, K_ch] = 1

    if dimX == 2:
        A = np.hstack((X, D))
    else:
        A = np.append(X, D)

    Y = (A.dot(beta)) + (np.random.standard_normal(n)) * sigma

    return A, Y, beta * sigma, sigma

def bandit(A_0, Y_0, d, beta, sigma=1., randomizer_scale=0.5):

    p = beta.shape[0]
    K = p - d

    contexts = np.zeros((p, K))
    for j in range(K):
        e = np.zeros(K)
        e[j] = 1
        x = _design(1, d, rho=0., equicorrelated= False)[:1]
        contexts[:, j] = np.append(x, e)

    cov_0 = np.linalg.inv(A_0.T.dot(A_0))
    eigen_w, V = np.linalg.eig(cov_0)
    target_0 = cov_0.dot(A_0.T.dot(Y_0))
    B = sigma * V.T.dot(np.sqrt(np.diag(eigen_w)).dot(V))

    w=np.random.multivariate_normal(np.zeros(beta.shape[0]),randomizer_scale*sigma*sigma*np.identity(beta.shape[0]),1)
    reg = contexts.T.dot((target_0 + w.dot(B.T)).reshape(beta.shape[0]))
    ch = np.argmax(reg)
    ch_bool = np.zeros(K, np.bool)
    ch_bool[ch] = 1
    nch = np.array([z for z in range(K) if ch_bool[z] == 0])

    target_linear = np.zeros((K - 1, beta.shape[0]))
    for t in range(K - 1):
        target_linear[t, :] = contexts[:, nch[t]] - contexts[:, ch]
    #opt_linear = target_linear.dot(B)
    #print("check constraints ", (target_linear.dot(target_0 + B.dot(w))<=0.).sum(), K-1)

    a_1, y_1, _, _ = generate_data(beta,
                                   X = (contexts[:, ch])[:d],
                                   K_ch=ch,
                                   n=1,
                                   sigma=sigma)

    A_1 = np.vstack((A_0, a_1))
    Y_1 = np.append(Y_0, y_1)

    cov_1 = np.linalg.inv(A_1.T.dot(A_1))
    target_1 = cov_1.dot(A_1.T.dot(Y_1))
    sd = sigma * np.sqrt(np.diag(cov_1))
    lower = target_1 - (1.65 * sd)
    upper = target_1 + (1.65 * sd)

    return ((lower < beta)*(beta < upper)), A_1, Y_1, (target_1-beta)

def test(beta = np.array([0., 0., -10., -10., 10.]), rho=0., equicorrelated=False, T=1000, sigma=0.1):

    X_0, _ = _design(n=2, p=2, rho=rho, equicorrelated=equicorrelated)[:2]
    D_0, y_0, _, _ = generate_data(beta,
                                   X_0,
                                   K_ch=0,
                                   n=2,
                                   sigma=sigma)

    X_1, _ = _design(n=2, p=2, rho=rho, equicorrelated=equicorrelated)[:2]
    D_1, y_1, _, _ = generate_data(beta,
                                   X_1,
                                   K_ch=1,
                                   n=2,
                                   sigma=sigma)

    X_2, _ = _design(n=2, p=2, rho=rho, equicorrelated=equicorrelated)[:2]
    D_2, y_2, _, _ = generate_data(beta,
                                   X_2,
                                   K_ch=2,
                                   n=2,
                                   sigma=sigma)

    A_0 = np.vstack((np.vstack((D_0, D_1)), D_2))
    Y_0 = np.append(np.append(y_0, y_1), y_2)

    cov = np.zeros((T, 5))
    bias = np.zeros((T, 5))
    for t in range(T):
        cov[t,:], A_1, Y_1, bias[t, :] = bandit(A_0, Y_0, d=2, beta=beta, sigma=sigma)
        A_0 = A_1
        Y_0 = Y_1

    return cov, bias

T = 500
cov = np.zeros((T,5))
bias = np.zeros((T, 5))
for i in range(300):
    result = test(T=T)
    cov += result[0]
    bias += result[1]
    print("iteration comp ", i + 1)

print("coverage so far ", i + 1, cov/float(i+1), bias/float(i+1))