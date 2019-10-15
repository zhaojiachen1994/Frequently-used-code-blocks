- Evaluation for clustering
```python
from sklearn.metrics.cluster import adjusted_rand_score, adjusted_mutual_info_score
def cluster_evaluate(y_pred, y_true):
    Acc = np.mean(y_pred == y_true)
    ARI = adjusted_rand_score(y_true, y_pred)
    AMI = adjusted_mutual_info_score(y_true, y_pred)
    evaluation = {'ACC': Acc, 'ARI': ARI, 'AMI': AMI}
    return evaluation
```

- Weighted covariance computing
```python
# MINE:
import numpy as np
def myWeightedCov(X, w, bias=False):
    """
    :param X: Sample matrix, ndarray, [num_sample, num_dim]
    :param w: weight array, ndarray, [num_sample, 1]
    :param bias: bool, 'False' for non bias covariance estimation
    :return: Weighted covariance matrix
    """
    w = w.reshape(-1, 1)
    v1 = np.sum(w)
    v2 = np.sum(w**2)
    Mean_w = np.sum(X*w, axis=0, keepdims=True)/v1
    X_m = X-Mean_w
    if bias==False:
        cov = np.dot(X_m.T, X_m*w)*v1/(v1**2-v2)
    else: cov = np.dot(X_m.T, X_m*w)/v1
    return(cov) 
    
# NUMPY:
cov = np.cov(x, bias=False, rowvar=False, aweights=w)
```

-增量式加权协方差计算
‵‵‵python
def IncWeightCov(M_t0, Cov_t0, v1_t0, v2_t0, x_t1, w_t1, bias=False):
    """
    :param M_t0: computed mean at time t0
    :param Cov_t0: computed covariance at time t0
    :param v1_t0: computed sum of wights at time t0
    :param v2_t0: sum(weight^2) at t0
    :param x_t1: arriving sample 
    :param w_t1: weight of arriveing sample
    :param bias: True for bias estimation
    :return: 
    """
    x_t1 = x_t1.reshape([1,-1])
    w_t1 = np.array([w_t1])
    v1_t1 = v1_t0 + w_t1
    M_t1 = (M_t0*v1_t0 + x_t1*w_t1)/v1_t1
    if bias==False:
        v2_t1 = v2_t0 + w_t1 \** 2
        run_sum = Cov_t0*(v1_t0\**2-v2_t0)/v1_t0 + np.outer(M_t0, M_t0)*v1_t0
        run_sum = run_sum + np.outer(x_t1, x_t1)*w_t1
        Cov_t1 = (run_sum - np.outer(M_t1, M_t1)*v1_t1)*v1_t1/(v1_t1\**2- v2_t1)
    else:
        v2_t1=None
        run_sum = Cov_t0*v1_t0 + np.outer(M_t0, M_t0)*v1_t0
        run_sum = run_sum + np.outer(x_t1, x_t1)*w_t1
        Cov_t1 = (run_sum - np.outer(M_t1, M_t1)*v1_t1)/v1_t1
    return M_t1, Cov_t1, v1_t1, v2_t1
```
