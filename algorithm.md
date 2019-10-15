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

Mine:
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
