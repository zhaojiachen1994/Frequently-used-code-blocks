<details>
    <summary><strong>   Evaluation for classification   </strong></summary>
    
```python
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    def classify_evaluate(y_true, y_pred):
        Acc = accuracy_score(y_true, y_pred)
        Pre = precision_score(y_true, y_pred, pos_label='positive', average='micro')
        Rec = recall_score(y_true, y_pred, pos_label='positive', average='micro')
        f1 = f1_score(y_true, y_pred, pos_label='positive', average='micro')
        evaluation = {'Acc': Acc, 'Pre': Pre, 'Rec': Rec, 'f1': f1}
        return evaluation
```

</details>

------------------------------------------------------------------------------------------------------------------------
<details>
    <summary><strong>   Evaluation for clustering   </strong></summary>
        
```python
    from sklearn.metrics.cluster import adjusted_rand_score, adjusted_mutual_info_score
    def cluster_evaluate(y_pred, y_true):
        Acc = np.mean(y_pred == y_true)
        ARI = adjusted_rand_score(y_true, y_pred)
        AMI = adjusted_mutual_info_score(y_true, y_pred)
        evaluation = {'ACC': Acc, 'ARI': ARI, 'AMI': AMI}
        return evaluation
```

</details>

------------------------------------------------------------------------------------------------------------------------
<details>
    <summary><strong>   Weighted covariance computing   </strong></summary>
    
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
</details>

------------------------------------------------------------------------------------------------------------------------
<details>
    <summary><strong>   增量式加权协方差计算   </strong></summary>
        
```python
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
            v2_t1 = v2_t0 + w_t1 ^ 2
            run_sum = Cov_t0*(v1_t0^2-v2_t0)/v1_t0 + np.outer(M_t0, M_t0)*v1_t0
            run_sum = run_sum + np.outer(x_t1, x_t1)*w_t1
            Cov_t1 = (run_sum - np.outer(M_t1, M_t1)*v1_t1)*v1_t1/(v1_t1^2- v2_t1)
        else:
            v2_t1=None
            run_sum = Cov_t0*v1_t0 + np.outer(M_t0, M_t0)*v1_t0
            run_sum = run_sum + np.outer(x_t1, x_t1)*w_t1
            Cov_t1 = (run_sum - np.outer(M_t1, M_t1)*v1_t1)/v1_t1
        return M_t1, Cov_t1, v1_t1, v2_t1
```
</details>

------------------------------------------------------------------------------------------------------------------------

<details> 
    <summary><strong>  OneClassSVM for outlier detection, AUC to evaluate the results </strong></summary>

```python
from sklearn.metrics import roc_auc_score
from sklearn.svm import OneClassSVM
def oneSVMmodel(X,y, gamma = 1e-6):
    '''
    :param X: [num_sample, num_dim]
    :param y: [num_sample, ], positive samples(minority, outliers) is labeled as 1
    :param gamma: gamma for rbf kernel used in OneClassSVM
    :return: y_pred: predicted label, outliers are labeled as 1
             y_prob: outlier scores, higher for more abnormal
             auc: Area Under the Receiver Operating Characteristic Curve (ROC AUC) from sklearn.
    '''
    clf = OneClassSVM(gamma=gamma).fit(X)
    y_pred = clf.predict(X)
    y_pred = np.where(y_pred == 1, 0, y_pred)
    y_pred = np.where(y_pred==-1, 1, y_pred)
    y_prob = clf.score_samples(X)
    y_prob = np.max(y_prob)-y_prob
    auc = roc_auc_score(y, y_prob)
    return y_pred, y_prob, auc
```
</details>

-----------------------------------------------------------------------------------------------------------------------------------

<details><summary><strong>   高斯分布某个点的概率密度函数值  </strong></summary><blockquote>
    
 ```python
    mean = np.array([0,0])
    cov = np.eye(2)
    point1 = np.array([0,0])
    point2 = np.array([1,1])
    mvnormal = multivariate_normal(mean, cov, allow_singular=True)
    score1 = -mvnormal.logpdf(point1)
    pdf1 = mvnormal.pdf(point1)
    score2 = -mvnormal.logpdf(point2)
    pdf2 = mvnormal.pdf(point2)
    print(pdf1, score1)
    print(pdf2, score2)

    # pdf1, score1: 0.15915494309189535 1.8378770664093453
    # pdf2, score2: 0.05854983152431917 2.8378770664093453
 ```
 
</blockquote></details>

<div align=left><img src ="https://github.com/zhaojiachen1994/Frequently-used-code-blocks/blob/master/Figures/gaussianpdf.png" width="200" height="120"/></div>
<div align=left><img src ="https://github.com/zhaojiachen1994/Frequently-used-code-blocks/blob/master/Figures/gaussiancdf.png" width="200" height="120"/></div>


-----------------------------------------------------------------------------------------------------------------------------------


<details><summary><strong>   异常点检测-Local outlier factor  </strong></summary><blockquote>
    
- [sklearn lof document](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html#sklearn.neighbors.LocalOutlierFactor)
    
- [sklearn source code](https://github.com/scikit-learn/scikit-learn/blob/95d4f0841/sklearn/neighbors/_lof.py#L19)
    
- [sklearn neighbor](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html#sklearn.neighbors.NearestNeighbors.kneighbors)
    
- [wiki](https://en.wikipedia.org/wiki/Local_outlier_factor)
    
 ```python
    import numpy as np
    from sklearn.neighbors import NearestNeighbors

 
    n_neighbors = 4     # 最近邻的个数，包括自身，设置为4，其实只包括三个最临近的样本
    X = np.array([[-10,0],[0,0],[1,0],[0,1], [0,-1]]) # 样本矩阵，with shape of [n_sample, n_feature]
    neigh = NearestNeighbors(n_neighbors=n_neighbors)
    neigh.fit(X)

    '''计算最近邻的距离和标签'''
    neigh_dist, neigh_ind = neigh.kneighbors(X)
    # neigh_dist [n_sample, n_neighbors], each row is the distance of the sample to its neighbors
    # neigh_ind [n_sample, n_neighbors], each row is the index of the sample's neighbors

    '''计算可达距离， reachability distance'''
    dist_k = neigh_dist[neigh_ind, n_neighbors-1]
    reach_dist_array = np.maximum(neigh_dist, dist_k)
    
    # reach_dist_array[i,j] 是第i个样本的第j个neighbor到样本i的可达距离
    '''计算局部可达密度'''
    lrd = 1. / (np.mean(reach_dist_array, axis=1) + 1e-10)
    # local_reachability_density是可达距离平均值的倒数

    '''计算 local outlier factor'''
    lrd_ratios_array = lrd[neigh_ind]/lrd[:, np.newaxis]
    lof = np.mean(lrd_ratios_array, axis=1) # outlier with larger lof

 ```
 
</blockquote></details>



-----------------------------------------------------------------------------------------------------------------------------------


