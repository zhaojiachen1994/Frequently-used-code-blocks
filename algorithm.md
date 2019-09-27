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
