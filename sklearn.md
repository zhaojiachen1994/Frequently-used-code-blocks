<details>
<summary><strong>   Generate the classification task  </strong></summary>
  
[Ref1](https://scikit-learn.org/stable/auto_examples/datasets/plot_random_dataset.html#sphx-glr-auto-examples-datasets-plot-random-dataset-py)    
[Ref2](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html)
  
```python
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X, y = make_classification(n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1,n_samples=1000)
rng = np.random.RandomState(0)
X += 0*rng.uniform(size=X.shape)
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
plt.scatter(X[:,0],X[:,1],c=y)
plt.show()
```

</details>

-----------------------------------------------------------------------------------------------------------------------------------

<details>
<summary><strong>   precision, recall, f1-score for outlier detection  </strong></summary>

[Fig from wiki](https://en.wikipedia.org/wiki/Precision_and_recall)

- Recall 描述了outlier是否找全 Precision 描述了outlier是否找对 f1-score 是两者的调和平均数 

 ```python
  y_true = np.array([1,1,1,0,0,0,0,0,0,0])
    y_pred_type1 = np.array([1,1,1,1,0,0,0,0,0,0])
    y_pred_type2 = np.array([1,1,0,0,0,0,0,0,0,0])

    print('-' * 90)
    print(f'Type I error, a 0 is predicted as 1, a normal as outlier')
    Recall_default = recall_score(y_true, y_pred_type1)
    Precision_default = precision_score(y_true=y_true, y_pred=y_pred_type1)
    TPR_type1 = recall_score(y_true=y_true, y_pred=y_pred_type1, average=None, labels=[1])[0]
    PPV_type1 = precision_score(y_true=y_true, y_pred=y_pred_type1, average=None, labels=[1])[0]
    f1_type1 = f1_score(y_true=y_true, y_pred=y_pred_type1, average=None, labels=[1])[0]
    f1_default = f1_score(y_true=y_true, y_pred=y_pred_type1)
    print(f'Recall={TPR_type1:0.4f}(default:{Recall_default:0.4f})\t'
          f'Precision={PPV_type1:0.4f}(default:{Precision_default:0.4f})\t'
          f'f1={f1_type1:0.4f}(default:{f1_default:0.4f})')
    
    print('-'*90)
    print(f'Type II error, a 1 is predicted as 0, a outlier as anomaly')
    Recall_default = recall_score(y_true, y_pred_type2)
    Precision_default = precision_score(y_true=y_true, y_pred=y_pred_type2)
    TPR_type2 = recall_score(y_true=y_true, y_pred=y_pred_type2, average=None, labels=[1])[0]
    PPV_type2 = precision_score(y_true=y_true, y_pred=y_pred_type2, average=None, labels=[1])[0]
    f1_type2 = f1_score(y_true=y_true, y_pred=y_pred_type2, average=None, labels=[1])[0]
    f1_default = f1_score(y_true=y_true, y_pred=y_pred_type2)
    print(f'Recall={TPR_type2:0.4f}(default:{Recall_default:0.4f})\t'
          f'Precision={PPV_type2:0.4f}(default:{Precision_default:0.4f})\t'
          f'f1={f1_type2:0.4f}(default:{f1_default:0.4f})')
          
          
Results:
------------------------------------------------------------------------------------------
Type I error, a 0 is predicted as 1, a normal as outlier
Recall=1.0000(default:1.0000)	Precision=0.7500(default:0.7500)	f1=0.8571(default:0.8571)
------------------------------------------------------------------------------------------
Type II error, a 1 is predicted as 0, a outlier as anomaly
Recall=0.6667(default:0.6667)	Precision=1.0000(default:1.0000)	f1=0.8000(default:0.8000)
 ```

</details>

<div align=left><img src ="https://github.com/zhaojiachen1994/Frequently-used-code-blocks/blob/master/Figures/Precisionrecall.png" width="200" height="300"/></div>

-----------------------------------------------------------------------------------------------------------------------------------
