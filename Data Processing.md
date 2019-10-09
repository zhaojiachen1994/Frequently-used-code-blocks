- 利用sklearn和Seglearn来构造多维的时间序列，并用划窗分割。
'''python
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from seglearn.transform import SegmentX, SegmentXY
def demoDataset():
    X, y = make_blobs(n_features=2, n_samples=300, centers=2, shuffle=False,
                      random_state=1)
    scaler = StandardScaler()
    ts = scaler.fit_transform(X)
    width = 1
    ts = [ts]
    segment = SegmentXY(width=width, overlap=0.5)#, y_func='middle'
    X, y, _ = segment.fit_transform(ts, [y])#,[y.reshape([-1,1])]
    X = X.reshape(X.shape[0],-1)
    return X, y
X, y = demoDataset()

plt.plot(X,'.')
plt.show()    
'''
