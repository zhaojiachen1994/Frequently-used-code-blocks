- 利用sklearn和Seglearn来构造多维的时间序列，并用划窗分割。

```python
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
X, y = demoDataset() # shape of X is [num_samples, n_features*width]

plt.plot(X,'.')
plt.show()    
```
![image](https://github.com/zhaojiachen1994/Frequently-used-code-blocks/blob/master/Figures/seglearn.png)

- 利用numpy和random生成三角函数曲线，并添加高斯噪声。
```python
# 在0-2*pi的区间上生成100个点作为输入数据
length = 2000
X = np.linspace(0,10*np.pi,length,endpoint=True)
Y = np.sin(X) + np.cos(3*X)
mu = 0
sigma = 0.1
noise = np.random.normal(mu, sigma, 2000)
X = X+noise
Y = Y+noise
```
![image](https://github.com/zhaojiachen1994/Frequently-used-code-blocks/blob/master/Figures/sin.png)

