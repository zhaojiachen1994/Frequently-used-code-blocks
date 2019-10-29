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

- 利用numpy和torch生成用于测试LSTM的加法测试数据。
```python
import torch
import numpy as np
import argparse
from time import time


parser = argparse.ArgumentParser(description='PyTorch IndRNN Addition test')
parser.add_argument('--time-steps', type=int, default=4,
                    help='length of addition problem (default: 100)')
parser.add_argument('--batch-size', type=int, default=3,
                    help='input batch size for training (default: 50)')

args = parser.parse_args()

def get_batch():
    """Generate the adding problem dataset"""
    # Build the first sequence
    add_values = torch.rand(
        args.time_steps, args.batch_size, requires_grad=False
    )

    # Build the second sequence with one 1 in each half and 0s otherwise
    add_indices = torch.zeros_like(add_values)
    half = int(args.time_steps / 2)
    for i in range(args.batch_size):
        first_half = np.random.randint(half)
        second_half = np.random.randint(half, args.time_steps)
        add_indices[first_half, i] = 1
        add_indices[second_half, i] = 1

    # Zip the values and indices in a third dimension:
    # inputs has the shape (time_steps, batch_size, 2)
    inputs = torch.stack((add_values, add_indices), dim=-1)
    targets = torch.mul(add_values, add_indices).sum(dim=0)
    return inputs, targets
if __name__ == "__main__":
    inputs, targets = get_batch()
    print('Input:')
    print(inputs)
    print('Targets:')
    print(targets)
```

生成的数据如下：
```
Input:
tensor([[[0.9717, 0.0000],
         [0.2204, 1.0000],
         [0.6764, 1.0000]],

        [[0.1681, 1.0000],
         [0.1470, 0.0000],
         [0.8341, 0.0000]],

        [[0.3317, 1.0000],
         [0.8175, 1.0000],
         [0.1524, 0.0000]],

        [[0.6449, 0.0000],
         [0.3645, 0.0000],
         [0.5261, 1.0000]]])
Targets:
tensor([0.4998, 1.0379, 1.2025])

```
Inputs with shape (time_steps, batch_size, num_dim=2), The first column is the add values that prepares to be added; The second column is the add indices that indicate which value to be added.
Targets with shape (batch_size), The sum of add values.


