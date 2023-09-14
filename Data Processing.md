<details>
    <summary><strong>利用sklearn和Seglearn来构造多维的时间序列，并用划窗分割。</strong></summary>
    
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
</details>

<div align=center><img src ="https://github.com/zhaojiachen1994/Frequently-used-code-blocks/blob/master/Figures/seglearn.png"/></div>

------------------------------------------------------------------------------------------------------------------------

<details>
    <summary><strong>利用numpy和random生成三角函数曲线，并添加高斯噪声</strong></summary>
    
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
</details>

<div align=left><img src ="https://github.com/zhaojiachen1994/Frequently-used-code-blocks/blob/master/Figures/sin.png" width="300" height="150"/></div>

------------------------------------------------------------------------------------------------------------------------

<details>
    <summary><strong>利用numpy和torch生成用于测试LSTM的加法测试数据</strong></summary>

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
</details>

------------------------------------------------------------------------------------------------------------------------

<details>
    <summary><strong>Transfer time series prediction task to supervised learning task</strong></summary>

```python
def series_to_supervised(data, n_in=1, n_out=1, interval=1, dropnan=True):
    '''
    :param data: time series data with shape of [sequence_length, num_features]
    :param n_in: length of past time series
    :param n_out: length of predict time series
    :param interval: interval between two samples, 1 or n_in
    :param dropnan:
    :return:
    '''
    n_vars = 1 if type(data) is list else data.shape[1]
    dff = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(dff.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(dff.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN valuesprint(agg)
    if dropnan:
        agg.dropna(inplace=True)
    agg = agg[agg.index%interval==0]

    values = agg.values
    X, y = values[:, :-n_vars*n_out], values[:, -n_vars*n_out:]
    X = X.reshape(-1, n_out, n_vars)
    y = y.reshape(-1, n_out, n_vars)
    return agg, (X, y)
def check_series_to_supervised():
    setup_seed(1)
    # data = np.random.rand(100,2)
    data = np.linspace(1,200,200).reshape([100,2])
    print(data[:10, :])
    agg, (X, y) = series_to_supervised(data, n_in=5, n_out=5, interval=1, dropnan=True)
    print(X[0:2, :, :])
    print(y[0:2, :, :])
```
</details>

------------------------------------------------------------------------------------------------------------------------

<details>
    <summary><strong> 1D autoregressive time series </strong></summary>

    - Basic time series is generated by a 1D autoregressive model
<a href="https://www.codecogs.com/eqnedit.php?latex=y(t)&space;=&space;0.6y(t-1)-0.5y(t-2)&plus;\epsilon_t" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y(t)&space;=&space;0.6y(t-1)-0.5y(t-2)&plus;\epsilon_t" title="y(t) = 0.6y(t-1)-0.5y(t-2)+\epsilon_t" /></a>, where <a href="https://www.codecogs.com/eqnedit.php?latex=\epsilon_t" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\epsilon_t" title="\epsilon_t" /></a> isa Gaussian noise.

**Change points**

a change point is inserted at every **length_ts/num_seg + <a href="https://www.codecogs.com/eqnedit.php?latex=\tau" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\tau" title="\tau" /></a>** time steps with jumping-mean or scaling-variance, where <a href="https://www.codecogs.com/eqnedit.php?latex=\tau" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\tau" title="\tau" /></a> is noise for change point location.

**Jumping-mean:** 

<center><a href="https://www.codecogs.com/eqnedit.php?latex=\mu_t=&space;\left\{\begin{matrix}&space;0&space;&&space;n=1&space;\\&space;n/5&space;&&space;n>1&space;\end{matrix}\right." target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mu_t=&space;\left\{\begin{matrix}&space;0&space;&&space;n=1&space;\\&space;n/5&space;&&space;n>1&space;\end{matrix}\right." title="\mu_t= \left\{\begin{matrix} 0 & n=1 \\ n/5 & n>1 \end{matrix}\right." /></a></center>

**Scaling-variance:**

<a href="https://www.codecogs.com/eqnedit.php?latex=\sigma_n=\left\{\begin{matrix}&space;\alpha&space;&&space;n=0,2,...&space;\\&space;ln(e&plus;2n)&space;&&space;n=1,3...&space;\end{matrix}\right." target="_blank"><img src="https://latex.codecogs.com/gif.latex?\sigma_n=\left\{\begin{matrix}&space;\alpha&space;&&space;n=0,2,...&space;\\&space;ln(e&plus;2n)&space;&&space;n=1,3...&space;\end{matrix}\right." title="\sigma_n=\left\{\begin{matrix} \alpha & n=0,2,... \\ ln(e+2n) & n=1,3... \end{matrix}\right." /></a>

```python
def buildDataAR(shiftmean=True, shiftvar=True, verbose=True):
    '''
    :param shiftmean: flag for mean shift
    :param shiftvar: flag for variance shift
    :param verbose:
    :return:    ts: array, 1d time series with length of 5000(length_ts)
                bkps: array, change points including head and end
    '''

    # set parameters
    seed = 0    # random seed
    length_ts = 5000    # length of time series
    num_seg = 10    # number of segments
    alpha = 0.1     # radio of variance

    # generate the change points
    np.random.seed(seed)
    bkps = np.linspace(0,length_ts, num_seg+1, endpoint=True, dtype=int)    # including head and end
    bkps = bkps + (np.random.normal(loc=0,scale=10, size=num_seg+1).astype(int))
    bkps[0], bkps[-1] = 0, length_ts
    # set mean and variance
    mu_segs = np.zeros(num_seg)
    sigma_segs = np.ones(num_seg)*alpha
    if shiftmean==True:
        mu_segs = np.array([0 if i==0 else i/5 for i in range(num_seg)])
    if shiftvar==True:
        sigma_segs = np.array([alpha if i%2 == 0 else np.log(np.e + 2*i)*alpha for i in range(num_seg)])
    # generate the time series
    ts = np.zeros(length_ts)
    for i in range(num_seg):
        if verbose == True:
            print('Segment-{} [{:4d}, {:4d}] with Mean {:0.4f} Var {:0.4f}'.format(i+1, bkps[i], bkps[i+1], mu_segs[i], sigma_segs[i]))
        for j in range(bkps[i], bkps[i+1]):
            if j > 2:
                ts[j] = 0.6*ts[j-1] - 0.5*ts[j-2] + np.random.normal(mu_segs[i], sigma_segs[i],1)
    ts = np.array(ts)
    return ts, bkps

```
</details>

------------------------------------------------------------------------------------------------------------------------

<details> 
    <summary><strong>  Extract the same classes from two sets sharing some classes    </strong></summary>

```python
def extractsharedclass(X_src, y_src, X_tar, y_tar):
    '''
    description: extract the same classes from two sets sharing some classes.
    '''
    shareLabels = set(y_src.flatten()) & set(y_tar.flatten())
    print(shareLabels)
    for i, j in enumerate(shareLabels):
        if i == 0:
            ind_src, ind_tar = y_src == j, y_tar == j
        else:
            ind_src, ind_tar = (y_src == j) + ind_src, (y_tar == j) + ind_tar
    print(ind_src.shape)
    X_src, y_src = X_src[ind_src.flatten()], y_src[ind_src]
    X_tar, y_tar = X_tar[ind_tar.flatten()], y_tar[ind_tar]
    return (X_src, y_src, X_tar, y_tar)
    
# X_src = np.linspace(1,18,18).reshape([-1,2])
# X_tar = np.linspace(1,18,18).reshape([-1,2])
# y_src = np.array([1,1,1,3,3,3,5,5,5])
# y_tar = np.array([1,1,2,2,2,2,5,5,5])
# print(X_src)
# print(y_src)
```
</details>

------------------------------------------------------------------------------------------------------------------------

<details> 
    <summary><strong>  seglearn.SegmentXY自定义Y的提取方法 eg:max   </strong></summary>

```python
def foldts_XY(x, y, length_win, tensorlize=False):
    '''
    :param x: [length_ts, num_dim]
    :param length_win:
    :return: X with shape of [batch_size, length_win, num_dim]
             y with shape of [batch_size,]
    '''
    segment = SegmentXY(width=length_win, overlap=0, y_func=lambda x: np.max(x,axis=1))

    output = segment.fit_transform([x], [y])
    X = output[0]#.reshape([-1, length_win])
    y = output[1]

    if tensorlize==True:
        X = torch.from_numpy(X).float()
        y = torch.from_numpy(y.copy()).float()

    return X, y
```
</details>

------------------------------------------------------------------------------------------------------------------------

<details> 
    <summary><strong>  负指数曲线的变化趋势  </strong></summary>

```python
    t = np.linspace(0,1,100)
    fig, ax = plt.subplots()
    for gamma in [20,5,2,1,0.5,0.25,0.1]:
        ts = np.exp(-gamma*t)
        line, = ax.plot(t, ts, label='gamma={:.2f}'.format(gamma))
    ax.legend(loc='upper right')
    plt.grid(True)
    plt.xlabel('t')
    plt.ylabel('exp(-gamma*t)')
    plt.show()
```
</details>


<div align=center><img src ="https://github.com/zhaojiachen1994/Frequently-used-code-blocks/blob/master/Figures/expcurve.png"/></div>

------------------------------------------------------------------------------------------------------------------------

<details> 
    <summary><strong>   保存实验组的结果到csv文件   </strong></summary>

```python
  def addResulttoCSV(df, file):
    '''
    :df: the result dictionary or dataframe
    :file: the target file name to save the results
    '''
    df = pd.DataFrame(df)
    try:
        with open(file, 'a+') as f:# if no file, then create it
            try:
                pd.read_csv(file)   # if file is not empty, then append it without header
                df.to_csv(f, header=False,  index=False)
            except:
                df.to_csv(f, index=False)   # if file is empty, then add first line with header
    except:
        print('!!!Cannot open {}!!!'.format(file))
```
</details>

------------------------------------------------------------------------------------------------------------------------

<details>
<summary><strong>   My code to read csv </strong></summary>

 ```python
def my_pd_describe(df):
np.set_printoptions(precision=3, suppress=True)
print(f'column names:{list(df.columns)}')
Categorical_features = {}
Numeric_features = {}
for column_name in list(df.columns):
    if df[column_name].dtypes == object:
        Categorical_features[column_name] = {'Dtypes': df[column_name].dtypes,
                                             'NAN': df[column_name].isna().sum(),
                                             'Categories': df[column_name].unique(),
                                             'Number of Instances': ''.join([f'{a}:{np.sum(df[column_name]==a)} ({np.mean(df[column_name]==a):0.2f}) | ' for a in df[column_name].unique()]),
                                             }
        # Categorical_features[column_name]={'mean':df[column_name].mean(), 'std':df[column_name].std()}
    elif df[column_name].dtypes != object:
        Numeric_features[column_name]={'dtype':df[column_name].dtypes,
                                       'mean':df[column_name].mean(),
                                       'std': df[column_name].std(), 
                                       'max': df[column_name].max(),
                                       'min': df[column_name].min(),
                                       'Unique': len(df[column_name].unique()),
                                       'NAN': df[column_name].isna().sum()}

print(f'Numeric features:')
print(tabulate(pd.DataFrame(Numeric_features).T, headers='keys', tablefmt='psql', showindex="always"))
print('Categorical_features:')
print(tabulate(pd.DataFrame(Categorical_features).T, headers='keys', tablefmt='psql', showindex="always"))
 ```

</details>

-----------------------------------------------------------------------------------------------------------------------------------

<details>
<summary><strong> 一个list的元素在另一个list中的位置 </strong></summary>

 ```python
    def find_positions(list1, list2):
        positions = []
        for item in list1:
            if item in list2:
                positions.append(list2.index(item))
            else:
                positions.append(None)
        return positions

    positions = [list2.index(item) if item in list2 else None for item in list1]

    # 示例使用
    list1 = [1, 2, 3, 4, 5]
    list2 = [5, 4, 3, 2, 1]
    result = find_positions(list1, list2)
    print(result)
 ```

</details>

