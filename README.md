# Frequently-used-code-blocks
Some frequently used code blocks.
---
- Pytorch
  - Device configuration
  - 统计网络的参数数量
  - 查看网络参数值
  - LSTM的输入输出形状
  - LSTMCell 的输入输出
  - Pytorch 如何保存模型加载模型
  - Pytorch 调整optimizer的learning rate
- Plot
  - matlab
    - 绘制groupbars
  - python
  
- File process
  - matlab
  
  - python
  
- Algorithm
  - The clustering evaluation metric from sklearn

- sklearn
  - generate the classification task

- Data processing
  - 利用sklearn和Seglearn来构造多维的时间序列，并用划窗分割
  - 利用numpy和random生成三角函数曲线，并添加高斯噪声
  - 利用numpy和torch生成用于测试LSTM的加法测试数据
  - Transfer time series prediction task to supervised learning task
  
- 

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
            return agg, (X, y)
        ```
        </details>
