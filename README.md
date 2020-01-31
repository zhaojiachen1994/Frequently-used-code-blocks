# Frequently-used-code-blocks
Some frequently used code blocks.
---

<details>
<summary><strong>   Python project structure  </strong></summary>
  
- Dataset(fold)
  
  - dataset1(fold)[用来保存数据]
  - dataset2(fold)[用来保存数据]


</details>

-----------------------------------------------------------------------------------------------------------------------------------

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
  
- TEMPLATE
<details><summary><strong>   双折叠  </strong></summary><blockquote>
  
        <details><summary><strong>   Title  </strong></summary><blockquote>
        <details><summary><strong>   Code  </strong></summary><blockquote>

        ```matlab
        code
        ```

        </blockquote></details>

        <details open><summary><strong>   Figure  </strong></summary>  
        <div align=left><img src ="https://github.com/zhaojiachen1994/Frequently-used-code-blocks/blob/master/Figures/groupedbar.png" width="300" height="150"/></div>
        </details>

        </blockquote></details>

        -----------------------------------------------------------------------------------------------------------------------------------

</blockquote></details>

<details><summary><strong>   单折叠  </strong></summary><blockquote>
  
    <details>
    <summary><strong>   Python ROC curves  </strong></summary>

     ```python
     
     ```

    </details>

    <div align=left><img src ="https://github.com/zhaojiachen1994/Frequently-used-code-blocks/blob/master/Figures/rocplot.png" width="200" height="120"/></div>

    -----------------------------------------------------------------------------------------------------------------------------------

</blockquote></details>

seeds = np.random.randint(np.iinfo(np.uint32).max, size=RUNS, dtype=np.uint32)
