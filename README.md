# Frequently-used-code-blocks
Some frequently used code blocks.
---

<details>
<summary><strong>   Python project structure  </strong></summary>

Reference: [DeepADoTS](https://github.com/KDD-OpenSource/DeepADoTS]), Self's AnomalyDetection

  
<pre>
- Dataset(Fold)           - 保存和处理真实数据集
  - dataset1(Fold)          - 用来保存数据
  - dataset2(Fold)          - 用来保存数据
  - Dataset.py(class)       - 数据集Class
  
- Simulation(Fold)        - 生成保存模拟数据集
  - dataset1(Fold)          - 用来保存数据
  - dataset1(Fold)          - 用来保存数据
  - simulation.py(Class)    - 仿真数据类
  
- Models(Fold)            - 模型文件夹
  - algorithm.utils.py(Base)      - base function for other model 模型的通用函数定义
  - model1.py(Class)               - 定义model1的类
  - model2.py(Class)               - 定义model2的类
  
- Evaluation(Fold)        - 评价网络优劣
  - config.py(Helper)               - copy from DeepADoTS Des: create the logging file
  - evaluator(Class)                - input is list of dataset, list of models
  - evaluate_self(Exp)              - experiment to evaluate proposed method 
  - evaluate_deep(Exp)              - experiment to evaluate other deep method
  - evaluate_sk(Exp)                - experiment to evaluate traditional method
  - Results(Fold)                   - fold to save evaluation results
    - logs(Fold)                          - fold to save logging file
    - csv(Fold)                           - fold to save the csv results
    
- Analysis(Fold)  
  
</pre>
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
