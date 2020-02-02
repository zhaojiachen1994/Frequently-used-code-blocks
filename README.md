# Frequently-used-code-blocks
Some frequently used code blocks.
---

<details>
<summary><strong>   Python project structure for a learning paper  </strong></summary>

  Reference: [DeepADoTS](https://github.com/KDD-OpenSource/DeepADoTS]), Self's AnomalyDetection
  
<pre>
- Dataset(Fold)                - 保存和处理真实数据集
  - dataset1(Fold)                - 用来保存数据, with Figure
  - dataset2(Fold)                - 用来保存数据, with Figure
  - dataset.py(class)             - 数据集Class
  
- Synthetic Dataset(Fold)             - 生成保存模拟数据集
  - dataset1(Fold)                - 用来保存数据
  - dataset1(Fold)                - 用来保存数据
  - Synthetic.py(Class)          - 仿真数据类
  
- Models(Fold)                 - 模型文件夹
  - algorithm_utils.py(Base)      - base function for other model 模型的通用函数定义
  - model1.py(Class)              - 定义model1的类
  - model2.py(Class)              - 定义model2的类
  
- Evaluation(Fold)             - 评价网络优劣
  - config.py(Helper)             - copy from DeepADoTS Des: create the logging file
  - evaluator(Class)              - input is list of dataset, list of models
  - evaluate_self(Exp)            - experiment to evaluate proposed method 
  - evaluate_deep(Exp)            - experiment to evaluate other deep method
  - evaluate_sk(Exp)              - experiment to evaluate traditional method
  - Results(Fold)                 - fold to save evaluation results
    - logs(Fold)                      - fold to save logging file
    - csv(Fold)                       - fold to save the csv results
    - fig(Fold)                       - fold to save fig if needed
    
- Analysis(Fold)               - 分析实验方法
  - evaluate_component1.py(exp)   - 衡量一个部分的效果
  - plotExample1.py(Exp)          - 可视化一个数据集的结果(从models,Evaluation中继承方法)
  - Results(Fold)                 - fold to save evaluation results
    - logs(Fold)                      - fold to save logging file
    - csv(Fold)                       - fold to save the csv results
    - fig(Fold)                       - fold to save fig if needed
    
- Refpackages(Fold)            - 保存参考代码
  - Package1(Fold)                - 下载保存package1, indluding the necessary dataset
  - rerunpackage1(exp)            - rerun package1，确定代码的正确性
  - Mypackage1.py(Class)          - 重新包装代码已适用于自己,(从Package1中中继承类)
  - Checkpackage1.py(exp)         - 测试代码是否适用于自己,(从models,Evaluation中继承方法)
  - Results(Fold)                 - fold to save evaluation results
    - logs(Fold)                      - fold to save logging file
    - csv(Fold)                       - fold to save the csv results
    - fig(Fold)                       - fold to save fig if needed
</pre>

TIPS:
  - 文件夹命名方式：首字母大写
  - py文件命名方式：小写
  - 类命名方式：大写， 类对象命名方式：首字母大写
  - 函数命名方式：大小写间隔
  - 变量命名方式：小写_小写

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
