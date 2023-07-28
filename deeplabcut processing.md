<details> 
    <summary><strong>   创建空的标注文件csv/h5   </strong></summary>

创建单个动物的标注文件：https://github.com/DeepLabCut/DeepLabCut/blob/main/examples/testscript.py
创建多个动物的标注文件：https://github.com/DeepLabCut/DeepLabCut/blob/main/examples/testscript_multianimal.py

<div align=left><img src ="https://github.com/zhaojiachen1994/Frequently-used-code-blocks/blob/master/Figures/dlc-label.png" width="300" height="150"/></div>
</details>

----------------------------------------------------------------------------------------------------------------------------------------

<details> 
    <summary><strong>   修改multi level columns的列名   </strong></summary>
    
```python  
    rename_dict = {'xiongzhui_1': 'thoracic_1', 'xiongzhui_2': 'thoracic_2'}
    df.rename(columns=rename_dict, level=1, inplace=True)
```
</details>

----------------------------------------------------------------------------------------------------------------------------------------
