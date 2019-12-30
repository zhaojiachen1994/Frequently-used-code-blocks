
<details> 
    <summary><strong>   对比实验执行并保存结果(以 domain adaptation 为例)   </strong></summary>

```python
method = 'DGSA' #point the method name

reusltfile = './results/results_{}.csv'.format(method) # define the results saving path
Acc_dict = {'method':method}  
Acc_list = []
for i in domains:
    for j in domains:
        if i != j:
            Xs, Ys, Xt, Yt = loadofficehome(i,j)
            t0 = time.time()

            # check if the code can run
            # acc = np.random.rand(1)[0]

            if method == 'svm':
              ...
            elif method == 'pca':
              ...

            runningTime = time.time()-t0

            print('{} | {}-{} | Acc: {:0.4f} | Time: {:0.4f} |'.format(method, i[0], j[0], acc, runningTime))

            print('-'*35)
            Acc_list.append(acc)
            Acc_dict['{}-{}'.format(i[0], j[0])] = [round(acc*100, 1)]

addResulttoCSV(Acc_dict, reusltfile)
```
</details>

----------------------------------------------------------------------------------------------------------------------------------------

<details> 
    <summary><strong>   GRID SEARCH   </strong></summary>

```python

# EXPERIMENTS FOR PARAMETERS GRID SEARCH.
from sklearn.model_selection import ParameterGrid

class MODEL:
    def __init__(self, param1=1, param2=4, param3 ='t'):
        self.param1, self.param2, self.param3 = param1, param2, param3
    def fit(self):
        print(self.param1, self.param2, self.param3)

# CHANGEABLE
model = MODEL()
param_search_dict = {'param1': [1, 2], 'param2': [4, 5], 'param3':['t','m','d']} #CAN CHOOSE PARTS OF THE PARAMETERS

# DON'T CHANGE
param_grid = ParameterGrid(param_search_dict)
for param_cur_dict in param_grid:
    for param_name in param_cur_dict:
        locals()[param_name] = param_cur_dict[param_name]
# CHANGEABLE
    model = MODEL(param1=param1, param2=param2, param3=param3)
    # MAY PLAY REPEATED RUNNING HERE
    model.fit()
    print(param_cur_dict)
    print('-'*50)


#TODO:
'''
1. save the results.
2. with cross validation and test set.
3. parallelize
'''

```
</details>

----------------------------------------------------------------------------------------------------------------------------------------

<details> 
    <summary><strong>   PARAMETERS GRID SEARCH WITH Multi-Process Parallel Computing   </strong></summary>

```python
# EXPERIMENTS FOR PARAMETERS GRID SEARCH WITH Multi-Process Parallel Computing
from sklearn.model_selection import ParameterGrid
import multiprocessing
import time
from sys import stdout


class MODEL:
	def __init__(self, param1=1, param2=4, param3='t'):
		super(MODEL, self).__init__()
		self.param1, self.param2, self.param3 = param1, param2, param3
	def fit(self):
		stdout.write('param1: {} | param2: {} | param3: {} | 5s\n'.format(self.param1, self.param2, self.param3))
		time.sleep(2)

	def pipe(self, param1=1, param2=4, param3='t'):
		self.__init__(param1, param2, param3)
		self.fit()

if __name__ == '__main__':
	### LOCAL ADAPTATION STEP1: DEFINE THE PARAMETER GRID AND MODEL.
	param_search_dict = {'param1': [1, 2], 'param2': [4, 5],'param3': ['t', 'm', 'd']}  # MUST CONTAIN ALL PARAMETERS
	model = MODEL()
	# DON'T CHANGE
	param_grid_dict = ParameterGrid(param_search_dict)
	param_grid_list = [tuple(param_cur_dict.values()) for param_cur_dict in param_grid_dict]
	cores = multiprocessing.cpu_count()
	with multiprocessing.Pool(processes=cores) as pool:
		t0 = time.time()
		### LOCAL ADAPTATION STEP2: DEFINE THE SOLVER PIPELINE.
		pool.starmap(model.pipe, param_grid_list)
		pool.close()
		pool.join()
		runningtime = time.time()-t0
		print(runningtime)
```
</details>

----------------------------------------------------------------------------------------------------------------------------------------

<details><summary><strong>   实验结果保存至log和csv文件  </strong></summary><blockquote>
<details><summary><strong>   Code  </strong></summary><blockquote>

```python
'''

```

</blockquote></details>

<details open><summary><strong>   Figure  </strong></summary>  
<div align=left><img src ="https://github.com/zhaojiachen1994/Frequently-used-code-blocks/blob/master/Figures/groupedbar.png" width="300" height="150"/></div>
</details>

</blockquote></details>

-----------------------------------------------------------------------------------------------------------------------------------



