
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
Copy from DeepADoTS
Des: create the logging file
'''

import logging
import os
import sys

import re
import time

# Use: logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL
LOG_LEVEL = logging.INFO######################################################################## Set the log level
CONSOLE_LOG_LEVEL = logging.INFO


def init_logging(output_dir='reports/logs', file_name=None):##################################################### the save direction
    # Prepare directory and file path for storing the logs
    timestamp = time.strftime('%Y-%m-%d-%H%M%S')
    if file_name is None:
        log_file_path = os.path.join(output_dir, '{}.log'.format(timestamp))######################### the log file name is timestamp
    else:
        log_file_path = os.path.join(output_dir, f'{file_name}.log')  ######################### the log file name is timestamp
    os.makedirs(output_dir, exist_ok=True)

    # Actually initialize the logging module
    log_formatter = logging.Formatter(fmt='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')
    root_logger = logging.getLogger()
    root_logger.setLevel(LOG_LEVEL)
    # Removes previous handlers (required for running pipeline multiple times)
    root_logger.handlers = []

    # Store logs in a log file in reports/logs
    file_handler = logging.FileHandler(log_file_path)  # mode='w'
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)

    # Also print logs in the standard output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(CONSOLE_LOG_LEVEL)
    console_handler.addFilter(DebugModuleFilter(['^src\.', '^root$']))
    root_logger.addHandler(console_handler)

    # Create logger instance for the config file
    logger = logging.getLogger(__name__)
    logger.debug('Logger initialized')


class DebugModuleFilter(logging.Filter):
    def __init__(self, pattern=[]):
        logging.Filter.__init__(self)
        self.module_pattern = [re.compile(x) for x in pattern]

    def filter(self, record):
        # This filter assumes that we want INFO logging from all
        # modules and DEBUG logging from only selected ones, but
        # easily could be adapted for other policies.
        if record.levelno == logging.DEBUG:
            # e.g. src.evaluator.evaluation
            return any([x.match(record.name) for x in self.module_pattern])
        return True
```

</blockquote></details>

<details open><summary><strong>   Figure  </strong></summary>  
<div align=left><img src ="https://github.com/zhaojiachen1994/Frequently-used-code-blocks/blob/master/Figures/groupedbar.png" width="300" height="150"/></div>
</details>

</blockquote></details>

-----------------------------------------------------------------------------------------------------------------------------------



