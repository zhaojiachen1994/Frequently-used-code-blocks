
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

