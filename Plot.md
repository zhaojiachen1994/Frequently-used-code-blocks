- Matlab

  - 绘制groupbars
```matlab
wa=[37.6,40.4;75.4,92.9];
aw=[29.8,53.5;79.3,88.1];
figure
bar(wa,'grouped')

set (gcf,'Position',[100,100,300,150], 'color','w')
set(gca, 'YGrid', 'on', 'XGrid', 'off')
xticklabels({'SURF','DeCaf6'})
ylabel('Accuracy (%)')
ylim([20,100])
legend('SVM', 'DGSA')

figure
bar(aw,'grouped')

set (gcf,'Position',[100,100,300,150], 'color','w')
set(gca, 'YGrid', 'on', 'XGrid', 'off')
xticklabels({'SURF','DeCaf6'})
ylabel('Accuracy (%)')
ylim([20,100])
```
![image](https://github.com/zhaojiachen1994/Frequently-used-code-blocks/blob/master/Figures/groupedbar.png)


-----------------------------------------------------------------------------------------------------------------------------------

<details> 
    <summary><strong>   一维高斯混合模型的拟合曲线绘制   </strong></summary>

ref: [1]  https://blog.csdn.net/miao_9/article/details/53511487
     [2]  官方文档-gmdistribution
```matlab
  % GENERATE DATAS
  p = [0.4 0.6]; % p is the proportion of two-component Gaussian distribution
  mu = [0; 5]
  sigma =[0.8]
  gm = gmdistribution(mu,sigma,p)
  rng('default'); % For reproducibility
  [X,compIdx] = random(gm,100);
  numIdx1 = sum(compIdx == 1)

  % FIT THE DATA WITH GMM MODEL
  options = statset('Display','final');
  obj = gmdistribution.fit(X,2,'Options',options);

  %PLOT THE CURVE AND RAW DATA
  figure
  fun = @(x)pdf(obj, [x]);
  t = linspace(-5,10)';
  hold on
  plot(t, fun(t))
  plot(X,0,'r*')
```
</details>

<div align=center><img src ="https://github.com/zhaojiachen1994/Frequently-used-code-blocks/blob/master/Figures/1d-gmm.png"/></div>

<details> 
    <summary><strong>   一维高斯模型的拟合曲线绘制（基础概率都可以拟合）-fitdist   </strong></summary>

```matlab
rng('default'); % For reproducibility
figure
hold on 
num=30;
s1 = normrnd(0,1,num,1)
pd_s1 = fitdist(s1, 'Normal');
t = -5:0.1:15;
y = pdf(pd_s1,t);
plot(t,y,'LineWidth',0.5)
plot(s1,zeros(num,1),    's',    'MarkerFaceColor','b',  'MarkerEdgeColor','b',   'MarkerSize', 5)
```

![image](https://github.com/zhaojiachen1994/Frequently-used-code-blocks/blob/master/Figures/1d-gaussian-fit.png)

