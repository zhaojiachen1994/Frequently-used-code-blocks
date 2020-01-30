<details>
<summary><strong>   Matplotlib 参考链接  </strong></summary>
  
 [Legend](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.legend.html)
 
</details>

-----------------------------------------------------------------------------------------------------------------------------------

<details><summary><strong>   绘制groupbars  </strong></summary><blockquote>
<details><summary><strong>   Code  </strong></summary><blockquote>
  
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

</blockquote></details>

<details open><summary><strong>   Figure  </strong></summary>  
<div align=left><img src ="https://github.com/zhaojiachen1994/Frequently-used-code-blocks/blob/master/Figures/groupedbar.png" width="300" height="150"/></div>
</details>

</blockquote></details>



-----------------------------------------------------------------------------------------------------------------------------------

<details><summary><strong>   一维高斯混合模型的拟合曲线绘制  </strong></summary><blockquote>
ref: 
  [1]  https://blog.csdn.net/miao_9/article/details/53511487
  [2]  官方文档-gmdistribution
  
<details><summary><strong>   Code  </strong></summary><blockquote>

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

<details open><summary><strong>   Figure  </strong></summary>  
<div align=left><img src ="https://github.com/zhaojiachen1994/Frequently-used-code-blocks/blob/master/Figures/1d-gmm.png" width="300" height="150"/></div>
</details>

</blockquote></details>

-----------------------------------------------------------------------------------------------------------------------------------

<details> 
    <summary><strong>   2维高斯混合模型的拟合曲线绘制   </strong></summary>

```matlab
  % GENERATE DATAS
  p = [0.4 0.6]; % p is the proportion of two-component Gaussian distribution
  mu = [1 2;-3 -5];% for 2 dimension
  sigma = cat(3,[2 .5],[1 1]); % shared diagonal covariance matrix for 2 dimensions
  gm = gmdistribution(mu,sigma,p)
  rng('default'); % For reproducibility
  [X,compIdx] = random(gm,200);
  numIdx1 = sum(compIdx == 1)

  % FIT THE DATA WITH GMM MODEL
  options = statset('Display','final');
  obj = gmdistribution.fit(X,2,'Options',options);

  %PLOT THE CURVE AND RAW DATA
  scatter(X(:,1),X(:,2),10,'.')
  hold on
  h = ezcontour(@(x,y)pdf(obj,[x y]),[-8 6],[-8 6]);
  hold off
```
</details>

<div align=left><img src ="https://github.com/zhaojiachen1994/Frequently-used-code-blocks/blob/master/Figures/2d-gmm.png" width="300" height="150"/></div>

------------------------------------------------------------------------------------------------------------------------------------


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
</details>

<div align=left><img src ="https://github.com/zhaojiachen1994/Frequently-used-code-blocks/blob/master/Figures/1d-gaussian-fit.png" width="300" height="150"/></div>

----------------------------------------------------------------------------------------------------------------------------------------

<details> 
    <summary><strong>   Group bars with variance   </strong></summary>

```matlab
%1. acc of srm only
Acc1=[53.94, 44.24, 42.69, 29.74, 46.40];
Var1=[0.24, 0.30, 1.20 ,1.25, 0.73];
% 2. acc of srm+distribution alignment
Acc2=[48.1, 38.57, 36.43, 33.32, 43.88];
Var2=[1.85, 1.50, 0.24, 0.82, 0.92];
%3. acc of dgfk+srm
Acc3=[59.33,46.66,44.84,40.54, 50.76];
Var3=[1.38,1.28,3.16,0.26,1.07];
%4. acc of dgfk+da+srm
Acc4=[61.67,46.37,47.51,44.98,53.19];
Var4=[1.57,0.88,0.86,0.22,0.77];

%5. acc of all
Acc5=[62.2,47.2,49.7,44.6, 54.10]
Var5=[1.38,1.78,1,0.26,0.70]

Acc=[Acc1;Acc2;Acc3;Acc4;Acc5]';
figure
set (gcf,'Position',[300,300,550,350], 'color','w')
bar(Acc,'grouped')
set(gca, 'YGrid', 'on', 'XGrid', 'off')
xticklabels({'C-A','A-C','A-D','D-A','Mean'})
set (gca,'position',[0.1,0.1,0.8,0.8] )
legend('SRM (Baseline)', 'SRM+DA', 'SRM+DML', 'SRM+DML+DA','SRM+DML+DA+DPL')
ylabel('Accuracy (%)')
ylim([25,70])

e=[Var1;Var2;Var3;Var4;Var5]';
hold on 
numgroups = size(e,1);
numbars = size(e,2);
groupwidth = min(0.8, numbars/(numbars+1.5));
for i = 1:numbars
    x = (1:numgroups) - groupwidth/2 + (2*i-1)* groupwidth / (2*numbars); %aligning error bar with individual bar
    h = errorbar (x, Acc(:,i), e(:,i),'k','linestyle','none','lineWidth',0.5,'CapSize',5);
end
```
</details>

<div align=left><img src ="https://github.com/zhaojiachen1994/Frequently-used-code-blocks/blob/master/Figures/barwithvars.png" width="300" height="150"/></div>

----------------------------------------------------------------------------------------------------------------------------------------

<details>
<summary><strong>   Python ROC curves  </strong></summary>
  
 ```python
    
    def plotroc(self, scoresdf):
        # scoresdf is Dataframe with detetors(classifier) name as column name, y_pred as df.data
        # print(scoresdf.head())
        y_true= scoresdf['y_true']
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        f = plt.figure()
        lw = 2
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'deeppink','navy' ])
        for det,color in zip(detectors, colors):
            fpr[det.name], tpr[det.name], _ = roc_curve(y_true=y_true, y_score=scoresdf[det.name])
            roc_auc[det.name] = round(auc(fpr[det.name], tpr[det.name]), 2)
            plt.plot(fpr[det.name], tpr[det.name], color=color, lw=lw, label=f'{det.name} (area={roc_auc[det.name]})')
        print(roc_auc)
        plt.plot([0, 1], [0, 1], color='green', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=16)
        plt.ylabel('True Positive Rate', fontsize=16)
        plt.title(f'{self.datasets[0].data[0].name}')
        plt.legend(loc="lower right", fontsize=12)

        plt.show()
        f.savefig(f"roc_{self.datasets[0].data[0].name}.pdf", bbox_inches='tight')
 ```
 
</details>

<div align=left><img src ="https://github.com/zhaojiachen1994/Frequently-used-code-blocks/blob/master/Figures/rocplot.png" width="200" height="120"/></div>

-----------------------------------------------------------------------------------------------------------------------------------

<details>
<summary><strong>   Matplotlib 画图模板  </strong></summary>
  
 ```python
    t = np.linspace(1,100,100)
    data1 = np.random.rand(100)*0.5
    data2 = np.random.rand(100)*0.6+0.5
# STEP1: CREATE FIGURE
    fig = plt.figure(num=None, figsize=(6.4, 4.8), dpi=100, facecolor='w', edgecolor='w')
    # TIP: default figure size is (6.4, 4.8); default dpi is 100;
# STEP2: CREATE AXES
    ax = plt.subplot(111, facecolor='antiquewhite')
    # TIP: set(111) when want to plot one
# STEP3: SET THE PARAS
    lw = 1
    color = ['b','r']
    xyticksize = 8
    xylabelfontsize = 14
    titlefontsize = 20
    legendfontsize = 12
# STEP4: PLOT THE FIGURE
    ax.plot(t, data1, lw=lw, color=color[0], label='blue line')
    ax.plot(t, data2, lw=lw, color=color[1], label='red line')
    ax.legend(loc="lower right", fontsize=legendfontsize)
    # legend set: https: // matplotlib.org / api / _as_gen / matplotlib.pyplot.legend.html
# STEP5: ADJUST THE PLOT
    ax.set_title('(a) Title', fontsize = titlefontsize)
    ax.set_xlabel('Time', fontsize=xylabelfontsize)
    ax.set_ylabel('Value (%)',fontsize=xylabelfontsize)

    ax.set_xlim([0, 100])
    ax.set_ylim([-0.5, 1.5])

    ax.grid(True, axis='both')
    ax.tick_params(axis='both', direction='in', length=3, which='major', labelsize=xyticksize)
    # TIP: axis could be {'x', 'y', 'both'}
    #      grid color, linestyle, linewidth can be adjusted by tick_params

    ax.set_xticks([0, 20, 25, 40, 60, 80, 100])
    ax.set_yticks([-0.5, 0, 0.5, 1, 1.5])

    fig.tight_layout()
    plt.show()
    f.savefig(f"figname.pdf")
 ```
 
</details>

<div align=left><img src ="https://github.com/zhaojiachen1994/Frequently-used-code-blocks/blob/master/Figures/matplotlib_template.png" width="200" height="120"/></div>

-----------------------------------------------------------------------------------------------------------------------------------


to do: from tabulate import tabulate 直接用python将字典生成latex的表格或者psql的表格。
