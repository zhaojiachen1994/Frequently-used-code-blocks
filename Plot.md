<details>
<summary><strong>   Matplotlib 参考链接  </strong></summary>
  

 - [Color references](http://tableaufriction.blogspot.com/2012/11/finally-you-can-use-tableau-data-colors.html)
 - [官方配色](https://matplotlib.org/examples/color/colormaps_reference.html)
 - [Color names](https://matplotlib.org/2.0.2/examples/color/named_colors.html)
 - [Legend](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.legend.html)
 - [Mathtext on figure](https://matplotlib.org/3.1.3/gallery/text_labels_and_annotations/mathtext_examples.html#sphx-glr-gallery-text-labels-and-annotations-mathtext-examples-py)
 
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
  
[How to add figlegend](https://stackoverflow.com/questions/10101700/moving-matplotlib-legend-outside-of-the-axis-makes-it-cutoff-by-the-figure-box) 
  
  
 ```python
    import matplotlib.patches as patches #用来画长方形
 
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
    linecolors = plt.get_cmap('Set1').colors # other useful colors: ['coral', 'seagreen', 'darkgrey','orangered','slateblue']
    markers = ['X', '^', 'P', 'd', '*'] # can be '. o v ^ s P + d * x X
    # TIP： uppercase letter means filled markers
    markersize = 6
    xyticksize = 8
    xylabelfontsize = 14
    titlefontsize = 20
    legendfontsize = 12
# STEP4: PLOT THE FIGURE
    t = np.arange(n) #n is the number of points in eachline
    ax.plot(t, AUC_ISF, marker=markers[0], color=linecolors[4], label='IsoForest', lw=lw, ms=markersize)
    ax.plot(t, AUC_IOF,         marker=markers[1], color=linecolors[1], label='IOF',       lw=lw, ms=markersize)
    ax.plot(t, AUC_oneclasssvm, marker=markers[2], color=linecolors[2], label='OSVM',      lw=lw, ms=markersize)
    ax.plot(t, AUC_autoEncoder, marker=markers[3], color=linecolors[3], label='DeepCoder', lw=lw, ms=markersize)
    ax.plot(t, AUC_unDevcoder,  marker=markers[4], color=linecolors[0], label='unDevCoder',lw=lw, ms=markersize)
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
    # TIPs: just lock the ticks 
    
    plt.xticks(t, (10, 50, 100, 500, 1000, 5000))
    # TIPs: Arbitrarily change the xticks. t is the values of x axis.
    
    # set empty xticks, yticks
    plt.xticks([])
    plt.yticks([])

# STEP6: Add text or rectangle if needed
    textstr='line1 \n25 line2 \n line3.'
    ax.annotate(textstr,fontsize=annnotefontsize, xy=(50, 1.2), xytext=(75, 0.6),
       arrowprops=dict(facecolor='b', edgecolor='b', width=5, shrink=0.1, alpha=0.5)) # xy是箭头位置，xytext是文本位置，标准为横纵坐标。
    rect = patches.Rectangle(xy=(25, 0.3), width=25, height=1.08, linewidth=1, edgecolor='r', facecolor='none') 
    ax.add_patch(rect) #添加长方形
    
    fig.tight_layout()
    plt.show()
    f.savefig(f"figname.pdf")
    
    
 # how to add leneng for multiple subplots
    fig = plt.figure(num=1,figsize=[10, 3.5])
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    handles, labels = ax1.get_legend_handles_labels()
    lgd = fig.legend(handles, labels, loc='upper center', ncol=5, labelspacing=0.,bbox_to_anchor=(0.5, 1.1))
    
    fig.tight_layout()
    fig.savefig('k_sensitivity.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')
    
    ref: https://stackoverflow.com/questions/10101700/moving-matplotlib-legend-outside-of-the-axis-makes-it-cutoff-by-the-figure-box
 ```
 
</details>

<div align=left><img src ="https://github.com/zhaojiachen1994/Frequently-used-code-blocks/blob/master/Figures/matplotlib_template.png" width="200" height="120"/></div>

-----------------------------------------------------------------------------------------------------------------------------------

<details>
<summary><strong>   Plot tree or graph structure with Graphviz package  </strong></summary>

- [How to install Graphviz package](https://stackoverflow.com/questions/35064304/runtimeerror-make-sure-the-graphviz-executables-are-on-your-systems-path-aft)
  - For Windows:
    1. Install windows package from [here](https://graphviz.gitlab.io/_pages/Download/Download_windows.html)
    2. Install python graphviz package by pip install graphviz
    3. Add C:\Program Files (x86)\Graphviz2.38\bin to User path
    4. Add C:\Program Files (x86)\Graphviz2.38\bin\dot.exe to System Path
    5. import os
       os.environ["PATH"] += os.pathsep + r'C:\Program Files (x86)\Graphviz2.38\bin'
- [Code examples](https://graphviz.readthedocs.io/en/stable/examples.html)

- [How to bold parts of labels](https://stackoverflow.com/questions/30194104/graphviz-bold-font-attribute)
  - successful example: tree.node(f'{ind}', label=f"< <B>{ind}</B> ({info_df['Dist2Peak'][i]:0.3f}) >") 
```python
    tree = Digraph('GASP tree', filename='tree.gv',node_attr={'color': 'lightblue2', 'style': 'filled'})
    tree.attr('node', shape='ellipse')
    tree.attr('node', fontname = "Arial")
    tree.attr('node', fontsize='20')
    tree.attr('edge', fontsize='14')
    for i, ind in enumerate(inds):
        print(ind, info_df['onestep'][i])
        if ind != 7:
            tree.node(f'{ind}', label=f"< <B>{ind}</B> ({info_df['Dist2Peak'][i]:0.2f})>")
        else:
            tree.node(f'{ind}', label=f"<<B>{ind}</B> (Density Peak)>")
    for i, ind in enumerate(inds):
        if ind != 7:
            tree.edge(f"{info_df['bigger_nn'][ind-1]+1}",f"{ind}", label=f" {info_df['onestep'][i]:0.3f}")
    tree.view()
 ```
  
</details>


-----------------------------------------------------------------------------------------------------------------------------------

<details>
<summary><strong>   Plot tree or graph structure with Graphviz package  </strong></summary>

- [How to install Graphviz package](https://stackoverflow.com/questions/35064304/runtimeerror-make-sure-the-graphviz-executables-are-on-your-systems-path-aft)
  - For Windows:
    1. Install windows package from [here](https://graphviz.gitlab.io/_pages/Download/Download_windows.html)
    2. Install python graphviz package by pip install graphviz
    3. Add C:\Program Files (x86)\Graphviz2.38\bin to User path
    4. Add C:\Program Files (x86)\Graphviz2.38\bin\dot.exe to System Path
    5. import os
       os.environ["PATH"] += os.pathsep + r'C:\Program Files (x86)\Graphviz2.38\bin'
- [Code examples](https://graphviz.readthedocs.io/en/stable/examples.html)

- [How to bold parts of labels](https://stackoverflow.com/questions/30194104/graphviz-bold-font-attribute)
  - successful example: tree.node(f'{ind}', label=f"< <B>{ind}</B> ({info_df['Dist2Peak'][i]:0.3f}) >") 
```python
    tree = Digraph('GASP tree', filename='tree.gv',node_attr={'color': 'lightblue2', 'style': 'filled'})
    tree.attr('node', shape='ellipse')
    tree.attr('node', fontname = "Arial")
    tree.attr('node', fontsize='20')
    tree.attr('edge', fontsize='14')
    for i, ind in enumerate(inds):
        print(ind, info_df['onestep'][i])
        if ind != 7:
            tree.node(f'{ind}', label=f"< <B>{ind}</B> ({info_df['Dist2Peak'][i]:0.2f})>")
        else:
            tree.node(f'{ind}', label=f"<<B>{ind}</B> (Density Peak)>")
    for i, ind in enumerate(inds):
        if ind != 7:
            tree.edge(f"{info_df['bigger_nn'][ind-1]+1}",f"{ind}", label=f" {info_df['onestep'][i]:0.3f}")
    tree.view()
 ```
  
</details>

-----------------------------------------------------------------------------------------------------------------------------------

<details><summary><strong>  plot坐标轴设置方法  </strong></summary>

（1）去除坐标轴使用axis off
      
      如果想要x的坐标没有：set（gca,'xtick',[])
      
      关闭边框：set(gcf,'box','off')
      
（2）坐标轴设置方法
```matlab
axis off;% 去掉坐标轴
axistight;% 紧坐标轴
axisequal;% 等比坐标轴
axis([-0.1, 8.1, -1.1, 1.1]);% 坐标轴的显示范围
% gca: gca, h=figure(...);
set(gca,'XLim',[3 40]);% X轴的数据显示范围
set(gca,'XTick',[-3.14,0,3.14] );% X轴的记号点
set(gca,'XTicklabel',{'-pi','0','pi'});% X轴的记号
set(gca,'XTick', []);% 清除X轴的记号点
set(gca,'XGrid','on');% X轴的网格
set(gca,'XDir','reverse');% 逆转X轴
set(gca,'XColor','red');% X轴的颜色
'''

1. axis([xmin xmax ymin ymax])
设置当前图形的坐标范围，分别为x轴的最小、最大值，y轴的最小最大值
2. V=axis
返回包含当前坐标范围的一个行向量
3. axis auto
将坐标轴刻度恢复为自动的默认设置
4. axis manual
冻结坐标轴刻度，此时如果hold被设定为on，那么后边的图形将使用与前面相同的坐标轴刻度范围
5. axis tight
将坐标范围设定为被绘制的数据范围
6. axis fill
这是坐标范围和屏幕的高宽比，使得坐标轴可以包含整个绘制的区域。该选项只有在PlotBoxaApectRatio或DataAspectRatioMode被设置为‘manual’模式才有效
7. axis ij
将坐标轴设置为矩阵模式。此时水平坐标轴从左到有取值，垂直坐标从上到下
8. axis xy
将坐标设置为笛卡尔模式。此时水平坐标从左到右取值，垂直坐标从下到上取值
9. axis equal
设置屏幕高宽比，使得每个坐标轴的具有均匀的刻度间隔
10. axis square
将坐标轴设置为正方形
11. axis normal
将当前的坐标轴框恢复为全尺寸，并将单位刻度的所有限制取消
12. axis vis3d
冻结屏幕高宽比，使得一个三维对象的旋转不会改变坐标轴的刻度显示
13. axis off
关闭所有的坐标轴标签、刻度、背景
14. axis on
打开所有的坐标轴标签、刻度、背景

</details>
-----------------------------------------------------------------------------------------------------------------------------------
