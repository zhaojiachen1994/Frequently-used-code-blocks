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

-----------------------------------------------------------------------------------------------------------------------------------
