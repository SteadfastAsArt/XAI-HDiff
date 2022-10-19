import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import pandas as pd

fig = plt.figure(figsize=(24, 16), dpi=300)
# 绘制画板(figsize 设置图形的大小，50为图形的宽，30为图形的高，单位为英寸,dpi 为设置图形每英寸的点数600)
plt.rcParams['font.sans-serif'] = 'Arial'
# 使用rc配置文件来自定义图形的各种默认属性,设置字体为 'Arial'
m = Basemap(projection='robin', lat_0=0,
            lon_0=0, )  # projection='robin', llcrnrlon = -30, llcrnrlat = -90, urcrnrlon = -30, urcrnrlat = 90
# lat_0=0, lon_0=-150, projection='moll',
# llcrnrlat=-90, urcrnrlat=90, llcrnrlon=-180, urcrnrlon=180,

# 实例化一个map ，投影方式为‘robin’
ax = plt.gca()
ax.xaxis.set_inverted(False)
m.shadedrelief(scale=0.3)  # scale=0.3
# m.warpimage()
m.drawcoastlines()
# 画海岸线
m.drawcountries()
# 画国界线
m.drawmapboundary()  # fill_color='white'
# 画大洲，颜色填充为白色
# m.bluemarble(scale=0.3)

parallels = np.arange(-90., 90., 45.)
m.drawparallels(parallels, labels=[True, True, True, False], fontsize=25, zorder=1)
# 标签=(左,右,上,下)
# 这两行画纬度，范围为[-90,90]间隔为60，标签标记在右、上边
meridians = np.arange(-180., 180., 60.)
m.drawmeridians(meridians, labels=[True, False, True, True], fontsize=25, zorder=1)
# 这两行画经度，范围为[-180,180]间隔为90,标签标记为左、下边


# my_dataset = pd.read_excel(r"D:\pre_grad_research\MantleWater\HDiff-XAI\data_all_aug02.xlsx")
# my_dataset = pd.read_excel(r"D:\pre_grad_research\MantleWater\HDiff-XAI\all_data_1221_01.xlsx")
my_dataset = pd.read_excel("./dataset/all_data_20220606.xlsx", sheet_name='all')
# 读取数据表
# lon, lat = m(df['lon'], df['lat'])   # TODO: custom dataset here!!!
# lon, lat为给定的经纬度，可以使单个的，也可以是列表
# clusters = [0, 1, 2, 3, 4, 5]
colors = ['red', 'pink', 'yellow', '#5fe1ff']  # 'fuchsia', 'lime', 'pink'
geosets = ['OIB', 'IAB', 'CIB', 'LIP']
markers = ['o', 's', 'd', 'v', 'p', 'h']
# 设定不同cluster值的颜色、形状
# for (geoset, color, marker) in zip(geosets, colors, markers):
#     my_data = my_dataset[my_dataset['Geo_Set'] == geoset]
#     print(my_data.shape)
#     lon, lat = m(my_data['longitude'], my_data['latitude'])
#     m.scatter(lon, lat, marker='D', s=100, c=color, edgecolor='0', alpha=0.75, label=geoset,
#               zorder=2)  # s=my_data['H2O']

lon, lat = m(my_dataset['longitude'], my_dataset['latitude'])
m.scatter(lon, lat, edgecolor='grey', c=my_dataset['H2O'], alpha=0.75, zorder=2, cmap='Blues', s=150)

"""legend"""
# ax.legend(title="Geological\nSetting", borderpad=0.55, labelspacing=0.3, handlelength=1.5, markerscale=1.5,
#           title_fontsize=25, fontsize=25, framealpha=1, bbox_to_anchor=(1.0, 0.95), )  # prop={'size': 20}

cb = m.colorbar(pad=1)
cb.ax.tick_params(labelsize=30)
# 设置色标刻度字体大小
cb.set_label('Water Content', fontsize=30)
plt.legend()

plt.savefig('./global_HDiff-XAI_0630_h2o.png', format='png', transparent=True)  # './global_HDiff-XAI.png', transparent=True
# plt.show()

# m.scatter(lon, lat, edgecolor='grey', marker='D', linewidths=0.5, s=75, alpha=0.15, cmap='Blues')
#  color='r', c=M, cmap='BuPu', vmax=3,vmin=0,
# 标注出所在的点，s为点的大小，还可以选择点的性状和颜色等属性 vmax=1200,vmin=0,vmax=300,vmin=0，
# cmap: colormap，用于表示从第一个点开始到最后一个点之间颜色渐进变化；
# alpha:  设置标记的颜色透明度
# linewidths:  设置标记边框的宽度值
# edgecolors:  设置标记边框的颜色
