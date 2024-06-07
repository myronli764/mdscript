import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
labels = 'H C N O F P S Cl Br I M'.split()
#mae

figure = plt.figure(figsize = (10,7.5),dpi=200)
x = np.arange(len(labels))
x_ = [0,1,2,3,10]
width = 10
left, bottom, right, top = 0.2, 0.2, 0.7, 0.7
ax = figure.add_axes([left,bottom,right,top])

s = 15
plt.errorbar(x, Graph, label = r'Our work', color = '#EA5578', fmt = 'o', markersize = s)
plt.errorbar(x, Bleiziffer,  label = f'Bleiziffer JCIM 2018', color = '#5A5CE6', fmt = 's',     markersize = s, capsize=5, capthick=1)
plt.errorbar(x, Chodera,  label = 'Chodera arXiv 2019', color = '#F2B73A',fmt = '*',            markersize = s, capsize=5, capthick=1)
plt.errorbar(x, Horsfiled,  label = 'Horsfiled JCIM 2024', color = '#6AE875', fmt = '^',      markersize = s, capsize=5, capthick=1)


font = FontProperties()
font.set_family('serif')
#font.set_weight('bold')
font.set_name('Arial')
font.set_size(25)

font1 = FontProperties()
font1.set_family('serif')
#font1.set_weight('bold')
font1.set_name('Arial')
font1.set_size(40)

plt.rcParams['font.size'] = 30
ax.set_ylabel('MAE (e)', fontproperties = font1,labelpad=15)
ax.set_xlabel('Element', fontproperties = font1,labelpad=15)
fontdict = {'fontproperties': font1}
ax.set_title('Comparison to other works', fontdict = fontdict,pad=25)
ax.set_title('Comparison to other works', fontdict = fontdict,pad=25)
ax.set_xticks(x)
ax.set_xticklabels(labels)
bwidth = 3.2
plt.ylim([0,0.125])
ax.spines['left'].set_linewidth(bwidth)
ax.spines['bottom'].set_linewidth(bwidth)
ax.spines['right'].set_linewidth(bwidth)
ax.spines['top'].set_linewidth(bwidth)
plt.xticks(fontsize=30,name='Arial')
plt.yticks(fontsize=30,name='Arial')
ax.tick_params(which='major', length=6, width=2, direction='in',right=True, top=True)
ax.tick_params(axis='x', which='major', pad=15)
ax.tick_params(axis='y', which='major', pad=15)
legend = ax.legend(markerscale=1,prop=font,edgecolor='black',fancybox=False,borderpad=0.8,loc='upper left',framealpha=1,frameon=False,handlelength=0.5,handletextpad=0.2)
plt.grid(True,color='b',alpha=0.2)
legend.get_frame().set_linewidth(0)
#plt.show()
#plt.savefig('chargeCom.png',bbox_inches='tight',dpi=300)