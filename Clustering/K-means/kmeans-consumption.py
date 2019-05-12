import pandas as pd
from sklearn.cluster import KMeans
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from datetime import datetime

start = datetime.now()
inputfile = 'consumption_data.xls'
outputfile = 'data_type.xls'

#分成三类
k = 3
iteration = 5000
data = pd.read_excel(inputfile,index_col='Id')
data_zs = 1.0*(data-data.mean())/data.std()
#数据预处理
scaler = preprocessing.StandardScaler()
x_data = scaler.fit_transform(data)
model = KMeans(n_clusters=k,max_iter=iteration)
model.fit(x_data)

#统计各个类别的数目
r1 = pd.Series(model.labels_).value_counts()
#找出聚类中心  聚类中心不一定是原始数据中存在的点
r2 = pd.DataFrame(model.cluster_centers_)
r = pd.concat([r2,r1],axis=1)  #横向连接
r.columns = list(data.columns)+[u'类别数目']


r = pd.concat([data,pd.Series(model.labels_,index=data.index)],axis=1)
r.columns = list(data.columns)+[u'聚类类别']
r.to_excel(outputfile)
#计算轮廓系数
from sklearn import metrics

score = metrics.silhouette_score(r,r[u'聚类类别'])
print(score)

#将聚类结果可视化
tsne = TSNE()
tsne.fit_transform(data_zs)
tsne = pd.DataFrame(tsne.embedding_,index=data_zs.index)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure()
d = tsne[r[u'聚类类别']==0]
plt.plot(d[0],d[1],'r.')
d = tsne[r[u'聚类类别']==1]
plt.plot(d[0],d[1],'go')
d = tsne[r[u'聚类类别']==2]
plt.plot(d[0],d[1],'b*')
plt.show()
end = datetime.now()
print("耗时=",(end-start).seconds)