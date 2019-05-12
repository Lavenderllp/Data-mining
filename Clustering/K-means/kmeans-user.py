import pandas as pd
from sklearn.cluster import KMeans
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from datetime import datetime
start = datetime.now()
inputfile = 'Data_User.csv'
outputfile = 'out_Data_User.csv'

#消费者数据分成四类  分别是very_low  Low Middle High
k = 4
iteration = 5000
data = pd.read_csv(inputfile,index_col=None)
data1 = pd.read_csv(inputfile,index_col=None)
data.drop(data.columns[5],axis=1,inplace=True)

'''
#数据预处理
scaler = preprocessing.StandardScaler()
x_data = scaler.fit_transform(data)
'''

model = KMeans(n_clusters=k,max_iter=iteration)
model.fit(data)
'''
#统计各个类别的数目
r1 = pd.Series(model.labels_).value_counts()
#找出聚类中心  聚类中心不一定是原始数据中存在的点
r2 = pd.DataFrame(model.cluster_centers_)
r = pd.concat([r2,r1],axis=1)  #横向连接
r.columns = list(data.columns)+[u'类别数目']
'''



r = pd.concat([data1,pd.Series(model.labels_)],axis=1)
r.columns = list(data1.columns)+[u'聚类类别']

r.to_csv(outputfile)
#聚类评估  轮廓系数
from sklearn import metrics

score = metrics.silhouette_score(data,r[u'聚类类别'])
print(score)
#将聚类结果可视化
tsne = TSNE()
tsne.fit_transform(data)
tsne = pd.DataFrame(tsne.embedding_)
color=['r','g','b','black']
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure()

for i in range(4):
    d = tsne[r[u'聚类类别']==i]
    plt.scatter(d[0], d[1],marker=str(i+1),c=color[i])
plt.show()
end = datetime.now()
print("耗时=",(end-start).seconds)