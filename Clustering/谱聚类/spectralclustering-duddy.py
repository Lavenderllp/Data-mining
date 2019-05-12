import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import SpectralClustering
from sklearn import metrics
from datetime import datetime
start = datetime.now()
inputfile = 'buddymove_holidayiq.csv'
outputfile = 'out_buddymove_holidayiq.csv'

data = pd.read_csv(inputfile,index_col=None)
data.drop(data.columns[0],axis=1,inplace=True)

#数据预处理
scaler = preprocessing.StandardScaler()
x_data = scaler.fit_transform(data)

scores=[]
s = dict()
for index,gamma in enumerate((0.01,0.1,1,10)):
    for index,k in enumerate((3,5,7,9)):
        pred_y = SpectralClustering(n_clusters=k,gamma=gamma).fit_predict(x_data)
        print("Calinski-Harabasz Score with gamma=", gamma, "n_cluster=", k, "score=",
              metrics.calinski_harabaz_score(x_data, pred_y))
        tmp = dict()
        tmp['gamma'] = gamma
        tmp['n_cluster'] = k
        tmp['score'] = metrics.calinski_harabaz_score(x_data, pred_y)
        s[metrics.calinski_harabaz_score(x_data, pred_y)] = tmp
        scores.append(metrics.calinski_harabaz_score(x_data, pred_y))
print(np.max(scores))
print("最大得分项：")
print(s.get(np.max(scores)))


#返回分类标签
pred_y = SpectralClustering(n_clusters=3,gamma=1,affinity='rbf').fit_predict(x_data)

#谱聚类
r = pd.concat([data,pd.Series(pred_y,index=data.index)],axis=1)
r.columns = list(data.columns)+[u'聚类类别']

r.to_csv(outputfile)

#将聚类结果可视化
tsne = TSNE()
tsne.fit_transform(x_data)
tsne = pd.DataFrame(tsne.embedding_)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure()

for i in range(3):
    d = tsne[r[u'聚类类别']==i]
    plt.scatter(d[0], d[1])
plt.show()
#print(pred_y)
from sklearn import metrics
score = metrics.silhouette_score(r,r[u'聚类类别'])
print(score)
end = datetime.now()
print("耗时=",(end-start).seconds)




'''
#将聚类结果可视化
tsne = TSNE()
tsne.fit_transform(x_data)
tsne = pd.DataFrame(tsne.embedding_)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure()

for i in range(4):
    d = tsne[r[u'聚类类别']==i]
    plt.scatter(d[0], d[1])
plt.show()
'''

