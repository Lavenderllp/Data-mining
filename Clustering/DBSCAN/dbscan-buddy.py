#基于密度的聚类

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from sklearn import preprocessing
from sklearn import metrics
start = datetime.now()
outputfile = 'out_buddymove_holidayiq.csv'
data = pd.read_csv('buddymove_holidayiq.csv')
data.drop(data.columns[0],axis=1,inplace=True)

scaler = preprocessing.StandardScaler()
x_data = scaler.fit_transform(data)

#用DBSCAN做聚类

from sklearn.cluster import DBSCAN

db = DBSCAN().fit(x_data)

labels = db.labels_
data['cluster_db'] = labels
data.to_csv(outputfile)
score_db = metrics.silhouette_score(data,labels)
print("轮廓系数=",score_db)
from sklearn.manifold import TSNE
tsne = TSNE()
tsne.fit_transform(data)
tsne = pd.DataFrame(tsne.embedding_)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure()

for i in set(labels):
    d = tsne[data['cluster_db']==i]
    plt.scatter(d[0], d[1])
plt.show()
end = datetime.now()
print("耗时=",(end-start).seconds)

