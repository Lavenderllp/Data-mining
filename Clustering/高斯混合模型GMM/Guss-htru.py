import pandas as pd
from sklearn import preprocessing
from datetime import datetime

start = datetime.now()
inputfile = 'HTRU_2.csv'
outputfile = 'out_HTRU_2.csv'

k = 2

iteration = 5000

data = pd.read_csv(inputfile)
data.drop(data.columns[5],axis=1,inplace=True)


scaler = preprocessing.StandardScaler()
x_data = scaler.fit_transform(data)

#数据预处理

import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture
model = GaussianMixture(n_components=k,covariance_type='full',random_state=0,max_iter=iteration)
model.fit(data)
labels = model.predict(data)
r = pd.concat([data,pd.Series(labels,index=data.index
                              )],axis=1)
r.columns = list(data.columns) + [u'聚类类别']
r.to_csv(outputfile)
#聚类评估  轮廓系数
from sklearn import metrics

score = metrics.silhouette_score(data,r[u'聚类类别'])
print("轮廓系数=",score)
from sklearn.manifold import TSNE
tsne = TSNE()
tsne.fit_transform(data)
tsne = pd.DataFrame(tsne.embedding_)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure()

for i in range(2):
    d = tsne[r[u'聚类类别']==i]
    plt.scatter(d[0], d[1])
plt.show()
end = datetime.now()
print("耗时=",(end-start).seconds)