#对数据进行聚类处理  KMeans算法处理
import pandas as pd
from sklearn.cluster import KMeans

datafile='Data_User.csv'
processedfile = 'data_processed.csv'

typelabel={'STG':'A','SCG':'B','STR':'C','LPR':'D','PEG':'E'}

k=3  #需要进行的聚类类别数

data = pd.read_csv(datafile)
keys=list(typelabel.keys())
result = pd.DataFrame()

print(data[[keys[0]]].as_matrix())

for i in range(len(keys)):
    print(u'正在进行"%s"的聚类'%keys[i])
    kmodel = KMeans(n_clusters=k)  #将每一列聚成三类
    kmodel.fit(data[[keys[i]]].as_matrix())  #训练模型
    r1 = pd.DataFrame(kmodel.cluster_centers_,columns=[typelabel[keys[i]]]) #计算出每一列属性中每一类的聚类中心
    print(r1)
    r2 = pd.Series(kmodel.labels_).value_counts()  #分类统计 表示每一类有多少个数据

    r2 = pd.DataFrame(r2,columns=[typelabel[keys[i]]+'n'])

    r = pd.concat([r1,r2],axis=1).sort_index() #将r1和r2连接起来

    r.index=[1,2,3]
    r[typelabel[keys[i]]] = pd.rolling_mean(r[typelabel[keys[i]]],2)  #计算相邻两列的均值  以此作为边界点
    r[typelabel[keys[i]]][1] = 0.0  #将原来的聚类中心改为边界点
    result = result.append(r.T)

result = result.sort_index()
result.to_csv(processedfile)