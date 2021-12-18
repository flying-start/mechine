#!/usr/bin/env python
# coding: utf-8

# In[2]:


#导入数据
import pandas as pd
pd.set_option('display.max_rows',None)
test= pd.read_csv("test.csv")
train=pd.read_csv("train.csv")
train["timestamp"]=train["timestamp"].astype(str)
train


# In[ ]:


#轨迹提取
def trans_data(data):
    S=list()  #存储轨迹点序列
    q=list()  #存储一个轨迹点
    Q=[]
    j=0
    length=data["mmsi"].size #获取轨迹点个数
    while j<length:
    q=data.loc[j]  #获取一个轨迹点，并加入到q
    S.append(q)     #将轨迹点加入到轨迹序列中
    #q.append(train.loc[j])
    q=[]
    j=j+1
    trace=list()  #用来存储轨迹
    i=1
    q=[]    #用来暂存一条轨迹
    while i<length:
    #如果两个轨迹点的前后时间差小于两小时，则认为是同一条轨迹，放入同一条轨迹序列
        if int(S[i]["timestamp"])-int(S[i-1]["timestamp"])<2*3600: 
            q.append(S[i]) #将轨迹点加入到同一轨迹序列中
        else:
            trace.append(q)  #将一条轨迹加入到轨迹序列
            q=[]   #重新划分一条轨迹
        i=i+1
    trace
    


# In[6]:



#轨迹可视化
import folium
def show_single_trajectory(i):
        data = data_trace[data_trace["traceid"]==i]
        loc = data[['lat', 'lon']]
        begin_time, end_time = data['timestamp'].min(), data['timestamp'].max()
        m = folium.Map(location=[loc.mean()[0], loc.mean()[1]], zoom_start=10)
        begin_lat, begin_lng = loc.iloc[0, 0], loc.iloc[0, 1]  # 起点经纬度
        end_lat, end_lng = loc.iloc[-1, 0], loc.iloc[-1, 1]  # 终点经纬度
        folium.Marker(location = [begin_lat, begin_lng], popup='begin_time:{}, lat:{}, lon:{}'.format(begin_time, begin_lat, begin_lng), tooltip='traceid: {}'.format(int(i))).add_to(m)
        folium.Marker(location = [end_lat, end_lng], popup='end_time:{}, lat:{}, lon:{}'.format(end_time, end_lat, end_lng), tooltip='traceid: {}'.format(int(i)), icon=folium.Icon(color='green')).add_to(m)
        my_PolyLine1= folium.PolyLine(locations=[[loc.min()[0],loc.min()[1]],[loc.min()[0],loc.max()[1]],[loc.max()[0],loc.max()[1]],[loc.max()[0],loc.min()[1]],[loc.min()[0],loc.min()[1]]], tooltip='traceid: {}'.format(int(i)), weight=5)
        my_PolyLine2= folium.PolyLine(locations=loc.values, tooltip='traceid: {}'.format(int(i)), weight=5)
#         my_PolyLine3= folium.PolyLine(locations=loc.values, tooltip='traceid: {}'.format(int(id)), weight=5)
        m.add_child(my_PolyLine1)
        m.add_child(my_PolyLine2)
        m.add_child(folium.LatLngPopup())
        return m
    


# In[143]:


#绘制轨迹折线图
def draw_p(s):
    import matplotlib.pyplot as plt
    i=0
    ALL_Color = [[1, 0, 0, 1], [0, 252 / 255, 0, 1], [0, 0, 1, 1], [46 / 255, 217 / 255, 1, 1],
             [1, 156 / 255, 85 / 255, 1], [1, 51 / 255, 129 / 255, 1],
              [186 / 255, 12 / 255, 1, 1],   [0, 0, 0, 1]]
#     s=[25, 118, 119, 122, 123]
    while i<len(s):
        data=s[s["traceid"]==i]
        x=data["lat"]
        y=data["lon"]
        plt.plot(x,y,label='trace_test'+str(i),linewidth=1,color=ALL_Color[i%8])
        plt.xlabel('lat')
        plt.ylabel('Lon')
        plt.title('trace_test')
        plt.legend()
        plt.show()
        i=i+1
    


# In[ ]:


# 计算每个轨迹的区域编码，num为网格规模num*num
def code(trace，num):
    j=0
    a=[]  #存储每一个轨迹点的网格编号（index_lat,index_lon）
    c=[]  #存储一个轨迹的网格编号
    L=[]  #存储所以轨迹的编号序列
    num_lat=(max_lat-min_lat)/num  #宽度
    num_lon=(max_lat-min_lat)/num   #长度
    lat_O=min_lat-0.0000000000001   #原点
    lon_O=min_lon-0.0000000000001   #原点
    i=0
    while i<len(trace):
        j=0
        while j<len(trace[i]): 
            index_lat=int((trace[i][j]["lat"]-lat_O)/num_lat)
            index_lat
            index_lon=int((trace[i][j]["lon"]-lon_O)/num_lon)
            index_lon
            a.append([index_lat,index_lon])
            if a[0] not in c:
                c.append([index_lat,index_lon])  #将获得的不重复网格区域编号加入区域序列
            j=j+1
            a=[]
        L.append(c)
        i=i+1
        c=[]
L


# In[72]:


#判断轨迹是否相似
def issimary(a,b):
    len1=len(a)
    len2=len(b)
    min_len=min(len1,len2)
    max_len=max(len1,len2)
    count=0
    if int(min_len) < 0.8*int(max_len): 
        return False 
    for i in range(len1):
        if a[i] in b:
            count=count+1
    if (count/len1)>0.8 and(count/len2)>0.8:
        return True
    else:
        return False


# In[116]:


def fenlei(L):
#轨迹分类
    i=0
    j=0
    lab=[]
    Q=[]
    while i<len(L):
        j=0
        lab=[]
        while j<len(L):
            if issimary(L[i],L[j]):
                lab.append(j)  #表示和L[i]类别一样的
            j=j+1
        Q.append(lab)
        i=i+1
    return Q

    
          
    


# In[258]:


#使用kmeans进行聚类
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
class K_Means(object):
    #初始化，参数n_clusters(即聚成几类，K)、max_iter（迭代次数）、centroids（初始质心）
    def __init__(self,n_clusters=6,max_iter=300,centroids=[]):
        self.n_clusters=n_clusters
        self.max_iter=max_iter
        self.centroids=np.array(centroids)  #数组转换为矩阵
    
    #训练模型方法，K-Means聚类过程，传入原始数据
    def fit(self,data):
        #假如没有指定初始质心，就随机选取data中的点作为初始质心（即x，x就包含了一个点的两个坐标）
        if(self.centroids.shape==(0,)):  #0行矩阵
            #从data中随机生成0-data行数的6个整数作为索引值
            #random.randint(a,b,c)方法随机生成一个整数，从a到b，生成c个
            self.centroids=data[np.random.randint(0,data.shape[0],self.n_clusters),:] #data.shape[0]为data行数，生成self.n_clusters个即6个
            
        #开始迭代
        for i in range(self.max_iter):
            #1.计算相似度，
            distances=cdist(data,self.centroids)   #cdist()只要要求同维度就可以
            
           
            c_index=np.argmin(distances,axis=1) 
            
           
            for i in range(self.n_clusters):
                 #
                if i in c_index: 
                    data[c_index==i]
                    self.centroids[i]=np.mean(data[c_index==i],axis=0)  
                    
    def predict(self,samples): 
        #跟上面一样，
        distances=cdist(samples,self.centroids)
        c_index=np.argmin(distances,axis=1)
        return c_index


# In[3]:

------------------------------------------------------------------杂项---------------------------------------------
S=list()  #存储轨迹点序列
q=list()  #存储一个轨迹点
Q=[]
j=0
length=train["mmsi"].size #获取轨迹点个数
while j<length:
    q=train.loc[j]  #获取一个轨迹点，并加入到q
    S.append(q)     #将轨迹点加入到轨迹序列中
    #q.append(train.loc[j])
    q=[]
    j=j+1
trace=list()  #用来存储轨迹
i=1
q=[]    #用来暂存一条轨迹
while i<length:
    #如果两个轨迹点的前后时间差小于两小时，则认为是同一条轨迹，放入同一条轨迹序列
    if int(S[i]["timestamp"])-int(S[i-1]["timestamp"])<2*3600: 
        q.append(S[i]) #将轨迹点加入到同一轨迹序列中
    else:
        trace.append(q)  #将一条轨迹加入到轨迹序列
        q=[]   #重新划分一条轨迹
    i=i+1
trace
i=1
data_trace=pd.DataFrame(trace[0]) #将序列转成dataFrame
data_trace["traceid"]=0
while i<len(trace):
    data1=pd.DataFrame(trace[i])
    data1["traceid"]=i
    data_trace=pd.concat([data_trace,data1]) #将多个dataFrame合并
    data1=pd.DataFrame()
    i=i+1
data_trace
# data_trace["traceid"]=0
# data2=pd.DataFrame(trace[1])
# data2["traceid"]=1
# data_trace=pd.concat([data_trace,data2])
# data_trace
# i=0
# j=0

# while i<len(trace):
#     while j<len(q):
#         if trace[i][j]["Sog"]==0:
#             Q.append(trace[i][j])
#         j=j+1
#     i=i+1
# print(Q)


# In[251]:


len(trace)


# In[109]:


import matplotlib.pyplot as plt
x1=data_trace[data_trace["traceid"]==1]
x2=data_trace[data_trace["traceid"]==2]
x=x1["lat"]
y=x1["lon"]
x1=x2["lat"]
y1=x2["lon"]
# my_x_ticks=x
# my_y_ticks=y
plt.plot(x,y,label='Frist trace',linewidth=1,color='g')
plt.plot(x1,y1,label='second trace',linewidth=1,color='b')
plt.xlabel('lat')
plt.ylabel('Lon')
plt.title('trace')
plt.legend()
plt.show()


# In[95]:



import matplotlib.pyplot as plt
i=0
ALL_Color = [[1, 0, 0, 1], [0, 252 / 255, 0, 1], [0, 0, 1, 1], [46 / 255, 217 / 255, 1, 1],
             [1, 156 / 255, 85 / 255, 1], [1, 51 / 255, 129 / 255, 1],
              [186 / 255, 12 / 255, 1, 1],   [0, 0, 0, 1]]
while i<len(trace):
    data=data_trace[data_trace["traceid"]==i]
    x=data["lat"]
    y=data["lon"]
    plt.plot(x,y,label='trace_'+str(i),linewidth=1,color=ALL_Color[i%8],marker='*')
    plt.xlabel('lat')
    plt.ylabel('Lon')
    plt.title('trace')
    plt.legend()    
    i=i+1
    plt.show()


    


# In[3]:


S=list()
q=list()
Q=list()
j=0
length=train["mmsi"].size
while j<length:
    q=train.loc[j]
    print(q)
    S.append(q)
    q=[]
    j=j+1


# In[5]:


min_lat=train.agg({"lat":"min"})
max_lat=train.agg({"lat":"max"})
min_lat
max_lat
max_lon=train.agg({"lon":"max"})
min_lon=train.agg({"lon":"min"})
min_lon
q=list()
q.append(min_lat)
q.append(max_lat)
q.append(min_lon)
q.append(max_lon)
q
max_lon=121.88135
min_lon=121.652935
min_lat=30.335862
max_lat=30.583262


# In[5]:


import folium
m = folium.Map([(max_lat+min_lat)/2, (min_lon+max_lon)/2], # 中心区域经纬度
               zoom_start=9, # 默认放大倍数
              ) 
def drow_m(locations,weight,color,opacity):
    route = folium.PolyLine(
        locations,
        weight=weight,
        color=color,
        opacity=opacity
    ).add_to(m)
location=[[30.335862,121.652935],[30.335862,121.88135],[30.583262, 121.88135],[30.583262,121.652935],[30.335862,121.652935]]
drow_m(location,3,'blue',0.8)
m


# In[166]:


show_single_trajectory(1)


# In[ ]:


# 绘制网格，这里按照20*20共400个网格
def get_polygons(latitude_num,longitude_num):
    latitude_step = (max_lat - min_lat)/latitude_num
    longitude_step = (max_lon -min_lon)/longitude_num
    polygons = []
    for i in range(latitude_num):
        latitude_right = min_lat + latitude_step * (i+1)
        polygons.append([[max_lon,latitude_right],[min_lon,latitude_right]])
    for j in range(longitude_num):
        longitude_down = max_lon - longitude_step * (j+1)        
        polygons.append([[longitude_down,min_lat],[longitude_down,max_lat]])     
    
    return polygons

polygons = get_polygons(20,20)
for polygon in polygons:
    drow_m(polygon,1,'black',0.8)
m


# In[115]:


# 存储每个轨迹的区域编码
j=0
a=[]  #存储每一个轨迹点的网格编号（index_lat,index_lon）
c=[]  #存储一个轨迹的网格编号
L=[]  #存储所以轨迹的编号序列
num_lon=0.11420749999999914  #宽度
num_lat=0.12370000000000126   #长度
lat_O=min_lat-0.0000000000001   #原点
lon_O=min_lon-0.0000000000001   #原点
i=0
while i<len(trace):
    j=0
    while j<len(trace[i]): 
        index_lat=int((trace[i][j]["lat"]-lat_O)/num_lat)
        index_lat
        index_lon=int((trace[i][j]["lon"]-lon_O)/num_lon)
        index_lon
        a.append([index_lat,index_lon])
        if a[0] not in c:
            c.append([index_lat,index_lon])  #将获得的不重复网格区域编号加入区域序列
        j=j+1
        a=[]
    L.append(c)
    i=i+1
    c=[]
L
# L[1]
# o=len(L[1])
# o
# o=len(L)
# o
# b=list()
# a=list()
# j=0
# while j<len(trace[0]):
#     index_lat=int((trace[0][j]["lat"]-lat_O)/num_lat)
#     index_lat
#     index_lon=int((trace[0][j]["lon"]-lon_O)/num_lon)
#     index_lon
#     a.append([index_lat,index_lon])
#     if a[0] not in b:
#         b.append(a)
#     j=j+1
#     a=[]
# b    
# len1=len(L[1])
# len2=len(b)
# len1
# len2

# min_len=min(len1,len2)
# max_len=max(len1,len2)
# if int(min_len) < 0.8*int(max_len): 
#     d=0 
# count=0
# for i in range(len1):
#     if L[1][i] in b:
#         count=count+1
# if (count/len1)>0.8 and(count/len2)>0.8:
#     d=1
# else:
#     d=0
# count    


# In[9]:


#轨迹折线图
import matplotlib.pyplot as plt
i=0
ALL_Color = [[1, 0, 0, 1], [0, 252 / 255, 0, 1], [0, 0, 1, 1], [46 / 255, 217 / 255, 1, 1],
             [1, 156 / 255, 85 / 255, 1], [1, 51 / 255, 129 / 255, 1],
              [186 / 255, 12 / 255, 1, 1],   [0, 0, 0, 1]]
while i<8:
    data=data_trace[data_trace["traceid"]==i]
    x=data["lat"]
    y=data["lon"]
    plt.plot(x,y,label='trace_'+str(i),linewidth=1,color=ALL_Color[i%8])
    plt.xlabel('lat')
    plt.ylabel('Lon')
    plt.title('trace')
    plt.legend()    
    i=i+1
plt.show()


# In[141]:


len(trace_test)


# In[140]:


len2


# In[132]:


S1=list()
q1=[]
j=0
length=test["mmsi"].size
while j<length:
    q1=test.loc[j]
    S1.append(q1)
    q1=[]
    j=j+1
trace_test=list()
i=1
q2=[]
while i<length:
    if int(S1[i]["timestamp"])-int(S1[i-1]["timestamp"])<2*3600:
        q2.append(S1[i])
    else:
        trace_test.append(q2)
        q2=[]
    i=i+1
len(trace_test)
i=1
data_trace2=pd.DataFrame(trace_test[0]) #将序列转成dataFrame
data_trace2["traceid"]=0
while i<len(trace_test):
    data1=pd.DataFrame(trace_test[i])
    data1["traceid"]=i
    data_trace2=pd.concat([data_trace2,data1]) #将多个dataFrame合并
    data1=pd.DataFrame()
    i=i+1
data_trace2


# In[266]:


import matplotlib.pyplot as plt
i=0
ALL_Color = [[1, 0, 0, 1], [0, 252 / 255, 0, 1], [0, 0, 1, 1], [46 / 255, 217 / 255, 1, 1],
             [1, 156 / 255, 85 / 255, 1], [1, 51 / 255, 129 / 255, 1],
              [186 / 255, 12 / 255, 1, 1],   [0, 0, 0, 1]]
while i<72:
    data=data_trace1[data_trace1["traceid"]==i]
    x=data["lat"]
    y=data["lon"]
    plt.plot(x,y,label='trace_test'+str(i),linewidth=1,color=ALL_Color[i%8])
    plt.xlabel('lat')
    plt.ylabel('Lon')
    plt.title('trace_test')
    plt.legend()    
    i=i+1
plt.show()


# In[117]:


result=fenlei(L)
result=pd.DataFrame(result)
# result2=pd.DataFrame(result.values.T, index=result.columns, columns=result.index)
# result2
result.T


# In[144]:


draw_p(data_trace2)


# In[ ]:


result=issimary(L[1],L[2])
result


# In[98]:


j=0
a=[]
c2=[]
L2=[]
num_lon=0.011420749999999914
num_lat=0.012370000000000126
lat_O=min_lat-0.0000000000001
lon_O=min_lon-0.0000000000001
i=0
while i<len(trace_test):
    j=0
    while j<len(trace_test[i]): 
        index_lat=int((trace_test[i][j]["lat"]-lat_O)/num_lat)
        index_lat
        index_lon=int((trace_test[i][j]["lon"]-lon_O)/num_lon)
        index_lon
        a.append([index_lat,index_lon])
        if a[0] not in c2:
            c2.append([index_lat,index_lon])
        j=j+1
        a=[]
    L2.append(c2)
    i=i+1
    c=[]
L3=pd.DataFrame(L)
L3


# In[12]:


import numpy as np
x_data1=np.array(L,dtype=object)
x_data1.shape


# In[ ]:


import matplotlib.pyplot as plt
i=0
ALL_Color = [[1, 0, 0, 1], [0, 252 / 255, 0, 1], [0, 0, 1, 1], [46 / 255, 217 / 255, 1, 1],
             [1, 156 / 255, 85 / 255, 1], [1, 51 / 255, 129 / 255, 1],
              [186 / 255, 12 / 255, 1, 1],   [0, 0, 0, 1]]
s=[25, 118, 119, 122, 123]
while i<4:
    data=data_trace[data_trace["traceid"]==s[i]]
    x=data["lat"]
    y=data["lon"]
    plt.plot(x,y,label='trace_test'+str(s[i]),linewidth=1,color=ALL_Color[s[i]%8])
    plt.xlabel('lat')
    plt.ylabel('Lon')
    plt.title('trace_test')
    plt.legend()
    i=i+1
plt.show()


# In[13]:


import numpy as np
x_data=np.array(L2,dtype=object)
x_data.shape


# In[254]:


dt=pd.read_csv("train.csv")
dt1=dt[["lat","lon"]]
dat=(dt1.values)
dat


# In[259]:


import numpy as np
import matplotlib.pyplot as plt  #画图
#定义一个绘制子图函数
def plotKMeans(x,y,centroids,subplot,title):    #x,y为样本点坐标；centroids为质心点；subplot子图编号
    #分配子图
    plt.subplot(subplot) #根据传进来的子图编号分配子图
    plt.scatter(x[:,0],x[:,1],c='g')
    #画出质心点
    plt.scatter(centroids[:,0],centroids[:,1],c=np.array(range(6)),s=100)  #c=np.array(range(6)按照不同颜色生成初始质心点，s指size
    plt.title(title)
    
#创建一个kmeans对象实例
kmeans=K_Means(max_iter=300) #6个初始质心点
kmeans.fit(dat)
plt.figure(figsize=(16,6))
plotKMeans(dat,y,kmeans.centroids,121,'Initial_State')  #121表示一行两列的子图中的第一个

# 开始聚类
kmeans.fit(dat)
plotKMeans(dat,y,kmeans.centroids,122,'Final_State')
# kmeans.centroids
#预测新数据点的类
# 别
# x_new=np.array([[0,0],[10,7]])  #二维数组
y_pred=kmeans.predict(dat)
print(kmeans.centroids)
print(y_pred)

# plt.scatter(x_new[:,0],x_new[:,1],s=100,c='black')


# In[167]:


data4=pd.read_csv("test.csv")
data6=data4.groupby('mmsi').mean()
data5=data4[["mmsi","lat","lon"]]
train["mmsi"].astype(str)
train["lat"].astype(float)
train["lon"].astype(float)
# outputpath='C:\\Users\\曾惠冰\\Desktop\\predict1.csv'
# data7=data6[["lat","lon"]]
# data7.to_csv(outputpath,sep=',',index=False,header=False)
data5


# In[164]:


dfg=pd.read_csv("submission.csv")
# dfg=dfg.head()
dfg["mmsi"].astype(str)
dfg["lat"].astype(float)
dfg["lon"].astype(float)
outputpath='C:\\Users\\曾惠冰\\Desktop\\predict2.csv'
dfg.to_csv(outputpath,sep=',',index=False,header=False)
# dfg

