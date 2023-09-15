import warnings
import numpy as np
import pandas as pd
import plotly.express as px
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori,association_rules,fpgrowth
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
df = pd.read_csv('tianchi_mobile_recommend_train_user.csv')

#region 一、基本数据统计/预处理
## 查看数据信息统计（数据类型）
df.info()

## 数据清洗\处理
#1.是否存在缺失值？
df.isna().sum()
df.drop(['user_geohash'],axis=1,inplace=True)   #删除user_geohas特征
#2.重复值的考虑(这里未考虑去重，因为时间细粒度为hour)（可以进行挖掘：比如根据用户浏览了多少次商品/同类商品后才下单--进行聚类）
df_diplicated = df[df.duplicated(keep=False)==1]
df_diplicated.groupby(['user_id','item_id','behavior_type','item_category'])['time'].count() #用户对某个商品在一小时内行为的次数
# df.drop_duplicates(inplace=True)
#3.时间特征处理
df['date'] = df['time'].str.split().str[0]
df['date'] = pd.to_datetime(df['date'])
weekdayMap = {1:'周二',2:'周三',3:'周四',4:'周五',5:'周六',6:'周日',0:'周一'} #国外的第一天是从周日开始算的
df['weekday'] = df['date'].dt.dayofweek.map(weekdayMap)
df['date'] = df['date'].astype(str)
df['hour'] = df['time'].str.split().str[1]
# df['date'].max()    # (2014,11,18)
# df['date'].min()    # (2014,12,18)
#endregion

#region 二、趋势性分析（pv（使用淘宝的次数）和uv（使用淘宝的人数））
## 1.按天数：
pu_daily = pd.DataFrame(df.groupby(['date'])['user_id'].count())
pu_daily['uv'] = df.groupby(['date'])['user_id'].nunique()
pu_daily.columns=['pv','uv']

fig,ax = plt.subplots(2,1,sharex=True)
ax[0].plot(pu_daily['pv'],color='red',label='pv')
ax[1].plot(pu_daily['uv'],color='blue',label='uv')
ax[0].set_title('31日pv、uv趋势图')
ax[1].xaxis.set_tick_params(rotation=30,labelsize=7)
ax[1].xaxis.set_ticks([x for x in pu_daily.index if int(str(x)[-1])%2==0])
ax[0].legend()
ax[1].legend()
plt.show(block=True)
## 2.按周：
pu_weekly = pd.DataFrame(df.groupby(['weekday'])['user_id'].count())
pu_weekly['uv'] = df.groupby(['weekday'])['user_id'].nunique()
pu_weekly.columns=['pv','uv']
pu_weekly.sort_values(['pv'],ascending=False,inplace=True)

fig_week,ax_week = plt.subplots()
ax_week.bar(pu_weekly.index,pu_weekly['pv'],label='pv',color='red')
ax_week2 = ax_week.twinx()
ax_week2.plot(pu_weekly['uv'],label='uv',color='blue')
ax_week.legend()
ax_week2.legend()
ax_week.set_title('周记pv、uv趋势图')
plt.show(block=True)
## 3.按小时：
pu_hourly = pd.DataFrame(df.groupby(['hour'])['user_id'].count())
pu_hourly['uv'] = df.groupby(['hour'])['user_id'].nunique()
pu_hourly.columns=['pv','uv']

## 双12那天的购买习惯是否会因为节日而改变
day12 = df[df['date'] == '2014-12-12']
without_day12 = df[df['date']!='2014-12-12']

pu_hourly_day12 = pd.DataFrame(day12.groupby(['hour'])['user_id'].count())
pu_hourly_day12['uv'] = day12.groupby(['hour'])['user_id'].nunique()
pu_hourly_day12.columns=['pv','uv']

pu_hourly_withoutday12 = pd.DataFrame(without_day12.groupby(['hour'])['user_id'].count())
pu_hourly_withoutday12['uv'] = without_day12.groupby(['hour'])['user_id'].nunique()
pu_hourly_withoutday12.columns=['pv','uv']

fig_hour,ax = plt.subplots(2,2)
ax[0,0].plot(pu_hourly_withoutday12['pv'],color='red',label='pv')
ax[1,0].plot(pu_hourly_day12['pv'],color='red',label='pv')
ax[0,1].plot(pu_hourly_withoutday12['uv'],color='blue',label='uv')
ax[1,1].plot(pu_hourly_day12['uv'],color='blue',label='uv')
ax[0,0].set_xticks([])
ax[0,1].set_xticks([])
ax[1,0].set_xticks(np.arange(0,24,2))
ax[1,1].set_xticks(np.arange(0,24,2))
ax[0,0].set_title('pv')
ax[0,1].set_title('uv')
ax[0,0].set_ylabel('without_day12',color='peru')
ax[1,0].set_ylabel('day12',color='peru')
ax[0,0].legend()
ax[0,1].legend()
ax[1,0].legend()
ax[1,1].legend()
plt.suptitle('pv、uv in 12-12 or not',fontsize=20)
fig_hour.tight_layout()
plt.show(block=True)
#endregion

#region 三、转化率分析
transfer_all = pd.DataFrame(df['behavior_type'].value_counts(),columns=['count'])
transfer_day12 = pd.DataFrame(day12['behavior_type'].value_counts(),columns=['count'])
transfer_without_day12 = pd.DataFrame(without_day12['behavior_type'].value_counts(),columns=['count'])
#将收藏和加入购物车视为一类：对商品感兴趣
transfer_all.loc[5] = transfer_all.loc[2]+transfer_all.loc[3]
transfer_day12.loc[5] = transfer_day12.loc[2]+transfer_day12.loc[3]
transfer_without_day12.loc[5] = transfer_without_day12.loc[2]+transfer_without_day12.loc[3]
transfer_all.drop([2,3],inplace=True)
transfer_day12.drop([2,3],inplace=True)
transfer_without_day12.drop([2,3],inplace=True)
transfer_all.sort_values(by='count',ascending=False,inplace=True)
transfer_day12.sort_values(by='count',ascending=False,inplace=True)
transfer_without_day12.sort_values(by='count',ascending=False,inplace=True)
stages = ['visit','cart/favor','purchase']
transfer_day12.index=stages
transfer_without_day12.index=stages
#计算转化率
transfer_day12['rate'] = 1
transfer_without_day12['rate'] = 1
transfer_day12.loc['cart/favor','rate'] = transfer_day12['count'][1]/transfer_day12['count'][0]
transfer_without_day12.loc['cart/favor','rate'] = transfer_without_day12['count'][1]/transfer_without_day12['count'][0]
transfer_day12.loc['purchase','rate'] = transfer_day12['count'][2]/transfer_day12['count'][1]
transfer_without_day12.loc['purchase','rate'] = transfer_without_day12['count'][2]/transfer_without_day12['count'][1]

plt.barh(stages[::-1],transfer_day12['rate'][::-1],color='green',height=0.5)
plt.barh(stages[::-1],-transfer_without_day12['rate'][::-1],color='yellow',height=0.5)
plt.legend(('day12','without day12'))
plt.title('转化率漏斗图')
plt.show(block=True)
print('一个月的总体转化：{}'.format(transfer_all))
print('除双12的转化：{}'.format(transfer_without_day12))
print('双12的转化：{}'.format(transfer_day12))

## 四.哪种商品是最受欢迎的（此处表现为购买次数最多的）
category_cnt = pd.DataFrame(df['item_category'].value_counts(),columns=['count'])
category_cnt['rate'] = category_cnt['count'].cumsum()/category_cnt['count'].sum()
pivot_category_type = pd.DataFrame(df.groupby(['item_category','behavior_type'])['user_id'].count())
pivot_category_type.columns = ['cnt']
pivot_category_type = pivot_category_type.unstack().fillna(0).astype('int')
pivot_category_type.sort_values([('cnt',4)],ascending=False,inplace=True)
plt.plot(list(pivot_category_type[('cnt',4)]),label='购买')
# plt.plot(list(pivot_category_type.iloc[:,2]),label='购物车')
# plt.plot(list(pivot_category_type.iloc[:,1]),label='收藏')
# plt.plot(list(pivot_category_type.iloc[:,0]),label='浏览')
plt.legend()
plt.show(block=True)
# pivot_category_type[('cnt',4)].reset_index().to_csv('pivot_category_type.csv')

#region 五.RFM模型（这里只考虑了购买过商品的用户）和聚类分析
###5.1 数据收集
df['date'] = pd.to_datetime(df['date']).dt.date
payInfo = df[df['behavior_type']==4][['user_id','date']]            #购买用户信息
unpayInfo = df[~df['user_id'].isin(payInfo.user_id)][['user_id']]   #未曾购买用户信息。注意~的使用。！！！
# payInfo.user_id.nunique()     #8886
# unpayInfo.user_id.nunique()   #1114
plt.pie(x=[payInfo.user_id.nunique(),unpayInfo.user_id.nunique()],labels=['pay','unpay'])
# plt.legend(RF['Label of Customer'].value_counts().index,bbox_to_anchor=(1, 1), loc='best', borderaxespad=0.7)
plt.title('购买/未购买用户类型占比')
plt.show(block=True)

#购买用户的RFM
R = pd.DataFrame(payInfo['date'].max()-payInfo.groupby(['user_id'])['date'].max())
R.columns=['R']
R['R'] = R['R'].astype('str').str.split().str[0]
R['R'] = R['R'].map(lambda x: 0 if x=='0:00:00' else x)
F = pd.DataFrame(payInfo.groupby(['user_id'])['user_id'].count())
F.columns = ['F']
pRF = pd.merge(R,F,left_on=R.index,right_on=F.index)
pRF.rename(columns={'key_0':'user_id'},inplace=True)
#未曾购买用户的RFM
upRF = pd.DataFrame(unpayInfo['user_id'].unique(),columns=['user_id'])
upRF['R'] = 31
upRF['F'] = 0
#合并
RF = pd.concat([pRF,upRF])
RF[['R','F']] = RF[['R','F']].astype(int)
RF.info()

###5.2 根据定义进行分类
RF['R_Label'] = pd.qcut(RF['R'],2)
RF['F_Label'] = pd.qcut(RF['F'],2)
from sklearn.preprocessing import LabelEncoder
RF['R_Label']=LabelEncoder().fit_transform(RF['R_Label'])
RF['F_Label']=LabelEncoder().fit_transform(RF['F_Label'])
def getRFMLabel(r,f):
    if (r==0)&(f==1):
        return '价值客户'
    if (r==0)&(f==0):
        return '发展客户'
    if (r==1)&(f==1):
        return '保持客户'
    if (r==1)&(f==0):
        return '挽留客户'
RF['Label of Customer'] = RF.apply(lambda x:getRFMLabel(x['R_Label'], x['F_Label']),axis=1)
RF['R_Label'] = np.where(RF['R_Label'] == 0, '高', '低')
RF['F_Label'] = np.where(RF['F_Label'] == 1, '高', '低')
# 绘图
plt.figure(figsize=(10, 7))
plt.pie(x=RF['Label of Customer'].value_counts().values,labels=RF['Label of Customer'].value_counts().index,
        autopct='%.2f%%',explode=[0.1,0.1,0,0],colors=['deepskyblue','steelblue','lightskyblue','aliceblue'],wedgeprops={'linewidth':0.5,'edgecolor':'black'},
textprops={'fontsize':12,'color':'black'})
plt.legend(RF['Label of Customer'].value_counts().index,bbox_to_anchor=(1, 1), loc='best', borderaxespad=0.7)
plt.title('用户类型占比')
plt.show(block=True)

###5.3 根据评分打分
# 通过RF对用户进行评级
RF[['R','F']].describe()
RF['R'].value_counts()
RF['F'].value_counts()
# plt.plot(RF['R'])
# plt.plot(RF['F'])
# plt.show(block=True)
RF['R_score'] = pd.cut(RF['R'],bins=[0,7,14,31,32],right=False,labels=[4,3,2,1])
RF['F_score'] = pd.cut(RF['F'],bins=[0,4,8,17,810],right=False,labels=[1,2,3,4])    #将购买用户的分位数作为了分割点
RF['score'] = RF.apply(lambda x:x['R_score']+2*x['F_score'],axis=1)     #这里设置了权重：F是R的两倍（根据业务实际调整，这里只是个例子）

###5.4计算各个用户的转化率
conv = pd.DataFrame(df[df['user_id'].isin(payInfo.user_id)][['user_id','behavior_type']].groupby(['user_id'])['behavior_type'].apply(lambda x:sum(x==4)/len(x))).reset_index()
conv.columns=['user_id','conv']
RF = pd.merge(RF,conv,how='left',on='user_id')
RF['conv']= RF['conv'].fillna(0)
RF['conv'].describe()
def getConv(x):
    if x==0:
        return 1
    elif x<0.05:
        return 2
    elif x<=0.01:
        return 3
    elif x<=0.02:
        return 4
    else:
        return 5
RF['conv'] = RF['conv'].apply(lambda x:getConv(x))

###5.5聚类分析
#标准化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
ss = scaler.fit_transform(RF[['score','conv']])
#K-Means聚类
from sklearn.cluster import KMeans
#碎石图
inertia=[]
for i in range(1,11):
    km=KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    km.fit(ss)
    inertia.append(km.inertia_)
plt.plot(range(1,11),inertia,'*-')
plt.title('k-means碎石图')
plt.show(block=True)

Kmeans=KMeans(n_clusters=4)                        # 建立KMean模型
Kmeans.fit(ss)                                     # 训练模型
k_char=Kmeans.cluster_centers_                     # 得到每个分类的质心
personas=pd.DataFrame(k_char.T,index=['score','conv'],)  # 用户画像表
print(personas)
#热力图
import seaborn as sns
fig,ax = plt.subplots(figsize=(8, 4))
sns.heatmap(personas, xticklabels=True, yticklabels=True, square=False, linewidths=.5, annot=True, cmap="YlGnBu")
plt.title('heatmap of kmeans')
plt.show(block=True)
#轮廓系数
from sklearn import metrics
score = metrics.silhouette_score(ss,Kmeans.predict(ss))
print('聚类个数为4时，轮廓函数:' , score)  #0.6975
#endregion

### 六.双12当天关联分析（通过双12当天的商品关联度挖掘折扣下用户购买商品的关联度，进而在平时可以实施定制化推荐）
# 1.绘制双12当天商品数图
day12.info()
day12['item_id'] = day12['item_id'].astype(str)
day12['item_category'] = day12['item_category'].astype(str)
day12['tool'] = 1
# itemCnt = pd.DataFrame(day12['item_id'].value_counts()).reset_index()
# treeColors = itemCnt.style.background_gradient(cmap='Blues')
# treefig = px.treemap(itemCnt.head(50),values='count')
# 2.生成数据
x = day12[day12['behavior_type']==4].groupby(['user_id'])['item_category'].transform(lambda x:','.join(x))
concat = pd.concat([day12[day12['behavior_type']==4][['user_id']],x],axis=1)
# concat = concat.drop_duplicates(['user_id','item_category'])
concat.info()

#2.AP和FP的分析
dt = list(concat['item_category'])
for i in range(len(dt)):
    dt[i] = dt[i].split(',')
te = TransactionEncoder()
dt_te = te.fit(dt).transform(dt)
dt_te_df = pd.DataFrame(dt_te,columns=te.columns_)    #由于ap算法需要是DF的数据格式，这里进行转换

## AP
frequent_itemsets = apriori(dt_te_df,min_support=0.02,use_colnames=True)#使用列的名字，否则使用序号

##FP
# frequent_itemsets = fpgrowth(dt_te_df,min_support=0.1,use_colnames=True)
apRes = association_rules(frequent_itemsets,metric='confidence',min_threshold=0.5)
print(apRes)
