#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df=pd.read_csv("C:\\Users\\sasidharreddy\\Downloads\\sales_data.csv")


# In[3]:


df.head(10)


# In[4]:


df.columns


# In[5]:


import datetime


# In[6]:


l=pd.to_datetime(df['FIRST_ORDER_DATE'])
l1=pd.to_datetime(df['LATEST_ORDER_DATE'])


# In[7]:


df['No.of.Days']=l1-l


# In[8]:


df.drop(['FIRST_ORDER_DATE','LATEST_ORDER_DATE'],axis=1, inplace=True)


# In[9]:


df


# In[10]:


df['No.of.Days']=df['No.of.Days'].apply(lambda x: x.days)


# In[11]:


from sklearn.cluster import KMeans


# In[12]:


x=df.drop(['CustomerID'],axis=1).values


# In[13]:


x.shape


# Scaling the data

# In[14]:


from sklearn.preprocessing import StandardScaler


# In[15]:


scaler = StandardScaler()


# In[16]:


x=scaler.fit_transform(x)


# Elbow method to find no.of clusters

# In[17]:


from scipy.spatial.distance import cdist


# In[18]:


distortions = []
inertias = []
mapping1 = {}
mapping2 = {}
K = range(1, 10)

for k in K:
    # Building and fitting the model
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(x)
    
    distortions.append(sum(np.min(cdist(x, kmeanModel.cluster_centers_,'euclidean'), axis=1)) / x.shape[0])
    
    inertias.append(kmeanModel.inertia_)
    
    mapping1[k] = sum(np.min(cdist(x, kmeanModel.cluster_centers_,'euclidean'), axis=1)) / x.shape[0]
    
    mapping2[k] = kmeanModel.inertia_


# In[19]:


plt.plot(K, distortions, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Distortion')
plt.title('The Elbow Method using Distortion')
plt.show()


# We found that optimal no.of clusters is 5

# In[20]:


kmeanModel = KMeans(n_clusters=5, random_state=53)
kmeans=kmeanModel.fit(x)


# In[21]:


df['Labels']=kmeans.labels_


# In[22]:


h1=['Label0','Label1','Label2','Label3','Label4']


# In[455]:


h=['REVENUE','TOTAL_ORDERS','AVGDAYSBETWEENORDERS','DAYSSINCELASTORDER']


# In[451]:


def va(df):
    v1=[]
    for i in range(5):
        d=df[df['Labels']==i]
        r=d['REVENUE'].mean()
        t=d['TOTAL_ORDERS'].mean()
        c=d['AVGDAYSBETWEENORDERS'].mean()
        d=d['DAYSSINCELASTORDER'].mean()
        v1.append([r,t,c,d])
    return v1


# In[452]:


t=va(df)


# In[453]:


t=np.array(t)


# In[27]:


# helper function to print values on the chart
def show_values_on_bars(axs):
    def _show_on_single_plot(ax):        
        for p in ax.patches:
            _x = p.get_x() + p.get_width() / 2
            _y = p.get_y() + p.get_height()
            value = '{:.2f}'.format(p.get_height())
            ax.text(_x, _y, value, ha="center") 

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)


# In[450]:


df


# In[456]:


fig,ax=plt.subplots(2,2,figsize=(20, 15))
for i in range(2):
    for j in range(2):
        sns.barplot(x = h1,
                y = t[:,2*i+j],ax=ax[i][j])
        ax[i][j].set_title(h[2*i+j])
        show_values_on_bars(ax[i][j])


# We can observe that Label 1 generated more revenue, following Label 3 & Label 0. Label 2, Label 4 are least

# In total orders Label 1 topped, following Label 3 & Label 0. Label 2, Label 4 are ordered on average 3-6 orders till date.

# Average days between orders should be as low as possible. Label 1, Label 3 has least days. Label 0, Label 2 are medium. Label 4 has highest days in between orders.

# Lesser the value of days since last order is good. Label 2 has highest value, following Label 4. On average Label 1, Label 3 customers ordered before 58 days. Label 0 has 65 days on an average

# In[460]:


y0=[]
y1=[]
for i in range(5):
    y0.append(df[df['Labels']==i]['AVGDAYSBETWEENORDERS'].mean())
    y1.append(df[df['Labels']==i]['DAYSSINCELASTORDER'].mean())


# In[462]:


plt.figure(figsize=(15, 7))
plt.plot( range(5), y0, marker='o', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4)
plt.plot( range(5), y1, marker='o', markerfacecolor='olive', markersize=12, color='skyblue', linewidth=4)
plt.xlabel('Labels')
plt.ylabel('Count_days')


# We can observe that, Label4 has highest average days between order. Least the value means, more frequently customer buy product. Here, Label1 customers buys more frequently.  

# Days since last order represents how many days before he ordered. On average Label2 customers ordered before 110 days, whereas Label4 ordered before 96 days.

# ## From the above bar, line chart observation we can divide customers into segments

# ## We can segment Label 1 as Champion, Label 0,2,3 as Potential customers and Label 4 as Need attention

# ## We segmented Label 2 as Potential Customers because average days between order is low, which is good

# In[259]:


week_order0=df[df['Labels']==0]
week_order1=df[df['Labels']==1]
week_order2=df[df['Labels']==2]
week_order3=df[df['Labels']==3]
week_order4=df[df['Labels']==4]
y0=[]
y1=[]
y2=[]
y3=[]
y4=[]
for i in week_order0.columns[22:26]:
    y0.append(week_order0[i].mean())
    y1.append(week_order1[i].mean())
    y2.append(week_order2[i].mean())
    y3.append(week_order3[i].mean())
    y4.append(week_order4[i].mean())


# In[267]:


plt.figure(figsize=(15, 7))
plt.plot( week_order0.columns[22:26], y0, marker='o', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4)
plt.plot( week_order1.columns[22:26], y1, marker='o', markerfacecolor='olive', markersize=12, color='skyblue', linewidth=4)
plt.plot( week_order2.columns[22:26], y2, marker='o', markerfacecolor='r', markersize=12, color='skyblue', linewidth=4)
plt.plot( week_order3.columns[22:26], y3, marker='o', markerfacecolor='g', markersize=12, color='skyblue', linewidth=4)
plt.plot( week_order4.columns[22:26], y4, marker='o', markerfacecolor='b', markersize=12, color='skyblue', linewidth=4)
plt.legend()


# In[ ]:





# # Cluster Divide

# ## CHAMPIONS

# ## Label: 1

# In[165]:


df1=df[df['Labels']==1]


# ## Average spending per day ??

# In[447]:


c=df1['REVENUE']/df1['No.of.Days']
from matplotlib import pyplot as plt
import numpy as np


# Creating dataset
a = np.array(c)

# Creating histogram
fig, ax = plt.subplots(figsize =(10, 7))
ax.hist(a, bins = range(0,69,3), edgecolor='black')

plt.title("Average spending")
plt.xlabel('\nspending_per_day\n', fontsize = 10)
plt.ylabel('\ncount\n', fontsize = 10)

# Show plot
plt.show()


# Most Champion Customer's average order value per day lies between 3-15 rupees.

# ## How many orders customer do ??

# In[379]:


from matplotlib import pyplot as plt
import numpy as np


# Creating dataset
a = np.array(df1['TOTAL_ORDERS'])

# Creating histogram
fig, ax = plt.subplots(figsize =(10, 7))
ax.hist(a, bins = range(min(a),max(a),5),edgecolor='black')

plt.title("Average spending")
plt.xlabel('\nspending_per_day\n', fontsize = 10)
plt.ylabel('\ncount\n', fontsize = 10)

# Show plot
plt.show()


# We can observe that, most Champion customer's order 30-90 orders till date.

# ## Average order value 

# In[387]:


from matplotlib import pyplot as plt
import numpy as np


# Creating dataset
a = np.array(df1['AVERAGE_ORDER_VALUE'])

# Creating histogram
fig, ax = plt.subplots(figsize =(10, 7))
ax.hist(a, bins = range(60,1000,20),edgecolor='black')

plt.title("Average spending")
plt.xlabel('\nspending_per_day\n', fontsize = 10)
plt.ylabel('\ncount\n', fontsize = 10)

# Show plot
plt.show()


# From the graph, most of the customers in this cluster spend 0-400 rupees on each order.

# We can recommend products that are in the range 0-400 rupees.
# If order value crosses 400 rupees then we can give additional discount or coupons, so that they can buy more.

# ## Average days between orders

# In[392]:


from matplotlib import pyplot as plt
import numpy as np


# Creating dataset
a = np.array(df1['AVGDAYSBETWEENORDERS'])

# Creating histogram
fig, ax = plt.subplots(figsize =(10, 7))
ax.hist(a, bins = range(0,78,3),edgecolor='black')

plt.title("Average spending")
plt.xlabel('\nspending_per_day\n', fontsize = 10)
plt.ylabel('\ncount\n', fontsize = 10)

# Show plot
plt.show()


# Most Champion customers order every 15-50 days.

# We can send reminder mails or notifications between 7-15 days, so that he can order even before 15 days has passed.

# ## Hour, Day, Week order data

# In[357]:


cols=df1.columns[8:15]
cols1=df1.columns[-10:-6]
cols2=df1.columns[22:26]
y=[]
y1=[]
y2=[]
for i in cols:
    y.append(df1[i].mean())
for i in cols1:
    y1.append(df1[i].mean())
for i in cols2:
    y2.append(df1[i].mean())


# In[358]:


fig, ax = plt.subplots(1, 1, figsize=(15,7))
sns.barplot(cols1,y1)
show_values_on_bars(ax)
fig, ax = plt.subplots(1, 1, figsize=(15,5))
sns.barplot(cols,y)
show_values_on_bars(ax)
fig, ax = plt.subplots(1, 1, figsize=(15,7))
sns.barplot(cols2,y2)
show_values_on_bars(ax)


# Most of the customers in this cluster order 6-18 hrs.
# Most of the customers in this cluster order during week days.
# Most of the customers in this cluster order between 2-3 week.

# ## POTENTIAL CUSTOMERS

# # Label: 0

# In[182]:


df0=df[df['Labels']==0]
df3=df[df['Labels']==3]


# ## Average spending per day ??

# In[396]:


c=df0['REVENUE']/df0['No.of.Days']
from matplotlib import pyplot as plt
import numpy as np


# Creating dataset
a = np.array(c)

# Creating histogram
fig, ax = plt.subplots(figsize =(10, 7))
ax.hist(a, bins = range(0,50,3),edgecolor='black')

plt.title("Average spending")
plt.xlabel('\nspending_per_day\n', fontsize = 10)
plt.ylabel('\ncount\n', fontsize = 10)

# Show plot
plt.show()


# Most of the Customers spending per day on order around 0-9 rupees 

# ## How many orders customer do ??

# In[399]:


from matplotlib import pyplot as plt
import numpy as np


# Creating dataset
a = np.array(df0['TOTAL_ORDERS'])

# Creating histogram
fig, ax = plt.subplots(figsize =(10, 7))
ax.hist(a, bins = range(4,40,4),edgecolor='black')

# Show plot
plt.show()


# In the graph, most of the potential customers ordered 8-34 orders till date.

# ## Average order value 

# In[402]:


from matplotlib import pyplot as plt
import numpy as np


# Creating dataset
a = np.array(df0['AVERAGE_ORDER_VALUE'])

# Creating histogram
fig, ax = plt.subplots(figsize =(10, 7))
ax.hist(a, bins = range(35,860,10),edgecolor='black')

# Show plot
plt.show()


# Most Potential Customer's average order value per day lies between 20-250 rupees. 

# We can recommend products that are in the range 20-250 rupees.
# If order value crosses 250 rupees then we can give additional discount or coupons, so that they can buy more.

# ## Average day between orders

# In[405]:


from matplotlib import pyplot as plt
import numpy as np


# Creating dataset
a = np.array(df0['AVGDAYSBETWEENORDERS'])

# Creating histogram
fig, ax = plt.subplots(figsize =(10, 7))
ax.hist(a, bins = range(0,370,10),edgecolor='black')

# Show plot
plt.show()


# Most of the customers in this cluster order every 0-180 days

# We can send reminder mails or notifications between 0-180 days.

# ## Hour, Day, Week order data

# In[359]:


day_order=df0.columns[8:15]
time_order=df0.columns[-10:-6]
week_order=df0.columns[22:26]
y=[]
y1=[]
y2=[]
for i in day_order:
    y.append(df0[i].mean())
for i in time_order:
    y1.append(df0[i].mean())
for i in week_order:
    y2.append(df0[i].mean())


# In[360]:


fig, ax = plt.subplots(1, 1, figsize=(15,5))
sns.barplot(time_order,y1)
show_values_on_bars(ax)
fig, ax = plt.subplots(1, 1, figsize=(15,5))
sns.barplot(day_order,y)
show_values_on_bars(ax)
fig, ax = plt.subplots(1, 1, figsize=(15,5))
sns.barplot(week_order,y2)
show_values_on_bars(ax)


# Most of the customers in this cluster order 6-18 hrs.
# Most of the customers in this cluster order during week days.
# Most of the customers in this cluster order between 2-4 week.

# #  Label: 3

# ## Average spending per day ??

# In[408]:


c1=df3['REVENUE']/df3['No.of.Days']
from matplotlib import pyplot as plt
import numpy as np


# Creating dataset
a = np.array(c1)

# Creating histogram
fig, ax = plt.subplots(figsize =(10, 7))
ax.hist(a, bins = range(0,100,5),edgecolor='black')

plt.title("Average spending")
plt.xlabel('\nspending_per_day\n', fontsize = 10)
plt.ylabel('\ncount\n', fontsize = 10)

# Show plot
plt.show()


# Most of the Customers spending per day on order around 0-15 rupees 

# ## How many orders customer do ??

# In[411]:


from matplotlib import pyplot as plt
import numpy as np


# Creating dataset
a = np.array(df3['TOTAL_ORDERS'])

# Creating histogram
fig, ax = plt.subplots(figsize =(10, 7))
ax.hist(a, bins = range(5,80,5),edgecolor='black')

# Show plot
plt.show()


# In the graph, most of the potential customers order 20-55 orders till date.

# ## Average order value 

# In[414]:


from matplotlib import pyplot as plt
import numpy as np


# Creating dataset
a = np.array(df3['AVERAGE_ORDER_VALUE'])

# Creating histogram
fig, ax = plt.subplots(figsize =(10, 7))
ax.hist(a, bins = range(10,1050,10),edgecolor='black')

# Show plot
plt.show()


# Most of the customers in this cluster spend 40-250 rupees on each order

# We can recommend products that are in the range 40-250 rupees.
# If order value crosses 250 rupees then we can give additional discount or coupons, so that they can buy more.

# ## Average day between orders

# In[417]:


from matplotlib import pyplot as plt
import numpy as np


# Creating dataset
a = np.array(df3['AVGDAYSBETWEENORDERS'])

# Creating histogram
fig, ax = plt.subplots(figsize =(10, 7))
ax.hist(a, bins = range(0,210,5),edgecolor='black')

# Show plot
plt.show()


# Most of the customers in this cluster order every 10-100 days

# We can promote ads between 7-100 days.

# ## Hour, Day, Week order data

# In[362]:


day_order=df3.columns[8:15]
time_order=df3.columns[-10:-6]
week_order=df3.columns[22:26]
y=[]
y1=[]
y2=[]
for i in day_order:
    y.append(df3[i].mean())
for i in time_order:
    y1.append(df3[i].mean())
for i in week_order:
    y2.append(df3[i].mean())


# In[363]:


fig, ax = plt.subplots(1, 1, figsize=(15,5))
sns.barplot(time_order,y1)
show_values_on_bars(ax)
fig, ax = plt.subplots(1, 1, figsize=(15,5))
sns.barplot(day_order,y)
show_values_on_bars(ax)
fig, ax = plt.subplots(1, 1, figsize=(15,5))
sns.barplot(week_order,y2)
show_values_on_bars(ax)


# Most of the customers in this cluster order 6-18 hrs.
# Most of the customers in this cluster order during week days.
# Most of the customers in this cluster order between 2-3 week.

# ## Label: 2

# In[463]:


df2=df[df['Labels']==2]


# ## Average spending per day ??

# In[418]:


c=df2['REVENUE']/df2['No.of.Days']
from matplotlib import pyplot as plt
import numpy as np


# Creating dataset
a = np.array(c)

# Creating histogram
fig, ax = plt.subplots(figsize =(10, 7))
ax.hist(a, bins = range(0,50,5),edgecolor='black')

plt.title("Average spending")
plt.xlabel('\nspending_per_day\n', fontsize = 10)
plt.ylabel('\ncount\n', fontsize = 10)

# Show plot
plt.show()


# Most of the Customers spending per day on order around 5 rupees 

# ## How many orders customer do ??

# In[422]:


from matplotlib import pyplot as plt
import numpy as np


# Creating dataset
a = np.array(df2['TOTAL_ORDERS'])

# Creating histogram
fig, ax = plt.subplots(figsize =(10, 7))
ax.hist(a, bins = range(0,15),edgecolor='black')

# Show plot
plt.show()


# In the graph, most of the potential customers order 0-12 orders in lifetime

# ## Average order value 

# In[425]:


from matplotlib import pyplot as plt
import numpy as np


# Creating dataset
a = np.array(df2['AVERAGE_ORDER_VALUE'])

# Creating histogram
fig, ax = plt.subplots(figsize =(10, 7))
ax.hist(a, bins = range(20,1600,20),edgecolor='black')

# Show plot
plt.show()


# Most of the customers in this cluster spend 40-240 rupees on each order

# We can recommend products that are in the range 40-240 rupees.
# If order value crosses 240 rupees then we can give additional discount or coupons, so that they can buy more.

# ## Average day between orders

# In[428]:


from matplotlib import pyplot as plt
import numpy as np


# Creating dataset
a = np.array(df2['AVGDAYSBETWEENORDERS'])

# Creating histogram
fig, ax = plt.subplots(figsize =(10, 7))
ax.hist(a, bins = range(0,510,10),edgecolor='black')

# Show plot
plt.show()


# Most of the customers in this cluster order every 0-180 days

# We can send reminder mails or notifications between 0-180 days.

# ## Hour, Day, Week order data

# In[364]:


day_order=df2.columns[8:15]
time_order=df2.columns[-10:-6]
week_order=df2.columns[22:26]
y=[]
y1=[]
y2=[]
for i in day_order:
    y.append(df2[i].mean())
for i in time_order:
    y1.append(df2[i].mean())
for i in week_order:
    y2.append(df2[i].mean())


# In[365]:


fig, ax = plt.subplots(1, 1, figsize=(15,5))
sns.barplot(time_order,y1)
show_values_on_bars(ax)
fig, ax = plt.subplots(1, 1, figsize=(15,5))
sns.barplot(day_order,y)
show_values_on_bars(ax)
fig, ax = plt.subplots(1, 1, figsize=(15,5))
sns.barplot(week_order,y2)
show_values_on_bars(ax)


# Most of the customers in this cluster order 6-18 hrs.
# Most of the customers in this cluster order during Thursday and Sunday days.
# Most of the customers in this cluster order in 1st, 4th week.

# ## NEED ATTENTION

# In[465]:


df4=df[df['Labels']==4]


# ## Label: 4

# ## Average spending per day ??

# In[449]:


c1=df4['REVENUE']/df4['No.of.Days']
from matplotlib import pyplot as plt
import numpy as np


# Creating dataset
a = np.array(c1)

# Creating histogram
fig, ax = plt.subplots(figsize =(10, 7))
ax.hist(a, bins = range(0,2,1),edgecolor='black')

plt.title("Average spending")
plt.xlabel('\nspending_per_day\n', fontsize = 10)
plt.ylabel('\ncount\n', fontsize = 10)

# Show plot
plt.show()


# Most of the Customers spending per day on order around 1 rupee 

# ## How many orders customer do ??

# In[437]:


from matplotlib import pyplot as plt
import numpy as np


# Creating dataset
a = np.array(df4['TOTAL_ORDERS'])

# Creating histogram
fig, ax = plt.subplots(figsize =(10, 7))
ax.hist(a, bins = range(0,18,2),edgecolor='black')

# Show plot
plt.show()


# In the graph, most of the customers order 2-14 orders till date.

# ## Average order value 

# In[440]:


from matplotlib import pyplot as plt
import numpy as np


# Creating dataset
a = np.array(df4['AVERAGE_ORDER_VALUE'])

# Creating histogram
fig, ax = plt.subplots(figsize =(10, 7))
ax.hist(a, bins = range(10,470,10),edgecolor='black')

# Show plot
plt.show()


# Most of the customers in this cluster spend 50-200 rupees on each order

# We can recommend products that are in the range 50-200 rupees.
# If order value crosses 200 rupees then we can give additional discount or coupons, so that they can buy more.

# ## Average day between orders

# In[443]:


from matplotlib import pyplot as plt
import numpy as np


# Creating dataset
a = np.array(df4['AVGDAYSBETWEENORDERS'])

# Creating histogram
fig, ax = plt.subplots(figsize =(10, 7))
ax.hist(a, bins = range(135,1410,10),edgecolor='black')

# Show plot
plt.show()


# Most of the customers in this cluster order every 150-1250 days

# We can send reminder mails or notifications between 150-1250 days.

# ## Hour, Day, Week order data

# In[366]:


day_order=df4.columns[8:15]
time_order=df4.columns[-10:-6]
week_order=df4.columns[22:26]
y=[]
y1=[]
y2=[]
for i in day_order:
    y.append(df4[i].mean())
for i in time_order:
    y1.append(df4[i].mean())
for i in week_order:
    y2.append(df4[i].mean())


# In[367]:


fig, ax = plt.subplots(1, 1, figsize=(15,5))
sns.barplot(time_order,y1)
show_values_on_bars(ax)
fig, ax = plt.subplots(1, 1, figsize=(15,5))
sns.barplot(day_order,y)
show_values_on_bars(ax)
fig, ax = plt.subplots(1, 1, figsize=(15,5))
sns.barplot(week_order,y2)
show_values_on_bars(ax)


# Most of the customers in this cluster order 6-24 hrs.
# Count is evenly distributed among all days.
# Here we can see increasing trend in orders as week passes by.

# # Observations

# Label 4 - Need Attention
# 
# Label 0, 2, 3 - Potential Customers
# 
# Label 1 - Champions

# Label 4 customers need attention. Average days between orders is very high(510 days). There revenue, total orders are also low. Most customers last order was 3 months back. On an average each customer spends 50-200 rupees per order.

# On average Label 0 customers place 20 orders. Most of the customers last order was 2 months back. Average days between order is 61 days.Most customers placed 8-34 orders till date. Spending per day is 0-9 rupees. On average each customer spending 20-250 rupees on each order. 

# Label 2 customers has least average revenue and total orders. But, they have good average days between orders value. Most of the customer's are inactive since 3-4 months. Label 2 budget is low(average spending per day is just 0-5 rupees)

# Label 3 has good ratio between average revenue and total orders. Average days between order is 39 days. Most customers last order was 2 months back. On average customers spending 40-250 rupees on each order. Spending per day is 0-15 rupees. Till date most customers placed 20-55 orders. They will buy new products every 10-180 days

# Label 1 customers are champion's. Their average revenue and total oreders are very high.  Every customer spend 3-15 rupees per day. Most customer's last order was 1 month back. They will place new order every 15-50 days. Average spending per order is around 0-400 rupees. Till date most customers placed 30-90 orders.

# ## What sort of  segmentation is best fit for this data?

# K-Means Clustering is an unsupervised learning algorithm that is used to solve the clustering problems in machine learning or data science.It is the best unsupervised algorithm in ML. It divides the given data points into various clusters. It is a centroid-based algorithm, where each cluster is associated with a centroid. The main aim of this algorithm is to minimize the sum of distances between the data point and their corresponding clusters. 

# It is a popular algorithm for this type of data. It can easily divide data into multiple clusters.

# In[ ]:




