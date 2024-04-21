
import numpy as np
import plotly as py
import pandas as pd
import xgboost as xg
import seaborn as sns
from sklearn import tree
from sklearn import metrics
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import matplotlib.gridspec as gridspec
from plotly.subplots import make_subplots
from matplotlib.patches import ConnectionPatch
from plotly.offline import init_notebook_mode, iplot, plot
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.colors
colors = ["#2f3e46","#354f52","#52796f",'#84a98c','#cad2c5','#edede9']
colors2 = ["#582f0e",'#7f4f24','#936639','#a68a64', "#b6ad90","#c2c5aa"]
cmap = matplotlib.colors.ListedColormap(colors2)
sns.palplot(sns.color_palette(colors))
sns.palplot(sns.color_palette(colors2))
import warnings
warnings.filterwarnings('ignore')
from sklearn import svm
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import confusion_matrix, auc, roc_curve
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report,confusion_matrix,precision_score,jaccard_score,recall_score,f1_score,accuracy_score

# Reading the train dataset and creating a copy
data = pd.read_csv('/content/train - train.csv')
df = data.copy()
df

# Reading the test dataset and creating a copy
test = pd.read_csv('/content/test - test.csv')
dff = test.copy()
dff

# Checking the columns of the DataFrame
df.columns

"""### There are 21 columns & 2000 rows in this dataset and our target is **price range**."""

# Function to check the DataFrame
def check_df(df: object, head: object = 5) -> object:
    print("\nShape")
    print(df.shape)
    print("\nTypes")
    print(df.dtypes)
    print("\nNANs")
    print(df.isnull().sum())
    print("\nInfo")
    print(df.info())
check_df(df)

# Checking for duplicated rows
print('Number of duplicated rows: ', len(df[df.duplicated()]))

# Creating a heatmap to visualize missing values
plt.figure(figsize=(22,4))
sns.heatmap((df.isna().sum()).to_frame(name='').T,cmap='GnBu', annot=True,
             fmt='0.0f').set_title('Count of Missing Values', fontsize=18)
plt.show()

"""### There is no duplicated row or NAN data
- There are 8 categorical variables: n_cores , price_range, blue, dual_sim, four_g, three_g, touch_screen, wifi
- There are 13 numeric variables: battery_power, clock_speed, fc, int_memory, m_dep, mobile_wt, pc, px_height, px_width, ram, talk_time, sc_h, sc_w
"""

# Descriptive statistics of the numerical variables
df.describe()[1:].T.style.background_gradient(cmap='GnBu', axis=1)

"""### We can see statistical information on the table above.

### <b><span style="color:#354f52">Finding unique data.</span></b>
"""

# Finding the number of unique values in each column
df.apply(lambda x: len(x.unique()))

# Creating a DataFrame to show the number of unique values in each column
unique = df.nunique()
unique.to_frame().T

# Creating box plots for categorical variables
df_categorical = df[['price_range', 'n_cores', 'blue', 'dual_sim', 'four_g', 'three_g', 'touch_screen', 'wifi']].astype(str)
fig = go.Figure()
for col in set(df.columns) - set(df_categorical):
    fig.add_trace(go.Box(x=df[col], name=col))
fig.update_layout(
    title_text="Box Plot Styling Outliers",
    title_font=dict(family='newtimeroman', size=25),title_x=0.45,
    font=dict(family='newtimeroman', size=16))
fig.show()

"""### <b><span style="color:#354f52">Finding statistical description of categorical & numerical variables.</span></b>"""

# Getting the number of unique values and unique values for categorical variables
unique_counts = df_categorical.nunique()
unique_values = df_categorical.apply(lambda x: x.unique())
pd.DataFrame({'Number of Unique Values': unique_counts, 'Unique Values': unique_values})

# Getting the statistical description of the numerical variables
df_numerical = df.drop(df_categorical.columns, axis=1)
unique_counts = df_numerical.nunique()
unique_values = df_numerical.apply(lambda x: x.unique())
pd.DataFrame({'Number of Unique Values': unique_counts, 'Unique Values': unique_values})

df_numerical.describe().T.round(3).T.style.background_gradient(cmap='GnBu', axis=1)

"""### We can see statistical information of numerical variables on the table above.

### <b><span style="color:#354f52">Value counts of each column that appear to be categorical</span></b>
"""

# Getting the value counts of columns that appear to be categorical
cols1 = ['price_range', 'n_cores', 'blue', 'dual_sim', 'four_g', 'three_g', 'touch_screen', 'wifi']
for col in cols1:
    print(f"\n{df[col].value_counts()}")
    print('_'*25)

"""<a id="vis"></a>
# <p style="background-color:#52796f;font-family:newtimeroman;font-size:100%;color:black;text-align:center;border-radius:15px 50px; padding:7px;border: 1px solid black;">Visualization</p>
"""

# Visualizing the distribution of 'px_height' and 'sc_w' using histograms
df.columns
sns.set(rc={'axes.labelsize': 15})
fig, ax = plt.subplots(1, 2, figsize=(20,4), dpi=120)
_, bin_edges = np.histogram(df['px_height'], range=(np.floor(df['px_heightHey there! I see that you have some code written in Python. How can I assist you with it?
                                                                

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
sns.set_palette('GnBu')

# Pair plot of numerical variables
dnp = sns.pairplot(df.loc[:, ~df.columns.isin(df_categorical)], diag_kind='kde')
plt.show()

# Pair plot of all variables with hue as 'price_range'
sns.pairplot(data=df, diag_kind='kde', hue='price_range',palette='GnBu')
plt.show()

warnings.filterwarnings("ignore")
plt.figure(figsize=(20,15))

# Kernel Density Estimation (KDE) plot for 'battery_power' and 'talk_time' with hue as 'price_range'
plt.subplot(5,5,1)
sns.kdeplot(data=df,x='battery_power', y='talk_time', hue='price_range',color='r',alpha=.7,weights=None,fill=True,multiple='fill',palette='GnBu')
plt.grid()

# KDE plot for 'sc_w' and 'sc_h' with hue as 'price_range'
plt.subplot(5,5,2)
sns.kdeplot(data=df,x='sc_w', y='sc_h', hue='price_range',color='r',alpha=.7,weights=None,fill=True,multiple='fill',palette='GnBu')
plt.grid()

# KDE plot for 'fc' and 'pc' with hue as 'price_range'
plt.subplot(5,5,3)
sns.kdeplot(data=df,x='fc', y='pc', hue='price_range',color='r',alpha=.7,weights=None,fill=True,multiple='fill',palette='GnBu')
plt.grid()

# KDE plot for 'ram' and 'int_memory' with hue as 'price_range'
plt.subplot(5,5,4)
sns.kdeplot(data=df,x='ram', y='int_memory', hue='price_range',color='r',alpha=.7,weights=None,fill=True,multiple='fill',palette='GnBu')
plt.grid()

# KDE plot for 'm_dep' and 'mobile_wt' with hue as 'price_range'
plt.subplot(5,5,5)
sns.kdeplot(data=df,x='m_dep', y='mobile_wt', hue='price_range',color='r',alpha=.7,weights=None,fill=True,multiple='fill',palette='GnBu')
plt.grid()

fig=plt.figure(figsize=(25,20))

# Distribution plots for each numerical variable with hue as 'price_range'
for i,col in enumerate(df_numerical):
    ax=fig.add_subplot(5,3,i+1)
    ax1=sns.distplot(df[col][df['price_range']==1],hist=False, kde=True,color='cadetblue')
    sns.distplot(df[col][df['price_range']==2],hist=False, kde=True,color='teal')
    sns.distplot(df[col][df['price_range']==3],hist=False, kde=True,color='lightseagreen')
    sns.distplot(df[col][df['price_range']==4],hist=False, kde=True,color='darkseagreen')

# Distribution plots for each column that appears to be categorical
warnings.filterwarnings("ignore")
plt.figure(figsize=(25,20))
plt.subplot(5,4,1)
sns.distplot(df['battery_power'],kde=True,color='darkseagreen')
plt.subplot(5,4,2)
sns.distplot(df['blue'],kde=True,color='darkseagreen')
plt.subplot(5,4,3)
sns.distplot(df['clock_speed'],kde=True,color='darkseagreen')
plt.subplot(5,4,4)
sns.distplot(df['dual_sim'],kde=True,color='darkseagreen')
plt.subplot(5,4,5)
sns.distplot(df['fc'],kde=True,color='olivedrab')
plt.subplot(5,4,6)
sns.distplot(df['four_g'],kde=True,color='olivedrab')
plt.subplot(5,4,7)
sns.distplot(df['int_memory'],kde=True,color='olivedrab')
plt.subplot(5,4,8)
sns.distplot(df['m_dep'],kde=True,color='olivedrab')
plt.subplot(5,4,9)
sns.distplot(df['n_cores'],kde=True,color='seagreen')
plt.subplot(5,4,10)
sns.distplot(df['pc'],kde=True,color='seagreen')
plt.subplot(5,4,11)
sns.distplot(df['px_height'],kde=True,color='seagreen')
plt.subplot(5,4,12)
sns.distplot(df['px_width'],kde=True,color='seagreen')
plt.subplot(5,4,13)
sns.distplot(df['ram'],kde=True,color='darkgreen')
plt.subplot(5,4,14)
sns.distplot(df['sc_h'],kde=True,color='darkgreen')
plt.subplot(5,4,15)
sns.distplot(df['sc_w'],kde=True,color='darkgreen')
plt.subplot(5,4,16)
sns.distplot(df['talk_time'],kde=True,color='darkgreen')
plt.subplot(5,4,17)
sns.distplot(df['three_g'],kde=True,color='darkolivegreen')
plt.subplot(5,4,18)
sns.distplot(df['touch_screen'],kde=True,color='darkolivegreen')
plt.subplot(5,4,19)
sns.distplot(df['wifi'],kde=True,color='darkolivegreen')
plt.subplot(5,4,20)
sns.distplot(df['mobile_wt'],kde=True,color='darkolivegreen')

"""### Normal distribution, also known as the Gaussian distribution, is a probability distribution that is symmetric about the mean, showing that data near the mean are more frequent in occurrence than data far from the mean.
### In graphical form, the normal distribution appears as a "bell curve".
### The normal distribution describes a symmetrical plot of data around its mean value, where the width of the curve is defined by the standard deviation. It is visually depicted as the "bell curve."
<img src="https://d2a032ejo53cab.cloudfront.net/Glossary/zl7vYCwx/std3.png" width="400" height="300">

### Skewness measures the degree of symmetry of a distribution. The normal distribution is symmetric and has a skewness of zero. If the distribution of a data set instead has a skewness less than zero, or negative skewness (left-skewness), then the left tail of the distribution is longer than the right tail; positive skewness (right-skewness) implies that the right tail of the distribution is longer than the left.
<img src="https://www.biologyforlife.com/uploads/2/2/3/9/22392738/c101b0da6ea1a0dab31f80d9963b0368_orig.png" width="800" height="600">

### The normal distribution follows the following formula. Note that only the values of the mean (μ ) and standard deviation (σ) are necessary
<img src="https://www.investopedia.com/thmb/lFaG1vgFO0XgA_Xzfw3yPLjG2Iw=/750x0/filters:no_upscale():max_bytes(150000):strip_icc():format(webp)/Clipboard01-fdb217713438416cadafc48a1e4e5ee4.jpg" width="400" height="300">

####    x = value of the variable or data being examined and f(x) the probability function
####    μ = the mean
####    σ = the standard deviation
"""

# measure of skewness
df.skew(axis=0, skipna=True)  # calculating skewness for each column

"""
- positive skewness (right-skewness) is visible for fc, m_dep, sc_w, clock_speed, px_height.
- normal distribution is visible for many columns such as int_memory, mobile_wt, pc, and talk time for all price ranges.
- majority of phones support three g
"""

# Creating count plots for different columns
plt.figure(figsize=(30, 20), dpi=90)
ax = sns.countplot(x='blue', data=df, palette='GnBu', hue="price_range")
plt.xticks(rotation=90, fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('blue', fontsize=20)
plt.ylabel('Count of blue', fontsize=20)
plt.grid()

# ... (similar count plots for other columns)

# Creating a bar plot to compare ram and battery_power by price range
fig, ax1 = plt.subplots(figsize=(12, 5))
ax1.bar(df['price_range'], df['ram'], color='#84a98c', align='center')
ax1.set_ylabel('ram', color='#000000')
plt.xticks(rotation=90)
plt.xlabel('price_range')
ax2 = ax1.twinx()
ax2.bar(df['price_range'], df['battery_power'], color='#52796f', align='edge')
ax2.set_ylabel('battery_power', color='#000000')
plt.title('Comparison of ram & battery_power by price range')
plt.show()

# Creating a stacked bar plot to show the average of each feature by price range
columns1 = ['battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc', 'four_g',
            'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc', 'px_height',
            'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time', 'three_g',
            'touch_screen', 'wifi']
df_mean = df.groupby('price_range')[columns1].mean()
df_mean.plot(kind='barh', stacked=True, figsize=(15, 10), cmap="GnBu")
plt.xlabel('Average')
plt.title('Average of each Feature VS Price Range')
plt.legend(loc='lower right')
plt.show()

"""
- average of each Feature VS price range can be seen from above.
"""

# Creating a bar plot to show the influence of each column on price range
n = df.groupby('price_range')[['battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc', 'four_g',
                               'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc', 'px_height',
                               'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time', 'three_g',
                               'touch_screen', 'wifi']].mean()
n.plot(kind='bar', figsize=(25, 10), cmap='GnBu')
plt.title("Influence of each column on price range")
plt.xlabel('price_range')
plt.ylabel('Average')
plt.show()

"""- **influence of each Feature VS price range can be seen from above.**"""

dbins=pd.cut(df["ram"],bins=[200,800,1000,1500,2500,3500,4500],labels=["200-800","800-1000","1000-1500","1500-2500","2500-3500","3500-4500"])
plt.figure(figsize=(20,10))
sns.countplot(x=dbins,data=df,hue="price_range",palette='GnBu')

dbins=pd.cut(df["px_width"],bins=[200,800,1000,1500,2500],labels=["200-800","800-1000","1000-1500","1500-2500"])
plt.figure(figsize=(20,10))
sns.countplot(x=dbins,data=df,hue="price_range",palette='GnBu')

dbins=pd.cut(df["px_height"],bins=[0,200,800,1000,1500,2500],labels=["0-200","200-800","800-1000","1000-1500","1500-2500"])
plt.figure(figsize=(20,10))
sns.countplot(x=dbins,data=df,hue="price_range",palette='GnBu')

"""- **From above, we can see distributions precisely.**"""

columns = ['battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc', 'four_g',
       'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc', 'px_height',
       'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time', 'three_g',
       'touch_screen', 'wifi']
palette ="GnBu"
for column in columns:
    plt.figure(figsize=(15,2))
    sns.violinplot(x=df[column], palette=palette)
    plt.title(column)
    plt.show()

columns = ['battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc', 'four_g',
       'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc', 'px_height',
       'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time', 'three_g',
       'touch_screen', 'wifi']
palette ="GnBu"
for column in columns:
    plt.figure(figsize=(15,2))
    sns.boxplot(x=df[column], palette=palette)
    plt.title(column)
    stats = df[column].describe()
    stats_text = ", ".join([f"{key}: {value:.2f}" for key, value in stats.items()])
    print(f"\n{column} Statistics:\n{stats_text}")
    plt.show()

plt.subplots(figsize=(20,15))
sns.boxplot(x='price_range', y='battery_power', data=df, palette='GnBu')
plt.subplots(figsize=(20,15))
sns.boxplot(x='price_range', y='blue', data=df, palette='GnBu')
plt.subplots(figsize=(20,15))
sns.boxplot(x='price_range', y='ram', data=df, palette='GnBu')
plt.subplots(figsize=(20,15))
sns.boxplot(x='price_range', y='wifi', data=df, palette='GnBu')
plt.subplots(figsize=(20,15))
sns.boxplot(x='price_range', y='clock_speed', data=df, palette='GnBu')
plt.subplots(figsize=(20,15))
sns.boxplot(x='price_range', y='pc', data=df, palette='GnBu')
plt.subplots(figsize=(20,15))
sns.boxplot(x='price_range', y='fc', data=df, palette='GnBu')
plt.subplots(figsize=(20,15))
sns.boxplot(x='price_range', y='dual_sim', data=df, palette='GnBu')
plt.subplots(figsize=(20,15))
sns.boxplot(x='price_range', y='four_g', data=df, palette='GnBu')
plt.subplots(figsize=(20,15))
sns.boxplot(x='price_range', y='three_g', data=df, palette='GnBu')
plt.subplots(figsize=(20,15))
sns.boxplot(x='price_range', y='touch_screen', data=df, palette='GnBu')
plt.subplots(figsize=(20,15))
sns.boxplot(x='price_range', y='talk_time', data=df, palette='GnBu')
plt.subplots(figsize=(20,15))
sns.boxplot(x='price_range', y='px_width', data=df, palette='GnBu')
plt.subplots(figsize=(20,15))
sns.boxplot(x='price_range', y='px_height', data=df, palette='GnBu')
plt.subplots(figsize=(20,15))
sns.boxplot(x='price_range', y='n_cores', data=df, palette='GnBu')
plt.subplots(figsize=(20,15))
sns.boxplot(x='price_range', y='mobile_wt', data=df, palette='GnBu')
plt.subplots(figsize=(20,15))
sns.boxplot(x='price_range', y='m_dep', data=df, palette='GnBu')
plt.subplots(figsize=(20,15))
sns.boxplot(x='price_range', y='sc_w', data=df, palette='GnBu')
plt.subplots(figsize=(20,15))
sns.boxplot(x='price_range', y='sc_h', data=df, palette='GnBu')
plt.subplots(figsize=(20,15))
sns.boxplot(x='price_range', y='int_memory', data=df, palette='GnBu')

fig, axes = plt.subplots(2,1, figsize=(20,10))
sns.stripplot(data=df, x='ram', palette='GnBu', hue='price_range', y='blue', orient='h', ax=axes[0])
axes[0].set_title('bluetooth - ram', fontsize='16')
axes[0].legend(loc=4)
sns.boxplot(data=df, x='ram', palette='GnBu', hue='price_range', y='blue', orient='h', ax=axes[1])
plt.show()

fig, axes = plt.subplots(2,1, figsize=(20,10))
sns.stripplot(data=df, x='ram', palette='GnBu', hue='price_range', y='dual_sim', orient='h', ax=axes[0])
axes[0].set_title('dual sim - ram', fontsize='16')
axes[0].legend(loc=4)
sns.boxplot(data=df, x='ram', palette='GnBu', hue='price_range', y='dual_sim', orient='h', ax=axes[1])
plt.show()

fig, axes = plt.subplots(2,1, figsize=(20,10))
sns.stripplot(data=df, x='ram', palette='GnBu', hue='price_range', y='three_g', orient='h', ax=axes[0])
axes[0].set_title('three g - ram', fontsize='16')
axes[0].legend(loc=4)
sns.boxplot(data=df, x='ram', palette='GnBu', hue='price_range', y='three_g', orient='h', ax=axes[1])
plt.show()

fig, axes = plt.subplots(2,1, figsize=(20,10))
sns.stripplot(data=df, x='ram', palette='GnBu', hue='price_range', y='four_g', orient='h', ax=axes[0])
axes[0].set_title('four g - ram', fontsize='16')
axes[0].legend(loc=4)
sns.boxplot(data=df, x='ram', palette='GnBu', hue='price_range', y='four_g', orient='h', ax=axes[1])
plt.show()

fig, axes = plt.subplots(2,1, figsize=(20,10))
sns.stripplot(data=df, x='ram', palette='GnBu', hue='price_range', y='n_cores', orient='h', ax=axes[0])
axes[0].set_title('n cores - ram', fontsize='16')
axes[0].legend(loc=4)
sns.boxplot(data=df, x='ram', palette='GnBu', hue='price_range', y='n_cores', orient='h', ax=axes[1])
plt.show()

fig, axes = plt.subplots(2,1, figsize=(20,10))
sns.stripplot(data=df, x='ram', palette='GnBu', hue='price_range', y='touch_screen', orient='h', ax=axes[0])
axes[0].set_title('touch screen - ram', fontsize='16')
axes[0].legend(loc=4)
sns.boxplot(data=df, x='ram', palette='GnBu', hue='price_range', y='touch_screen', orient='h', ax=axes[1])
plt.show()

fig, axes = plt.subplots(2,1, figsize=(20,10))
sns.stripplot(data=df, x='ram', palette='GnBu', hue='price_range', y='wifi', orient='h', ax=axes[0])
axes[0].set_title('wifi - ram', fontsize='16')
axes[0].legend(loc=4)
sns.boxplot(data=df, x='ram', palette='GnBu', hue='price_range', y='wifi', orient='h', ax=axes[1])
plt.show()

fig = plt.figure(figsize=(10, 8.5),dpi=90)
ax = fig.add_subplot(111,projection='3d')
p1 = ax.scatter(df['fc'], df['pc'], df['px_height'],c=df['price_range'],cmap='GnBu')
fig.colorbar(p1, shrink=0.5,label='price_range',anchor=(2,0.5))
ax.set_xlabel("fc")
ax.set_ylabel("pc")
ax.set_zlabel("px_height")
ax.set_title("Correlation Between fc & battery power & px_height",fontdict={'fontsize': 12})
fig.show()

fig = plt.figure(figsize=(10, 8.5),dpi=90)
ax = fig.add_subplot(111,projection='3d')
p1 = ax.scatter(df['ram'], df['battery_power'], df['wifi'],c=df['price_range'],cmap='GnBu')
fig.colorbar(p1, shrink=0.5,label='price_range',anchor=(2,0.5))
ax.set_xlabel("ram")
ax.set_ylabel("battery_power")
ax.set_zlabel("wifi")
ax.set_title("Correlation Between ram & battery power & wifi",fontdict={'fontsize': 12})
fig.show()

"""- **the dispersion of Each Feature at each specified feature can be seen in above plots.**

- **by increasing ram in mobile phones, we can see a increase in price range across all categorical features.**
"""

plt = px.scatter(df, x="ram", y="price_range",size="wifi", color ="three_g",size_max=15,color_continuous_scale="BrBG",hover_data=df[['fc']],template = 'plotly_white')
plt.show()

plt = px.scatter(df, x="ram", y="battery_power",size="talk_time", color ="price_range",size_max=15,color_continuous_scale="BrBG",hover_data=df[['mobile_wt']],template = 'plotly_white')
plt.show()

"""- **mobile phones higher ram & battery power, are more pricy.**"""

gb = sns.FacetGrid(df, col="price_range", hue="price_range")
gb.map(sns.scatterplot, "int_memory", "ram")
gb.set_axis_labels("Internal Memory", "ram")
plt.show()

import matplotlib.pyplot as plt
feature = (['battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc', 'four_g',
       'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc', 'px_height',
       'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time', 'three_g',
       'touch_screen', 'wifi'])
plt.figure(figsize = (20,20))
plot_count = 1
for feature in list (df.columns)[:-1]:
    plt.subplot(5,4, plot_count)
    plt.scatter(df[feature], df['price_range'],c="#52796f")
    plt.xlabel(feature.title())
    plt.ylabel('Personal Loan')
    plot_count+=1
plt.show()

"""- **ram and price range have a positive correlation, as ram increases, price also increases.**"""

fig=plt.figure(figsize=(20,10))
for i,col in enumerate(df_numerical):
    ax=fig.add_subplot(5,3,i+1)
    sns.scatterplot(x='ram',y=col,hue='price_range',data=df,palette="GnBu")

fig=plt.figure(figsize=(20,10))
for i,col in enumerate(df_numerical):
    ax=fig.add_subplot(5,3,i+1)
    sns.scatterplot(x='battery_power',y=col,hue='price_range',data=df,palette="GnBu")

cor=df.corr()
cor

df_c=['battery_power','ram','clock_speed','int_memory','m_dep','mobile_wt',"n_cores","px_width","px_height","talk_time","sc_h","sc_w"]
for i in df_c:
    sns.set_theme(style="white")
    g = sns.JointGrid(data=df, x=i,y='price_range',space=0)
    g.plot_joint(sns.kdeplot,fill=True,thresh=0, levels=100, cmap="GnBu")
    g.plot_marginals(sns.histplot, color="#52796f", alpha=1, bins=20)

results = pd.pivot_table(data=df, index='wifi', columns='three_g', values='price_range')
sns.heatmap(results, cmap='GnBu', annot=True)
plt.show()

results = pd.pivot_table(data=df, index='wifi', columns='four_g', values='price_range')
sns.heatmap(results, cmap='GnBu', annot=True)
plt.show()

results = pd.pivot_table(data=df, index='dual_sim', columns='blue', values='price_range')
sns.heatmap(results, cmap='GnBu', annot=True)
plt.show()

"""- A pivot table is a table of values which are aggregations of groups of individual values from a more extensive table within one or more discrete categories."""

from IPython.core.display import HTML
def multi_table(table_list):
    return HTML(
        '<table><tr style="background-color:white;">' +
        ''.join(['<td>' + table._repr_html_() + '</td>' for table in table_list]) +
        '</tr></table>')

print("Percentages of each unique value of categorical features:")
nunique = {var: pd.DataFrame((df[var].value_counts()/len(df[var])*100).map('{:.3f}%'.format))
              for var in {'blue', 'dual_sim', 'four_g', 'n_cores', 'three_g', 'touch_screen', 'wifi'}}
multi_table([nunique['blue'],nunique['dual_sim'],nunique['four_g'],nunique['n_cores'],nunique['three_g'],nunique['touch_screen'],nunique['wifi']])

"""Observations:
- Bluetooth:   
  - The blue chart indicates that mobile phones without Bluetooth have a higher frequency compared to those with Bluetooth. 📶
  - Among mobile phones without Bluetooth, Low-cost and High-cost phones show the highest frequency, while Very high-cost phones dominate the group of phones with Bluetooth. 📱💙

- Dual Sim:  
  - The dual_sim chart reveals that mobile phones equipped with Dual Sim have the highest frequency. 📱📱
  - In terms of price range, High-cost and Low-cost phones lead among devices without Dual Sim, whereas Very high-cost phones are most prevalent among those with Dual Sim. 📈💵

- 4G:
  - Mobile phones with 4G exhibit a higher frequency compared to those without 4G on the four_g chart. 📶
  - Among phones with 4G, Medium-cost phones are most common in terms of price range. 💲

- Number of Cores:
  - Mobile phones with 4 cores show the highest frequency on the n-cores chart. 🔢
  - High-cost phones are predominant among devices with 4 cores, while Medium-cost phones lead among those containing 4 cores. 💸🔋

- 3G:  
  - The three_g chart displays that mobile phones with 3G have a higher frequency than those without 3G. 📶
  - High-cost phones are most common among devices with 3G, while Low-cost phones dominate the group without 3G. 📱🌐

- Touch Screen:    
  - Phones with Touch Screen have a higher frequency than those without, as shown in the touch_screen chart. 📱🖥️
  - Among devices with Touch Screen, Low-cost phones are most prevalent, while High-cost phones are more common in the group without Touch Screen. 💰💻

- Wifi:     
  - Devices with Wifi have a higher frequency than those without on the wifi chart. 📶
  - Very high-cost phones are most prevalent among devices with Wifi, whereas Low-cost phones lead among those without Wifi. 💰📶

<a id="ml"></a>
# <p style="background-color:#52796f;font-family:newtimeroman;font-size:100%;color:black;text-align:center;border-radius:15px 50px; padding:7px;border: 1px solid black;">Classification Model ((SVM)</p>
"""

df

df.info()

"""# <p style="background-color:#354f52; font-family:calibri; font-size:120%; color:white; text-align:center; border-radius:15px 50px; padding:10px">SVM</p>

SVM 🤖:
Support Vector Machine (SVM) is a widely-used machine learning algorithm utilized for classification and regression tasks. Its operation involves identifying the optimal hyperplane that separates data points into distinct classes. This hyperplane is positioned by maximizing the margin, which denotes the distance between the hyperplane and the nearest data points of each class.

Advantages of SVM classifier 🌟:
1. Robust to noise & outliers 🛡️ - SVM exhibits lower sensitivity to noise and outliers compared to other algorithms, rendering it suitable for tasks where the presence of noise and outliers is anticipated.
2. Good generalization 🧠 - SVM is recognized for its proficient generalization capabilities, allowing it to perform effectively on new and unseen data.
3. Handles non-linear data 🔄 - SVM can manage non-linear decision boundaries, proving valuable in scenarios where the data is not linearly separable.
4. Flexibility in kernel selection 🌐 - SVM offers versatility in choosing diverse kernel functions like linear, polynomial, and radial basis function (RBF), enabling it to function across a broad array of datasets.
5. Effective in high-dimensional spaces 🚀 - SVM can operate effectively with high-dimensional datasets, a feature beneficial in instances where the number of features surpasses the number of samples.

Disadvantages of SVM classifier 📉:
1. Computationally intensive 💻 - SVM demands significant computational resources, potentially resulting in slower performance and heightened memory requirements when handling voluminous datasets.
2. Sensitivity to hyperparameters ⚖️ - SVM's performance is notably influenced by the selection of parameters such as the kernel function and regularization parameters, impacting its overall efficacy.
3. Limited interpretability ❓ - SVM lacks explicit models for interpreting the relationship between features and the target variable, posing challenges in result interpretation.
4. Bias-Variance Tradeoff ⚖️ - Similar to all machine learning algorithms, SVM is subject to the bias-variance tradeoff, hence it can either overfit or underfit the data, leading to subpar performance.
"""

X = df.drop('price_range', axis=1)
y = df['price_range'].ravel()

print ('X:', X.shape,'\ny:', y.shape)

from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# Assuming you have X and y defined

# Remove rows with NaN values from X and y
mask = pd.notnull(X).all(axis=1)
X = X[mask]
y = y[mask]

# Create an imputer with the desired strategy
imputer = SimpleImputer(strategy='mean')

# Create a pipeline with the imputer and the SVC classifier
pipeline = make_pipeline(imputer, SVC())

# finding the best test size value from 0.2 to 0.3
test_size = np.arange(start=0.2, stop=0.35, step=0.05)
score = []

for size in test_size:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size, random_state=0)
    pipeline.fit(X_train, y_train)
    score.append(pipeline.score(X_test, y_test))

r = pd.DataFrame({'Test size': test_size, 'Score': score})
r.sort_values(by=['Score'], ascending=False, inplace=True)
r.style.highlight_max(color='#84a98c')

# Display the result
r

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
# training will be based on 80% of data and we split data to test_size=0.2 & train_size=0.8

print('number of rows in train data is', X_train.shape)
print('number of rows in test data is', X_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)

scaler = StandardScaler() # scale features
X_train = scaler.fit_transform(X_train) #fit the scaler ro training data
X_test = scaler.transform(X_test)

svm_model = svm.SVC(probability=True)
svm_model.fit(X_train, y_train) #fit model to training data

y_pred = svm_model.predict(X_test)

print('\33[46mModel Accuracy Score: {0:0.5f}'. format(accuracy_score(y_test, y_pred)))
print('training set score: {:.5f}'.format(svm_model.score(X_train, y_train)))
print('test set score: {:.5f}'.format(svm_model.score(X_test, y_test)))

import statistics
from statistics import stdev
kf = KFold(n_splits=10, shuffle=False) #cross-validation
score = cross_val_score(svm_model, X_train, y_train, cv=kf, scoring='accuracy')
svm_model_cv_score = score.mean()
svm_model_cv_stdev = stdev(score)
print('cross validation accuracy scores are:\n {}'.format(score))

accuracy = ['CV Accuracy']
svm_A = pd.DataFrame({'CV MEAM':svm_model_cv_score,'STD':svm_model_cv_stdev},index=accuracy)
svm_A

"""- **10-fold cross-validation accuracy does not result in performance improvement for model.**"""

plt.figure(figsize=(12, 8))
sns.set(font_scale=1.2)
ConfusionMatrixDisplay.from_estimator(svm_model, X_test, y_test, colorbar=False, cmap='GnBu')
plt.title('Confusion Matrix of Base SVM')
plt.grid(False)

print(classification_report(y_test, y_pred))

def metrics_calculator(y_test, y_pred, model_name):
    result = pd.DataFrame(data=[accuracy_score(y_test, y_pred),
                                precision_score(y_test, y_pred, average='macro'),
                                recall_score(y_test, y_pred, average='macro'),
                                f1_score(y_test, y_pred, average='macro')],
                          index=['Accuracy','Precision','Recall','F1-score'],
                          columns = [model_name])
    return result

BaseSVM = metrics_calculator(y_test, y_pred, 'Base CVM')
BaseSVM

!pip install scikit-plot

import scikitplot as skplt
#ROC Curve
y_pred_proba = svm_model.predict_proba (X_test)
skplt.metrics.plot_roc_curve (y_test, y_pred_proba, figsize = (12,7))
plt.show ()

from sklearn.model_selection import GridSearchCV
# finding optimal hyperparameters(GridSearchCV)
model = svm.SVC(probability=True) # define model
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=0) # define evaluation
# searching parameters
C = [0.1, 1, 10, 100]
kernel = ['linear', 'poly', 'rbf', 'sigmoid']
gamma = ['scale', 'auto']
param_grid = {'C': C, 'kernel': kernel, 'gamma': gamma}
search = GridSearchCV(model, param_grid, scoring='accuracy', n_jobs=-1, cv=cv)
GridSearchCV = search.fit(X_train, y_train) # execute search
svm_modelcv = GridSearchCV.best_estimator_ # set clf to the best combination of parameters
print('Best Score is: %s' % GridSearchCV.best_score_)
print('Best Hyperparameters are: %s' % GridSearchCV.best_params_)

# tuned model
svm_modelcv.fit(X_train, y_train)

y_pred = svm_modelcv.predict(X_test)

print('\33[46mModel Accuracy Score: {0:0.5f}'. format(accuracy_score(y_test, y_pred)))

plt.figure(figsize=(12, 8))
sns.set(font_scale=1.2)
ConfusionMatrixDisplay.from_estimator(svm_modelcv, X_test, y_test, colorbar=False, cmap='GnBu')
plt.title('Confusion Matrix of Tuned SVM')
plt.grid(False)

print(classification_report(y_test, y_pred))

SVM = metrics_calculator(y_test, y_pred, 'Tuned SVM')
SVM

#ROC Curve
y_pred_proba = svm_modelcv.predict_proba (X_test)
skplt.metrics.plot_roc_curve (y_test, y_pred_proba, figsize = (12,7))
plt.show ()