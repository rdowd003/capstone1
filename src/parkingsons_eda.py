# Basic EDA & Plotting
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from getdata import get_df
from itertools import cycle
cycol = cycle('bgrcmk')


###Importing data
#______________________________________________________________________________

df = get_df('parkinsons_data.csv')

###Grouping Data by patient (subject# column), assessing Final TS - Initial
df_by_patient = df.groupby('subject#')['motor_UPDRS'].agg(np.ptp)
df_by_patient2 = df.groupby('subject#')['age'].value_counts()

df_by_age = df.groupby('age')['total_UPDRS']


# Binning test_time by level (week), 1-27, each been ~7 days
labels = list(np.arange(1,24))
df['week'] = pd.cut(df['test_time'],bins=27,include_lowest=True)


#Group by subject# and Week, get mean of each bin
df_tt = df.groupby(['subject#','week'])['total_UPDRS','motor_UPDRS','age','Jitter(%)','Shimmer','HNR'].mean()
df_tt=df_tt.dropna()

#Second groupby to grab first row(bin) for each subject to plot age vs first obs.
df_onset=df_tt.groupby('subject#')['total_UPDRS','motor_UPDRS','age','Jitter(%)','Shimmer','HNR'].last()
#print(df_by_patient.head())
df_onset.rename(columns={'total_UPDRS':'Total Score','motor_UPDRS':'Motor Score','Jitter(Abs)':'Jitter'})

# Creating score change column series




#Plots
#______________________________________________________________________________

#Time Progression Plots:
#_______________________

###Histogram of Total Scores and Motor Scores (By Patient)
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(9,5))

ax1.hist(df_by_patient,zorder=2,bins=9)
ax1.set_xlabel('Change in Score (final-initial)')
ax1.set_ylabel('Patient Count')
ax1.set_title('Change in Total Score')
ax1.set_ylim([0,8.5])
ax1.grid(zorder=0)

ax2.hist(df_by_patient,zorder=2,bins=9)
ax2.set_title('Change in Motor UPDRS Score')
ax2.set_xlabel('Change in Score (final-initial)')
ax2.set_ylim([0,8.5])
ax2.grid(zorder=0)

plt.suptitle('Distribution of Change in Score Over Time')
plt.show()



#Plotting UPDRS progression for 5 Random patients to Show Trends
#To plot progression for all patients:
#pts = np.arange(1,43)

#Choose 5 random patients:
pts = np.random.choice(43,size=5,replace=False)
#Plot
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(12,5))
colors = ['black','blue','red','green','orange']
for i,k in enumerate(pts):
    pt = df[df['subject#']==k].sort_values(by='test_time')
    x = pt['test_time'].values
    y_total = pt['total_UPDRS'].values
    y_motor = pt['motor_UPDRS'].values
    c = colors[i]
    ax1.plot(x, y_total, 'go--', linewidth=2, markersize=4,c=c)
    ax2.plot(x,y_motor, 'go--', linewidth=2, markersize=4,c=c)
ax1.set_xlabel('Days')
ax2.set_xlabel('Days')
ax1.set_ylabel('Score')
ax2.set_ylabel('Score')
ax1.set_title('Total Score')
ax2.set_title('Motor Score')
plt.suptitle('Time Progression of UPDRS For n=5 Patients')
plt.legend(labels=['Patient 1','Patient 2','Patient 3','Patient 4','Patient 5'])
plt.show()


#Single Patient time-progression plots with: Jitter, Shimmer
pt_x = df[df['subject#']==1].sort_values(by='test_time')
x1 = pt_x['test_time'].values
y_jitter = pt_x['Jitter(%)'].values
y_shimmer = pt_x['Shimmer'].values
y_dfa = pt_x['DFA'].values
y_ppe = pt_x['PPE'].values

#jitter and Shimmer over time for Patient 1
fig,ax = plt.subplots()
ax.plot(x1,y_dfa,'yo--')
ax.set_xlabel('Days in Trial')
ax2 = ax.twinx()
ax2.spines['right'].set_position(('axes', 1.0))
ax2.plot(x1,y_ppe,'ko--',alpha=0.3)


#plt.plot(x1,y_jitter)
#plt.plot(x1,y_shimmer)
#plt.plot(x1,y_dfa)
ax2.legend(['Jitter','Shimmer'])
plt.show()

#work on legend ^^^^^


#Scatters:
#_______________________

#Scatter matrices of Jitter and Shimmer, Energy, 1 from each categrory
df_jitter = df[['total_UPDRS','motor_UPDRS','Jitter(%)','Jitter(Abs)','Jitter:RAP','Jitter:PPQ5','Jitter:DDP']]
df_shimmer = df[['total_UPDRS','motor_UPDRS','Shimmer','Shimmer(dB)','Shimmer:APQ3','Shimmer:APQ5','Shimmer:APQ11','Shimmer:DDA']]
df_energy = df[['total_UPDRS','motor_UPDRS','RPDE','PPE']]
df_noise = df[['total_UPDRS','motor_UPDRS','HNR','NHR','DFA']]
df_subset = df[['total_UPDRS','motor_UPDRS','Jitter(%)','Shimmer','age','HNR','RPDE']]

ax1 = pd.plotting.scatter_matrix(df_jitter,alpha=0.2, figsize=(6, 6), diagonal='kde')
plt.suptitle('Jitter Components')
[plt.setp(item.yaxis.get_label(), 'size', 7,rotation=0) for item in ax1.ravel()]
[plt.setp(item.xaxis.get_label(), 'size', 6,rotation=45) for item in ax1.ravel()]
plt.show()
ax2 = pd.plotting.scatter_matrix(df_shimmer,alpha=0.2, figsize=(6, 6), diagonal='kde')
plt.suptitle('Shimmer Components')
[plt.setp(item.yaxis.get_label(), 'size', 7,rotation=0) for item in ax2.ravel()]
[plt.setp(item.xaxis.get_label(), 'size', 6,rotation=45) for item in ax2.ravel()]
plt.show()
ax3 = pd.plotting.scatter_matrix(df_energy,alpha=0.2, figsize=(6, 6), diagonal='kde')
plt.suptitle('Energy Components')
[plt.setp(item.yaxis.get_label(), 'size', 7,rotation=0) for item in ax3.ravel()]
[plt.setp(item.xaxis.get_label(), 'size', 6,rotation=45) for item in ax3.ravel()]
plt.show()
ax4 = pd.plotting.scatter_matrix(df_noise,alpha=0.2, figsize=(6, 6), diagonal='kde')
plt.suptitle('Noise Components')
[plt.setp(item.yaxis.get_label(), 'size', 7,rotation=0) for item in ax4.ravel()]
[plt.setp(item.xaxis.get_label(), 'size', 6,rotation=45) for item in ax4.ravel()]
plt.show()
ax5 = pd.plotting.scatter_matrix(df_subset,alpha=0.2, figsize=(6, 6), diagonal='kde')
plt.suptitle('Overall Components')
[plt.setp(item.yaxis.get_label(), 'size', 7,rotation=0) for item in ax5.ravel()]
[plt.setp(item.xaxis.get_label(), 'size', 6,rotation=45) for item in ax5.ravel()]
plt.show()



#Distribution of Scores at Onset
fig2,(ax1,ax2) = plt.subplots(1,2,figsize=(9,5))
ax1.hist(df_onset['total_UPDRS'],zorder=2,bins=10)
ax1.set_xlabel('Score')
ax1.set_ylabel('Patient Count')
ax1.set_title('Total Scores at Start')
#ax1.set_ylim([0,8.5])
ax1.grid(zorder=0)

ax2.hist(df_onset['motor_UPDRS'],zorder=2,bins=10)
ax2.set_title('Motor Scores At Start')
ax2.set_xlabel('Score')
#ax2.set_ylim([0,8.5])
ax2.grid(zorder=0)
plt.suptitle('Distribution of Scores at Beginning of Trials')
plt.show()












#####
