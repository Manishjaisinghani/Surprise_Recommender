import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#RMSE
# df = pd.read_csv('/Users/manish/Documents/UniversityofMissouri/Spring2017/Spatial/Homeworks/HW4/RMSE_Measure.csv')
#
# sns.barplot(x = 'Fold Number',y='Value',hue = 'Algorithm',data=df)
# plt.legend(loc='center left', bbox_to_anchor=(1,.5))
# plt.title("RMSE Measure Comparison Fold 1,2 and 3")
# sns.plt.show()

# df = pd.read_csv('/Users/manish/Documents/UniversityofMissouri/Spring2017/Spatial/Homeworks/HW4/Facet_2.csv')
#
# sns.barplot(x = 'Fold Number',y='Measure',hue = 'Algorithm',data=df)
# plt.legend(loc='center left', bbox_to_anchor=(1,.5))
# plt.title("RMSE Measure Comparison Fold 1,2 and 3")
# sns.plt.show()


# Algorithm = ['SVD', 'PMF', 'NMF', 'UBCF', 'IBCF']
# RMSE = [0.9409, 0.95, 0.975, 0.988, 0.9814]
#
# ax1 = plt.subplot(111)
# plt.title('RMSE Measure Comparison Fold - 1')
# plt.xlabel('Algorithm')
# plt.ylabel('RMSE')
#
# bar1 = ax1.bar(Algorithm, RMSE,width=0.2,color='b',align='center')
# # ax1.legend((bar3[0], bar4[0]), ('KNN Classifier', 'Decision tree'),loc=2)
#
# plt.show()

#MAE
# df = pd.read_csv('/Users/manish/Documents/UniversityofMissouri/Spring2017/Spatial/Homeworks/HW4/MAE_Measure.csv')
#
# sns.barplot(x = 'Fold Number',y='Measure',hue = 'Algorithm',data=df)
# plt.legend(loc='center left', bbox_to_anchor=(1,.5))
# plt.title("MAE Measure Comparison Fold 1,2 and 3")
# sns.plt.show()

# Algorithm = ['SVD', 'PMF', 'NMF', 'UBCF', 'IBCF']
# RMSE = [0.9409, 0.95, 0.975, 0.988, 0.9814]
#
# ax1 = plt.subplot(111)
# plt.title('MAE Measure Comparison Fold - 1')
# plt.xlabel('Algorithm')
# plt.ylabel('RMSE')
#
# bar1 = ax1.bar(Algorithm, RMSE,width=0.2,color='b',align='center')
# # ax1.legend((bar3[0], bar4[0]), ('KNN Classifier', 'Decision tree'),loc=2)
#
# plt.show()

#fcp

# df = pd.read_csv('/Users/manish/Documents/UniversityofMissouri/Spring2017/Spatial/Homeworks/HW4/FCP_Measure.csv')
#
# sns.barplot(x = 'Fold Number',y='Measure',hue = 'Algorithm',data=df)
# plt.legend(loc='center left', bbox_to_anchor=(1,.5))
# plt.title("FCP Measure Comparison")
# sns.plt.show()

#Mean RMSE / FCP / MAE
# df = pd.read_csv('/Users/manish/Documents/UniversityofMissouri/Spring2017/Spatial/Homeworks/HW4/Mean_FacetGrid.csv')
# # g = sns.FacetGrid(df, hue = "Accuracy", size = 3,legend_out=False)
# # g = (g.map(plt.hist, "Algorithm", "Measure", edgecolor="w"))
# sns.barplot(x = 'Algorithm',y='Measure',hue = 'Accuracy',data=df)
# # g.add_legend()
# # plt.legend(loc='best')
# plt.legend(bbox_to_anchor=(1, .5), loc='center left', borderaxespad=0.)
# plt.title("Mean RMSE/MAE comparison")
# plt.tight_layout()
# sns.plt.show()

#FCP Mean
# df = pd.read_csv('/Users/manish/Documents/UniversityofMissouri/Spring2017/Spatial/Homeworks/HW4/Mean_FCP.csv')
#
# sns.barplot(x = 'Algorithm',y='Measure',data=df)
# plt.title("FCP Mean Comparison")
# sns.plt.show()

#MAE Mean
# df = pd.read_csv('/Users/manish/Documents/UniversityofMissouri/Spring2017/Spatial/Homeworks/HW4/MAE_Mean.csv')
#
# sns.barplot(x = 'Algorithm',y='Measure',data=df)
# plt.title("MAE Mean Comparison")
# sns.plt.show()

#RMSE Mean
# df = pd.read_csv('/Users/manish/Documents/UniversityofMissouri/Spring2017/Spatial/Homeworks/HW4/RMSE_Mean.csv')
#
# sns.barplot(x = 'Algorithm',y='Measure',data=df)
# plt.title("RMSE Mean Comparison")
# sns.plt.show()

#User based Collaborative Filtering Similarity Metrics Comparison
# df = pd.read_csv('/Users/manish/Documents/UniversityofMissouri/Spring2017/Spatial/Homeworks/HW4/Similarity_UBCF.csv')
#
# plt.legend(loc='best ',bbox_to_anchor=(1,.5))
# sns.barplot(x = 'Accuracy',y='Measure',hue = 'Algorithm',data=df)
# plt.title("User Based COllabrative Filtering different similarity metrics Comparison")
# sns.plt.show()

# df = pd.read_csv('/Users/manish/Documents/UniversityofMissouri/Spring2017/Spatial/Homeworks/HW4/Similarity_UBCF.csv')
#
# plt.legend(loc='best ',bbox_to_anchor=(1,.5))
# sns.boxplot(x = 'Accuracy',y='Measure',hue = 'Algorithm',data=df)
# plt.title("User Based COllabrative Filtering different similarity metrics Comparison")
# sns.plt.show()


#Item based Collaborative Filtering Similarity Metrics Comparison
# df = pd.read_csv('/Users/manish/Documents/UniversityofMissouri/Spring2017/Spatial/Homeworks/HW4/Similarity_IBCF.csv')
#
# plt.legend(loc='best ',bbox_to_anchor=(1,.5))
# sns.barplot(x = 'Accuracy',y='Measure',hue = 'Algorithm',data=df)
# plt.title("Item Based COllabrative Filtering different similarity metrics Comparison")
# sns.plt.show()


#User based V/s Item Based metrics RMSE
# df = pd.read_csv('/Users/manish/Documents/UniversityofMissouri/Spring2017/Spatial/Homeworks/HW4/UBCF_IBCF_RMSE.csv')
#
#
# sns.barplot(x = 'Similarity Matrix',y='RMSE',hue = 'Algorithm',data=df)
# plt.title("User based V/s Item Based RMSE Accuracy Comparison")
# plt.legend(bbox_to_anchor=(1, .5), loc=2, borderaxespad=0.)
# plt.show()


#User based V/s Item Based metrics FCP
# df = pd.read_csv('/Users/manish/Documents/UniversityofMissouri/Spring2017/Spatial/Homeworks/HW4/UBCF_IBCF_FCP.csv')
#
# sns.barplot(x = 'Similarity Matrix',y='FCP',hue = 'Algorithm',data=df)
# plt.title("User based V/s Item Based FCP Accuracy Comparison")
# plt.legend(bbox_to_anchor=(1, .5), loc=2, borderaxespad=0.)
# plt.show()


#User based V/s Item Based metrics MAE
# df = pd.read_csv('/Users/manish/Documents/UniversityofMissouri/Spring2017/Spatial/Homeworks/HW4/UBCF_IBCF_MAE.csv')
#
# sns.barplot(x = 'Similarity Matrix',y='MAE',hue = 'Algorithm',data=df)
# plt.title("User based V/s Item Based MAE Accuracy Comparison")
# plt.legend(bbox_to_anchor=(1, .5), loc=2, borderaxespad=0.)
# plt.show()

#Pair Plot User based V/s Item Based metrics MAE
# df = pd.read_csv('/Users/manish/Documents/UniversityofMissouri/Spring2017/Spatial/Homeworks/HW4/UBCF_IBCF_PairPlot.csv')
# # print (df.head())
# sns.pairplot(df, hue = 'Algorithm')
# plt.legend(bbox_to_anchor=(1, 1), loc='center left', borderaxespad=0.)
# plt.show()

# df = pd.read_csv('/Users/manish/Documents/UniversityofMissouri/Spring2017/Spatial/Homeworks/HW4/UBCF_IBCF_FacetGrid.csv')
# g = sns.FacetGrid(df, col="Accuracy",  row="Matrix", hue = "Algorithm", size = 3,legend_out=True)
# g = (g.map(sns.pointplot, "Run", "Measure", edgecolor="w"))
# # g.add_legend()
# # plt.legend(loc='best')
# plt.legend(bbox_to_anchor=(1, .5), loc='center left', borderaxespad=0.)
# plt.tight_layout()
# plt.show()


# df = pd.read_csv('/Users/manish/Documents/UniversityofMissouri/Spring2017/Spatial/Homeworks/HW4/UBCF_IBCF_Mean_FacetGrid.csv')
# g = sns.FacetGrid(df, col="Matrix", hue = "Algorithm", size = 3,legend_out=True)
# g = (g.map(sns.barplot, "Accuracy", "Measure", edgecolor="w"))
# # g.add_legend()
# # plt.legend(loc='best')
# plt.legend(bbox_to_anchor=(1, .5), loc='center left', borderaxespad=0.)
# plt.tight_layout()
# sns.plt.show()

df = pd.read_csv('/Users/manish/Documents/UniversityofMissouri/Spring2017/Spatial/Homeworks/HW4/UBCF_IBCF_Mean_FacetGrid.csv')
g = sns.FacetGrid(df, col="Matrix", hue = "Algorithm", size = 3,legend_out=True)
g = (g.map(sns.pointplot, "Accuracy", "Measure", edgecolor="w"))
# g.add_legend()
# plt.legend(loc='best')
plt.legend(bbox_to_anchor=(1, .5), loc='center left', borderaxespad=0.)
plt.tight_layout()
sns.plt.show()


#Comparison of all algorithms on RMSE and MAE
# df = pd.read_csv('/Users/manish/Documents/UniversityofMissouri/Spring2017/Spatial/Homeworks/HW4/Facet_2.csv')
# # g = sns.FacetGrid(df, col="Fold Number",  row="Algorithm", size = 3,legend_out=False)
# sns.factorplot("Algorithm", "Measure", col="Fold Number", row="Accuracy", kind = 'violin',size=3, aspect=.8,data=df,margin_titles=True)
# # g = (g.map(plt.lmplot, "Accuracy", "Measure", edgecolor="w"))
# # g.add_legend()
# # plt.legend(loc='best')
# # plt.legend(bbox_to_anchor=(1, .5), loc='center left', borderaxespad=0.)
# plt.tight_layout()
# sns.plt.show()


# import matplotlib.pyplot as plt
# vals=[1,2,3,4,5]
# inds=range(len(vals))
# labels=["A","B","C","D","E"]
#
# fig,ax = plt.subplots()
# rects = ax.bar(inds, vals)
# ax.set_xticks([ind+0.5 for ind in inds])
# ax.set_xticklabels(labels)




# plt.figure(figsize=(10,6))
# error_rate=[1.3075,1.1351,1.0720,1.0415,1.0228,1.0105,1.0017,0.9951,0.9906,0.9870,0.9843,0.9823,0.9807,0.9793,0.9782,0.9773,0.9768,0.9764,0.9759,0.9756]
# plt.plot(range(1,21),error_rate, color = 'blue',linestyle = 'dashed', marker = 'o', markerfacecolor='red',markersize=5)
# plt.title('UBCF Performance V/S K Value')
# plt.xlabel('k')
# plt.ylabel('error rate')
# plt.show()


# plt.figure(figsize=(10,6))
# error_rate=[1.4252,1.2243,1.1442,1.1015,1.0738,1.0544,1.0395,1.0287,1.0202,1.0133,1.0076,1.0029,0.9992,0.9954,0.9924,0.9901,0.9878,0.9858,0.9843,0.9830]
# plt.plot(range(1,21),error_rate, color = 'blue',linestyle = 'dashed', marker = 'o', markerfacecolor='red',markersize=5)
# plt.title('IBCF Performance V/S K Value')
# plt.xlabel('k')
# plt.ylabel('error rate')
# plt.show()



# error_rate_IBCF=[1.4252,1.2243,1.1442,1.1015,1.0738,1.0544,1.0395,1.0287,1.0202,1.0133,1.0076,1.0029,0.9992,0.9954,0.9924,0.9901,0.9878,0.9858,0.9843,0.9830]
# error_rate_UBCF=[1.3075,1.1351,1.0720,1.0415,1.0228,1.0105,1.0017,0.9951,0.9906,0.9870,0.9843,0.9823,0.9807,0.9793,0.9782,0.9773,0.9768,0.9764,0.9759,0.9756]
# fig, ax1= plt.subplots()
# lns1 = ax1.plot(range(1,21),error_rate_UBCF,linestyle = 'dashed',color = 'blue', marker = 'o',markersize=5,label='UBCF')
# ax1.set_xlabel('K Value')
# ax1.set_ylabel('RMSE_UBCF')
# ax2 = ax1.twinx()
# lns2 = ax2.plot(range(1,21),error_rate_IBCF,linestyle = 'dashed',color='red', marker = 'o',markersize=5,label='IBCF')
# ax2.set_ylabel('RMSE_IBCF')
# # labels = ['UBCF','IBCF']
# # fig.legend([ax1, ax2], ['UBCF','IBCF'])
# # labels = ['UBCF','IBCF']
# # plt.legend(labels)
# lns = lns1+lns2
# labs = [l.get_label() for l in lns]
# ax1.legend(lns, labs, loc=0)
# # ax1.legend()
# # ax2.legend(loc=2)
# plt.title('Best K Value in terms of RMSE UBCF & IBCF - 20 iterations')
# plt.show()
