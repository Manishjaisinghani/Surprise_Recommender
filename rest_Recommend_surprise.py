import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from surprise import *
from surprise import accuracy
# from surprise import Reader
# from surprise import NMF
# from surprise import KNNBasic
# from surprise import SVD
# from surprise import evaluate, print_perf
import os

# column_names = ['user', 'item', 'rating', 'timestamp']
# df_csv = pd.read_csv('restaurant_ratings.csv',names = column_names)
#load data from a file
file_path = os.path.expanduser('restaurant_ratings.txt')
reader = Reader(line_format='user item rating timestamp', sep='\t')
data = Dataset.load_from_file(file_path, reader=reader)
data.split(n_folds=3)



def algorithm_to_results(trainset, testset, algo):
    # algo = algo()
    algo.train(trainset)
    predictions = algo.test(testset)
    rmse = accuracy.rmse(predictions, verbose=True)
    fcp = accuracy.fcp(predictions, verbose=True)
    mae = accuracy.mae(predictions, verbose=True)

i = 0
for trainset, testset in data.folds():
    i += 1
    print ("########## Iteration {} ##########".format(i))
    print ("########## SVD Algorithm ##########")
    algorithm_to_results(trainset,testset, SVD())
    print ("########## PMF Algorithm ##########")
    algorithm_to_results(trainset,testset, SVD(biased=False))
    print ("########## NMF Algorithm ##########")
    algorithm_to_results(trainset,testset, NMF())
    print ("########## User based Collaborative Filtering Algorithm ##########")
    algorithm_to_results(trainset,testset, KNNBasic(sim_options = {'user_based': True}))
    print ("########## Item based Collaborative Filtering Algorithm ##########")
    algorithm_to_results(trainset,testset, KNNBasic(sim_options = {'user_based': False}))
