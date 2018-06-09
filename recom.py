import numpy as np
import pandas as pd
import sys

'''
get argument from commandline and make Dataframe from it 

'''
header = ['user_id', 'item_id', 'rating', 'timestamp']
train = pd.read_csv(sys.argv[1],sep='\t',names=header)
test = pd.read_csv(sys.argv[2],sep='\t',names=header)
'''
Get unique number of users and items

'''

n_users = train.user_id.unique().max()
n_items = train.item_id.unique().max()


''' 
Make matrix of user and item number

'''

train_matrix = np.zeros((n_users, n_items))
for attr in train.itertuples():
	train_matrix[attr[1]-1, attr[2]-1] = attr[3]

test_matrix = np.zeros((n_users, n_items))
for attr in train.itertuples():
	train_matrix[attr[1]-1, attr[2]-1] = attr[3]
	

'''
Get Similarity matrix

'''

from sklearn.metrics.pairwise import pairwise_distances

user_similarity = pairwise_distances(train_matrix, metric = 'cosine')


'''
rui = rubar + k sigma simil(u,u')(ru,i - rubar')
'''
def predict (ratings, similarity, type = 'user'):
	mean_user_rating = ratings.mean(axis=1)
	ratings_diff = (ratings-mean_user_rating[:,np.newaxis])
	pred = mean_user_rating[:,np.newaxis] + similarity.dot(ratings_diff)/np.array([np.abs(similarity).sum(axis=1)]).T
	return pred

user_prediction = predict(train_matrix, user_similarity, type='user')


'''
Write out Result
'''
f = open("u"+sys.argv[1][1:2]+".base_prediction.txt","w")

u = 1
for i in user_prediction:
	t=1
	for j in i:
		j*=10
		if (j>10):
			j%=4
		if(j<0):
			j+=10
			j%=4
		data = "%d\t%d\t%0.0f\n" %(u, t, j)
		f.write(data)
		t+=1
	u+=1

f.close()
