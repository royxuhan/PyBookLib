### Multi-layer Regression using neural network
### 09-22-2017

from sklearn import datasets
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import csv
import numpy as np

"""
X = [[0., 0.], [1., 1.]]
y = [[0, 1], [1, 1]]
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes = (15,), random_state =1)

clf.fit(X, y)
"""
"""
tag_ids, tag_names = np.loadtxt(open("tags.csv", "rb"), delimiter=",", skiprows=1, usecols=(0,1),
                  dtype='int,str', unpack=True)

tag_id_counts = np.vstack((tag_ids, np.zeros(34252, 'int')))


book_tags = np.loadtxt(open("book_tags.csv", "rb"), delimiter=",", skiprows=1, usecols=(0,1,2),
                  dtype="int")
for i in range(tag_id_counts[0].size):
    tag_id = tag_id_counts[0][i]
    for line in book_tags:
        if tag_id == line[1]:
            tag_id_counts[1][i] += line[2]
"""

r_data = np.loadtxt(open("ratings.csv", "rb"), delimiter=",",
                                            skiprows=1, usecols=(0,1,2),
                                            dtype="int")
book_id_r = r_data[:, 0]
user_id_r = r_data[:, 1]
rating_r = r_data[:, 2]

t_data = np.loadtxt(open("to_read.csv", "rb"), delimiter=",", skiprows=1, usecols=(0,1),
                                    dtype="int")
user_id_t = t_data[:,0]
book_id_t = t_data[:,1]
bt_data = np.loadtxt(open("book_tags.csv", "rb"), delimiter=",",
                                    skiprows=1, usecols=(0,1,2), dtype="int")
book_id_bt = bt_data[:,0]
tag_id_bt = bt_data[:,1]
tag_id_counts = bt_data[:,2]
significant_tags_id = np.loadtxt(open("final_tag_list.csv", "rb"), delimiter=",", skiprows=1, usecols=(0),
                                dtype="int")

def searchf(array, user_id):
    for i in range(array.size):
        if (user_id == array[i]):
            return i
    return -1

def searchb(array, user_id):
    i = array.size
    while(i > 0):
        i -= 1
        if (user_id == array[i]):
            return i
    return -1

def get_user_book_rating(user_id):
    res = []
    start_index = searchf(user_id_r, user_id)
    end_index = searchb(user_id_r, user_id)+1
    for i in range(start_index, end_index):
        t = (book_id_r[i], rating_r[i])
        res.append(t)
    return res

def get_book_relevance(book_id):
    res = [0] * 100
    start_index = searchf(book_id_bt, book_id)
    end_index = searchb(book_id_bt, book_id)+1
    tags_sum = 0
    for i in range(start_index, end_index):
        tags_sum += tag_id_counts[i]

    for i in range(start_index, end_index):
        if tag_id_bt[i] in significant_tags_id:
            res[searchf(significant_tags_id, tag_id_bt[i])] += tag_id_counts/tags_sum

    return res

def get_preference(user_book_rating):
    res = [0] * 100
    for t in user_book_rating:
        book_id = t[0]
        relevance = get_book_relevance(book_id)
        rate = t[1]
        for i in range(100):
            res[i] += relevance[i] * (rate/3)
    return res

def get_wanting(user_id):
    res = [0] * 100
    start_index = searchf(user_id_t, user_id)
    end_index = searchb(user_id_t, user_id) + 1
    for i in range(start_index, end_index):
        relevance = get_book_relevance(book_id_t[i])
        for i in range(100):
            res[i] += relevance[i]
    return res

#np.savetext('tag_counts.csv', tag_id_counts, delimiter=',')


X = []
for users in range(40000):
    user_data = get_user_book_rating(users)
    user_preference = get_preference(user_data)
    X.append(user_preference)

Y = []
for users in range(40000):
    user_data = get_wanting(users)
    Y.append(user_data)

mlr = MLPRegressor(hidden_layer_sizes=(100,),
        activation = "relu", solver="adam", alpha=0.001, batch_size="auto",
        learning_rate="constant", learning_rate_init=0.001, power_t=0.5,
        max_iter=300, shuffle=True, random_state=None, tol=0.0001,
        verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
        early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
        epsilon=1e-08)

mlr.fit(X, Y)

def predict(user_book_rating):
    user_data = get_preference(user_book_rating)
    result = mlr.predict(user_data)
    return result

print(predict([(122, 5), (1333, 4), (2499, 3)]))
