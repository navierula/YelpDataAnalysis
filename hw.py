# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 16:36:42 2016

@author: navienarula
"""

import json
from nltk.stem.porter import *
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn import metrics
import scipy as sp
import pandas as pd
import sklearn.datasets as sk_data
import sklearn.metrics as metrics
from sklearn.cluster import KMeans
import seaborn as sns
from sklearn import preprocessing

yelp_reviews = "yelp_academic_dataset_review.json"
yelp_business = "yelp_academic_dataset_business.json"

pos_words = "positive-words.txt"
neg_words = "negative-words.txt"


def strip_punc(s):
    return ''.join(c for c in s if c not in (",","?","!","."))
    
# Grabs data from file name passed in
# Specifically designed for Yelp reviews
def get_reviews(filename):
    
    reviews = []
    count = 0
    with open(filename, "r") as fp:
        for line in fp:
            #print line
            line = line.rstrip()
            line = json.loads(line)
            reviews.append(line)
            count += 1 
            if count == 150:
                break
    return reviews
  
# Returns the stars and score (via sentimentalize) associated with a 
# specific review  
def get_info(reviews, positive, negative):
    
    count = 0
    stars_score = []
    for review in reviews:
        score = sentimentalize(review['text'], positive, negative)
        stars_score.append((review['stars'],score))
        count +=1 
        if count == 150:
            break
    
    return stars_score

# Implement algorithm that will assign scores to words
# Positive = + 1
# Negative = - 1
# Neutral = + 0      
def sentimentalize(text, positive, negative):
    
    stemmer = PorterStemmer()
    pos_word_count = 0 
    neg_word_count = 0 
    
    
    for word in text.split(" "):
        #print word
        word = word.lower()
        word = strip_punc(word)
        word = stemmer.stem(word)
        
        if word in positive: # no overlap
            pos_word_count += 1

          
        if word in negative:
            neg_word_count += 1
            
    return float(pos_word_count)/(pos_word_count + neg_word_count + 0.01)
  
# Grab words, specificallty from positive and negative files                
def get_emotion_words(filename):
    
    words = set()
    with open(filename, "r") as fp:
        for line in fp:
            line = line.rstrip()
            words.add(line)
    return words
    
# Call functions necessary to perform sentiment analysis
yelp = get_reviews(yelp_reviews) 

positive = get_emotion_words(pos_words)
negative = get_emotion_words(neg_words)

info = get_info(yelp, positive, negative)

#print info[0:30]

# Draw visualization of stars and scores
def draw_viz(info):
    
    stars = [tup[0] for tup in info]
    #scores = [tup[1] for tup in info]
    
    plt.hist(stars, bins=np.arange(6)+0.5) #scores range
    plt.xticks(range(1,6))
    plt.xlim([0.5,5.5])
    plt.show()
     

# Retrieves entries from specified filenames
# Specifically designed for business objects              
def get_business_objects(filename):
    
    bus_obj = []
    with open(filename, "r") as fp:
        for line in fp:
            line = line.rstrip()
            line = json.loads(line)
            bus_obj.append(line)
            #print
    return bus_obj  
    
yelp = get_business_objects(yelp_business)

# Retrieve latitude, longitude, attributes, and categories from Las Vegas restaurants
def get_info(bus_objs):
    
    bus_obj_LV = []
    for bus_obj in bus_objs:
        
        if "Las Vegas" in bus_obj["full_address"] and "Restaurants" in bus_obj["categories"]:
            restaurant = {}
            restaurant["latitude"] = bus_obj['latitude']
            restaurant["longitude"] = bus_obj['longitude']
            
            for attribute in bus_obj['attributes']:
                if isinstance(bus_obj['attributes'][attribute], dict):
                    for sub_attrib in bus_obj['attributes'][attribute]:
                        restaurant[sub_attrib] = bus_obj['attributes'][attribute][sub_attrib]
                else:
                    restaurant[attribute] = bus_obj['attributes'][attribute]
            
            for category in bus_obj['categories']:
                restaurant[category] = True
                
            bus_obj_LV.append(restaurant)
            #print
 
    return bus_obj_LV
        
# Initialize k-means++ algorithm by saving features in a vector        
restaurants = get_info(yelp)

vec = DictVectorizer()

R = vec.fit_transform(restaurants).toarray()

features = vec.get_feature_names()

# Retrieved from lecture notes,
# this function will determine the right k

# returns a visualization denoting the best k
def sc_evaluate_clusters(X,max_clusters):
    s = np.zeros(max_clusters+1)
    s[0] = 0;
    s[1] = 0;
    for k in range(2,max_clusters+1):
        kmeans = KMeans(init='k-means++', n_clusters=k, n_init=10)
        kmeans.fit_predict(X)
        s[k] = metrics.silhouette_score(X,kmeans.labels_,metric='cosine')
    plt.plot(range(2,len(s)),s[2:])
    plt.xlabel('Number of clusters')
    plt.ylabel('Adjusted Rand Index')
    
#sc_evaluate_clusters(R,10)
    
k = 5

# Returns the k-means ++ visualization    
def k_means(features, R, k):

    #print features.index("latitude")
    #print features.index("longitude")
    
    
    R = preprocessing.scale(R)
    
    kmeans = KMeans(init='k-means++', n_clusters=k, n_init=10)
    assignments = kmeans.fit_predict(R)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    error = kmeans.inertia_

    print "The total error of the clustering is: ", error
    print '\nCluster labels'
    print labels
    print '\n Cluster Centroids'
    print centroids
    
    
    print("Top terms per cluster:")
    asc_order_centroids = kmeans.cluster_centers_.argsort()
    order_centroids = asc_order_centroids[:,::-1]
    #terms = features
    for i in range(k):
        print "Cluster %d:" % i
        for ind in order_centroids[i, :10]:
            print ' %s' % features[ind]
            print
    
    #euclidean_dists = metrics.euclidean_distances(R)
    
    colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk']) #blue, green, red
    colors = np.hstack([colors] * 20)
    

    plt.scatter(R[:, 229], R[:, 231], color=colors[assignments].tolist(), s=10, alpha=0.8)

    plt.show()
    
#draw_viz(info)
#sc_evaluate_clusters(R,10)    
#k_means(features, R, k)  
    
 
     
     


    