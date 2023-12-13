import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
#import math
from decimal import Decimal
import csv

#rounding csm float values
#def round_up(n, decimals=0):
#    multiplier = 10 ** decimals
#    return math.ceil(n * multiplier) / multiplier

#use pandas to extract input captions data from 'Caption' column from .csv file
input_file = input("Enter the .csv input filename: ")
source = pd.read_csv(input_file)
input_captions = list(source.Caption)

print(input_captions)

threshold = Decimal(input("Enter the threshold value between 0 and 1: ")) #threshold value to determine key captions

#vectorize data
vec = TfidfVectorizer()
X = vec.fit_transform(input_captions) #vectorized input captions data

#calculate pairwise cosine similarities
S = cosine_similarity(X)

print(S) #print cosine similarity matrix

is_keyframe_list = []
is_keyframe_list.append(False) #assign False value to first sentence as baseline

#for row_index in range(len(S) - 1):
#    for col_index in range(1, len(S)):
#        if Decimal(S[row_index][col_index]) < threshold:
#            is_keyframe_list.append(True)
#        else:
#            is_keyframe_list.append(False)
#        break #only care about pairwise sentence comparisons (1,2) ... (2991,2292)

#only care about pairwise sentence comparisions (1,2) ... (2991,2292)
row_index = 0
col_index = 1

while row_index < len(S) - 1:
    if S[row_index][col_index] < threshold:
        is_keyframe_list.append(True)
    else:
        is_keyframe_list.append(False)
    row_index += 1
    col_index += 1

print(is_keyframe_list)
print("Total # of Captions: " + str(len(is_keyframe_list)))

num_of_keyframes = 0

for i in is_keyframe_list:
    if i is True:
        num_of_keyframes += 1
print("Total # of Key Captions: "+ str(num_of_keyframes))

#writing output to .csv file
fields = ['Is Keyframe', 'Caption', 'Cosine Similarity Matrix']
output_file = input("Enter the .csv output filename: ")

#formatting .csv file into 3 columns
with open(output_file, 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields)
    #csvwriter.writerows(S)
    #while i < num_of_keyframes:
        #csvwriter.writerow(is_keyframe_list)
        #csvwriter.writerow(S)
        #i += 1
    #for ele in is_keyframe_list:
        #csvwriter.writerow([ele])
    #csvwriter.writerows(S)
    rows = zip(is_keyframe_list, input_captions, S)
    for row in rows:
        csvwriter.writerow(row)