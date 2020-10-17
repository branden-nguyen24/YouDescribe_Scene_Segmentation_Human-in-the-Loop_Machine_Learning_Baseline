import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import re

stop_words = set(stopwords.words('english'))

#a dark dark dark dark dark dark dark night
#a dark dark dark dark dark dark background
#a close up of a ski lift in the snow
#a car driving down a street next to a fence
#a car driving down a street next to a building

caption_Tokens = []
max_Length = 5 #number of image caption inputs
#isKeyframe = False

#scoring for key frame determination
def isKeyFrame(score):
    threshold = 0.6 #flexible threshold constant to determine key frame
    is_KF = False
    if score <= threshold: #if percentage of similar words is less than 60% then it is a key frame
        is_KF = True
    return is_KF

#tokenize and cleanse image captions input
for i in range(max_Length):
    sentence = input("Enter Image Caption " + str(i + 1) + ":\n")

    while sentence == "": #provide multiple chances for user input
        sentence = input("Not A Valid Caption!\nEnter Image Caption " + str(i + 1) + ":\n")

    sentence = re.sub("[^a-zA-Z1-9]", " ", sentence) #all characters that are not letters or numbers are replaced with space
    sentence = sentence.lower() #convert to lower case

    word_List = word_tokenize(sentence) #tokenize converted caption
    word_List = [word for word in word_List if not word in stop_words] #filter out common stop words
    caption_Tokens.append(sorted(set(word_List))) #result is cleansed set of tokens within a 2D matrix

    #caption_List.append( (word_List, isKeyframe) )
    #caption_List.append( (word_tokenize(sentence), isKeyframe) )
    #caption_List.append( (sentence, False) )

print()
print("Caption Tokens: ", end='') #cleansed image caption tokens jagged matrix
print(caption_Tokens)
print()

print("Is Image Caption 1 a Key Frame? : False (Baseline)")

for row_Index, sub_List in enumerate(caption_Tokens):
    score = 0
    for col_Index, token in enumerate(sub_List):
        check_Index = 0
        if row_Index+1 < len(caption_Tokens):
            while check_Index < len(caption_Tokens[row_Index+1]) and row_Index+1 < len(caption_Tokens): #test: len(sub_List)
                if token == caption_Tokens[row_Index+1][check_Index]:#compare current token with tokens of the next row
                    score += 1 #increment score by 1 for each similar word between next 2 caption token lists
                #else:
                    #score -= 1
                check_Index += 1
    if row_Index+1 < len(caption_Tokens):
        print("Is Image Caption " + str(row_Index+2) + " a Key Frame? : " + str(isKeyFrame(score/len(caption_Tokens[row_Index+1]))) + " (" + str(score/len(caption_Tokens[row_Index+1])) + ")") #key frame determination based on score ratio





