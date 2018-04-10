import pandas as pd
import numpy as np
from nltk.corpus import stopwords # for NLP

stop_words = set(stopwords.words("english")) #create a set of stop words in english

#can remove if there is no train dataset
train_frame = pd.read_csv('train.csv') # read the training csv file

#The Jaccard index or Jaccard similarity coefficient, defined as the size of the intersection divided by the size of the union of two label sets, 
#is used to compare set of predicted labels for a sample to the corresponding set of labels in y_true.

#function to compute the score
def compute_score(row):
    "Computes Jaccard similarity score between question1 and question2"
    list_q1 = []
    list_q2 = []
    #removing the stop words from the input questions and appending it to a list
    for word in str(row['question1']).lower().split():
        if word not in stop_words:
            list_q1.append(word)
    for word in str(row['question2']).lower().split():
        if word not in stop_words:
            list_q2.append(word)
    similarity_len = len(set(list_q1) & set(list_q2)) #calculating the similarity by set operation
    total_len = len(set(list_q1) | set(list_q2)) #calculating the size of the entire set
    if len(list_q1) == 0 or len(list_q2) == 0:
        return 0
    score = float(similarity_len)/float(total_len) #gives the score of how much the pair of the questions are similar
    return score

#not required in case of the datset is only single file
#because this doesnt create a model doesnt needs any training dataset to train the model, can comment if want to
score_list = []
for index, row in train_frame.iterrows():
    score = compute_score(row)
    score_list.append(score)

train_frame['jaccard_score'] = score_list
A = np.array([score_list, np.ones(len(score_list))])
w = np.linalg.lstsq(A.T,train_frame['is_duplicate'])[0]
#upto here are the trainig datasection , comment them if not required


#test data goes here
test_frame = pd.read_csv('test.csv')
score_list = []

#computing the score for the entire test data
for index, row in test_frame.iterrows():
    score = compute_score(row)
    score_list.append(score)    

#create a frame for writing the output file
test_frame['jaccard_score'] = score_list
test_frame['is_duplicate'] = test_frame['jaccard_score']*w[0]+w[1]

#saving the computed score set to a csv file
sub = pd.DataFrame()
sub['test_id'] = test_frame['test_id']
sub['is_duplicate'] = test_frame['is_duplicate']
sub.to_csv('least_square_submission.csv', index=False)


