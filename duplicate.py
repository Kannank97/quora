import pandas as pd
import numpy as np
from nltk.corpus import stopwords # for NLP

stops = set(stopwords.words("english")) #create a set of stop words in english
file_frame = pd.read_csv('train.csv') # read the training csv file

#The Jaccard index or Jaccard similarity coefficient, defined as the size of the intersection divided by the size of the union of two label sets, 
#is used to compare set of predicted labels for a sample to the corresponding set of labels in y_true.
def compute_score(row):
    "Computes Jaccard similarity score between question1 and question2"
    q1_list = []
    q2_list = []
    #removing the stop words from the input questions and appending it to a list
    for word in str(row['question1']).lower().split():
        if word not in stops:
            q1_list.append(word)
    for word in str(row['question2']).lower().split():
        if word not in stops:
            q2_list.append(word)
    num = len(set(q1_list) & set(q2_list))
    deno = len(set(q1_list) | set(q2_list))
    if len(q1_list) == 0 or len(q2_list) == 0:
        return 0
    score = float(num)/float(deno)
    return score


score_list = []
for index, row in file_frame.iterrows():
    score = compute_score(row)
    score_list.append(score)

file_frame['jaccard_score'] = score_list
A = np.array([score_list, np.ones(len(score_list))])
w = np.linalg.lstsq(A.T,file_frame['is_duplicate'])[0]

test_frame = pd.read_csv('test.csv')
score_list = []
for index, row in test_frame.iterrows():
    score = compute_score(row)
    score_list.append(score)    

test_frame['jaccard_score'] = score_list
test_frame['is_duplicate'] = test_frame['jaccard_score']*w[0]+w[1]

#saving the computed score set to a csv file
sub = pd.DataFrame()
sub['test_id'] = test_frame['test_id']
sub['is_duplicate'] = test_frame['is_duplicate']
sub.to_csv('least_square_submission.csv', index=False)











