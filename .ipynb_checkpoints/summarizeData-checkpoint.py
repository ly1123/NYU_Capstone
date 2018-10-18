import numpy as np
import pandas as pd

import re

import spacy
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
nlp = spacy.load('en_core_web_sm')

import string
from collections import Counter
from nltk.corpus import stopwords
stopwords = stopwords.words('english')

punctuations = string.punctuation

df = pd.read_csv('data/qaData.csv')

def cleanText(col):
    
    texts = []
    
    for text in col:
        text = re.sub("\d+", "", text)
        text = re.sub("-", "", text)
        text = text.replace('’', '').replace('\'', "").replace(".", "").replace("-", "").replace("...","")
        doc = nlp(text, disable=['parser', 'ner'])
        tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != '-PRON-']
        tokens = [tok for tok in tokens if tok not in stopwords and tok not in punctuations]
        tokens = ' '.join(tokens)
        
        texts.append(tokens)
    return texts

###############################################################################
##############################SUMMARY DF FOR SINGLLE VARIABLE##################
###############################################################################

def summarizeVar(orig_var_name, new_var_name, words_to_save):

    summary_dict = {}

    for j,i in df.groupby(orig_var_name):
        aa = ' '.join(cleanText(i["Question"])).split()
        most_used = Counter(aa).most_common(words_to_save)
        summary_dict[j] = {u[0]:u[1]/len(aa)*100 for u in most_used}

    summary_df = pd.DataFrame.from_dict(summary_dict, orient='index').fillna(0)
    summary_df_long = pd.melt(summary_df.reset_index(), id_vars='index', var_name='Word', value_name='Value')
    summary_df_long.columns = [new_var_name, "Word", "Value"]
    return summary_df_long, summary_dict

summarizeVar('AnalystName', 'Analyst', 10)[0].to_csv('data/analyst_df.csv', index=False)
summarizeVar('Company', 'Company', 10)[0].to_csv('data/company_df.csv', index=False)
summarizeVar('Participants', 'Participants', 10)[0].to_csv('data/participants_df.csv', index=False)
summarizeVar('AnalystCompany', 'AnalystCompany', 10)[0].to_csv('data/analystCompany_df.csv', index=False)
summarizeVar('RegularTag1', 'RegularTag1', 10)[0].to_csv('data/regularTag1.csv', index=False)
summarizeVar('RegularTag2', 'RegularTag2', 10)[0].to_csv('data/regularTag2.csv', index=False)
summarizeVar('RegularTag3', 'RegularTag3', 10)[0].to_csv('data/regularTag3.csv', index=False)
summarizeVar('EarningTag1', 'EarningTag1', 10)[0].to_csv('data/earningTag1.csv', index=False)
summarizeVar('EarningTag2', 'EarningTag2', 10)[0].to_csv('data/earningTag2.csv', index=False)
summarizeVar('EarningTag3', 'EarningTag3', 10)[0].to_csv('data/earningTag3.csv', index=False)

###############################################################################
##############################SUMMARY DF FOR ALL VARIABLES#####################
###############################################################################

cols = ['Company','Participants','Date',
        'EventName','EventType',
        'AnalystName', 'AnalystCompany', 
        'RegularTag1', 'RegularTag2', 'RegularTag3', 
        'EarningTag1', 'EarningTag2', 'EarningTag3']

groups = df.groupby(cols)

summary = {}

for group in groups:
    q_cleaned = cleanText(group[1]['Question'])
    summary[group[0]] = Counter(' '.join(q_cleaned).split())
    
summary_df = pd.DataFrame.from_dict(summary, orient='index').reset_index().fillna(0)
summary_df.columns = cols + summary_df.columns[13:].tolist()
summary_df_long = pd.melt(summary_df, id_vars=cols, var_name='Word', value_name='Value')
summary_df_long = summary_df_long.loc[summary_df_long['Value']>0]
summary_df_long.to_csv('data/summary_df.csv', index=False)


###############################################################################
##############################Tag Summaries by Other Var#######################
###############################################################################

cols2 = ['Company','Participants','Date',
        'EventName','EventType',
        'AnalystName', 'AnalystCompany']

tags = ['RegularTag1', 'RegularTag2', 'RegularTag3', 
        'EarningTag1', 'EarningTag2', 'EarningTag3']


df[cols2 + tags].to_csv('data/tag_summary.csv', index=False)

###############################################################################
##############################Complete Tag Summaries###########################
###############################################################################

tag_summary = {}

tag_cols = ['EarningTag1', 'EarningTag2', 'EarningTag3']
df[tag_cols] = df[tag_cols].fillna("N/A")

for group in df.groupby(tag_cols):
    q_cleaned = cleanText(group[1]['Question'])
    tag_summary[group[0]] = {w:v for w, v in Counter(' '.join(q_cleaned).split()).most_common(25)}
    
tag_summary_df = pd.DataFrame.from_dict(tag_summary, orient='index').reset_index().fillna(0)
tag_summary_df.columns = tag_cols + tag_summary_df.columns[3:].tolist()
tag_summary_df_long = pd.melt(tag_summary_df, id_vars=tag_cols, var_name='Word', value_name='Value')
tag_summary_df_long = tag_summary_df_long.loc[tag_summary_df_long['Value']>0]
tag_summary_df_long.to_csv('data/tag_summary_df.csv', index=False)