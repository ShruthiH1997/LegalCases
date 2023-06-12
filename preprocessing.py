import pandas as pd
import numpy as np
import string
import nltk
import re
from tqdm import tqdm
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
from pandarallel import pandarallel
pandarallel.initialize()
# from fast_sentence_transformers import FastSentenceTransformer as SentenceTransformer
import pickle

def preprocess(op):
    txt = op.split("text\':")
    if len(txt)>1:
      txt = txt[1]
    else:
      return ""
    txt = re.sub('\\n',' ',txt)#txt.split('\\n')
    #txt = " ".join(txt)    
    txt = "".join(c for c in txt if c not in string.punctuation)
    words = (txt.split())
    words = [w for w in words if w.lower() not in stop_words and len(w) > 3][:512]
    final_txt = " ".join(words)    
    return final_txt

print('reading file')
df = pd.read_excel('full-unlabelled.xlsx')
print('reading file done')
print("preprocessing")
df['opinion_text'] = df['opinions'].parallel_apply(lambda x: preprocess(x))
print("preprocessing done")
df.to_excel('preprocessed_full.xlsx')