import json
import numpy as np
import pickle

arr = np.load('source_embedding.npy')
print(arr.shape)

with open('source_sentences.pkl','rb') as f:
    sents = pickle.load(f)

meta_json = json.load(open('meta.json','r'))
print('sentences',len(sents))
print('indices',len(meta_json['source_indices']))

