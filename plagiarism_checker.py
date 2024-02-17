import sys
import subprocess
import download
from pypdf import PdfReader
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
from nltk.tokenize import sent_tokenize
from tqdm.auto import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st
import os
from pathlib import Path
import json
import pickle
import string
import re
import nltk
DIRNAME = Path(os.getcwd())

if not os.path.exists(DIRNAME/'nltk_data'):
    os.mkdir(DIRNAME/'nltk_data')
    nltk.download('punkt', DIRNAME/'nltk_data')
nltk.data.path.append((DIRNAME/'nltk_data').__str__())

if not os.path.exists(DIRNAME/'sources'):
    os.mkdir(DIRNAME/'sources')

if not os.path.exists(DIRNAME/'meta.json'):
    data = {
        'processed_sources':[],
        'source_sentences':0,
        'source_lengths':[],
        'source_indices':[]
    }
    with open(DIRNAME/'meta.json','w') as f:
        json.dump(data,f)

# load embedding model
embedding_model = SentenceTransformer((DIRNAME/ 'st_model').__str__())

st.title("DLBC Plagarism Checker")
# load source data
def extract_pdf(filepath):
    # a pdf file that is locally available in system
    try:
        book = PdfReader(filepath)
        total = ''
        for page_num in tqdm(range(book.pages.__len__())):
            extracted = book.pages[page_num].extract_text()
            total += '/n'+extracted
        return total
    except Exception as e:
        print(filepath)
        print('Error:',e)
        return ''

def extract_txt(filepath):
    # a txt file that is locally available
    try:
        with open(filepath, 'r') as f:
            extracted = f.read()
        return extracted
    except Exception as e:
        print(filepath)
        print('Error:',e)
        return ''

def download_file(file_url):
    # download pdf/txt file with proper url
    try:
        filepath = DIRNAME /'sources'/ file_url.split('/')[-1]
        print(filepath)
        download.download(file_url, filepath)
        return filepath
    except Exception as e:
        print(filepath)
        print('Error:',e)

def add_source_embedding():
    source_files = os.listdir(DIRNAME/'sources')
    meta_json = json.load(open(DIRNAME/'meta.json','r'))
    sources = [] 
    for file in source_files:
        if file not in meta_json['processed_sources']:
            meta_json['processed_sources'].append(file)
            print('Processing:',file)
            if file.endswith('.pdf'):
                sources.append(extract_pdf(DIRNAME/'sources'/file))
            elif file.endswith('.txt'):
                sources.append(extract_txt(DIRNAME/'sources'/file))
    if sources:
        # text
        source_lengths = meta_json['source_lengths']
        source_indices = meta_json['source_indices']
        source_sentences = []
        base_length = len(source_lengths)
        # sentence tokenize documents
        for doc_id, doc in enumerate(sources):
            sentences = sent_tokenize(doc)
            source_lengths.append(len(sentences))
            for sent_id, sentence in enumerate(sentences):
                source_indices.append((doc_id+base_length, sent_id))
                source_sentences.append(sentence)
        meta_json['source_sentences'] = len(source_indices)
        meta_json['source_indices'] = source_indices
        
        sources_embedding_list = []
        BATCH_SIZE = 10
        for i in tqdm(range(1+len(source_sentences)//BATCH_SIZE)):
            tmp = source_sentences[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            if tmp:
                with torch.no_grad():
                    emb = embedding_model.encode(tmp)
                    sources_embedding_list.append(emb)
        sources_embedding = np.vstack(sources_embedding_list)
        if os.path.exists(DIRNAME/'source_sentences.pkl'):
            with open(DIRNAME/'source_sentences.pkl','rb') as f:
                tmp_sents = pickle.load(f)
            source_sentences = tmp_sents + source_sentences
        with open(DIRNAME/'source_sentences.pkl','wb') as f:
                pickle.dump(source_sentences, f)
        if os.path.exists(DIRNAME/'source_embedding.npy'):
            tmp_array = np.load(DIRNAME/'source_embedding.npy')
            sources_embedding = np.vstack((tmp_array,sources_embedding))
        np.save(DIRNAME/'source_embedding.npy',sources_embedding)
        print('embeddings updated in file')
        with open(DIRNAME/'meta.json','w') as f:
            json.dump(meta_json,f)
def most_similar(query_embedding, source_embedding, topk=50):
    scores = util.dot_score(query_embedding, source_embedding).max(dim=0).values
    print('dot score calculation completed')
    top_indices = torch.argsort(scores,descending=True)[:topk]
    print('top indices selected')
    return top_indices

def preprocess_sentences(sentences):
    maketrans = str.maketrans('','',string.punctuation)
    preprocessed = []
    for s in sentences:
        # convert to lowercase
        s = s.lower()
        # remove punctuations
        s = s.translate(maketrans)
        # remove unncessary spaces, newline characters
        s = re.sub('[\t\n\s]+',' ',s).strip()

        preprocessed.append(s)
    return preprocessed


def highlight_indices(index, sentence):
    grams = [vocab[ii] for ii in index]
    results = []
    for gram in grams:
        loc = sentence.lower().find(gram.lower())
        if loc>=0:
            results.append((loc, loc+len(gram)))
    results.sort()
    consolidated = []
    start, stop = results[0]
    for res in results[1:]:
        aa,bb = res
        if aa<=stop:
            stop = bb
        else:
            consolidated.append((start,stop))
            start,stop = res
    if (start,stop) not in consolidated:
        consolidated.append((start,stop))
    return consolidated


# load query
            
# query = """In the laboratory we like to take things apart, look under the hood,
# poke and prod, hook up our diagnostic tools and check out what
# is really going on. Today, we’re investigating JavaScript’s type
# system and we’ve found a little diagnostic tool called typeof to
# examine variables. Put your lab coat and safety goggles on, and
# come on in and join us."""

def colored_markdown(text):
    print('color',text)
    return f'<span style="color:yellow; font-size: 16px;">{text}</span>'
def normal_markdown(text):
    print('normal',text)
    return f'<span style="color:white; font-size: 16px;">{text}</span>'
    


select  =st.selectbox("Select any one from dropdown: Sources/Query",["Sources","Query"])
if select == "Query":
    query = st.text_area("Enter Query")
    button = st.button("Submit")
    if button:
        try:
            source_embedding = np.load(DIRNAME/'source_embedding.npy')
            with open(DIRNAME/'source_sentences.pkl','rb') as f:
                source_sentences = pickle.load(f)
            print('source embedding',source_embedding.shape)
            print('source sentences',len(source_sentences))
            meta_json = json.load(open(DIRNAME/'meta.json','r'))

            query_sentences = sent_tokenize(query)
            print(len(query_sentences))
            query_embedding_list = []
            BATCH_SIZE = 20
            for i in tqdm(range(1+len(query_sentences)//BATCH_SIZE)):
                tmp =query_sentences[i*BATCH_SIZE:(i+1):BATCH_SIZE]
                if tmp:
                    with torch.no_grad():
                        emb = embedding_model.encode(tmp)
                        query_embedding_list.append(emb)
            query_embedding = np.vstack(query_embedding_list)
            print('query embedding',query_embedding.shape)
            idx = most_similar(query_embedding, source_embedding,topk=50)
            source_lengths = meta_json['source_lengths']
            source_indices = meta_json['source_indices']
            #print('top indices',idx)
            print('source indices',len(source_indices))
            topk_sentences = np.array(source_sentences)[idx]
            topk_indices = np.array(source_indices)[idx]


            query_processed = preprocess_sentences(query_sentences)
            source_processed = preprocess_sentences(topk_sentences)

            # ngram modeling
            ngram_vectorizer = TfidfVectorizer(ngram_range=(3,3),)

            source_vec = ngram_vectorizer.fit_transform(source_processed)
            vocab = {v:k for k,v in ngram_vectorizer.vocabulary_.items()}


            query_vec = ngram_vectorizer.transform(query_processed)

            colored_query = []
            total_length = 0
            colored_length = 0
            for qv,sent in zip(query_vec,query_processed):
                ind = qv.indices
                total_length+=len(sent)
                if ind.any():
                    print(ind)
                    print([vocab[ii] for ii in ind])
                    cons = highlight_indices(ind,sent)
                    print(cons)
                    current = 0
                    for aa,bb in sorted(cons):
                        colored_length += bb-aa
                        if aa>current:
                            # add normal string
                            colored_query.append(normal_markdown(sent[current:aa]))
                        # add highlighted string
                        colored_query.append(colored_markdown(sent[aa:bb]))
                        current = bb
                    if current < len(sent):
                        # add normal string
                        colored_query.append(normal_markdown(sent[current:]))
                            
                else:
                    # add normal string
                    colored_query.append(normal_markdown(sent))
            percent = f'Plagiarism Percentage {round(colored_length/total_length*100,2)}%'
            st.info(percent)
            st.success('Plagiarized text is yellow in colour and other text is white in color:')
            for out in colored_query:
                st.markdown(f'<p>{out}</p>',unsafe_allow_html=True)
        except Exception as e:
            st.success(e)
else:
    input_type = st.radio("Select source",["URL","PDF"])
    if input_type == "URL":
        url = st.text_input("Enter source URL")
        button = st.button("Submit source")
        if button:
            file_name = download_file(url)
            st.info(f"File downloaded...")
            add_source_embedding()
    else:
        file = st.file_uploader("Upload sourse PDF",type= ["pdf"])
        button = st.button("Submit source")
        if button:
            with open(DIRNAME/"sources"/file.name,'wb') as f:
                f.write(file.getbuffer())            
            st.success(f"file saved...")
            add_source_embedding()




