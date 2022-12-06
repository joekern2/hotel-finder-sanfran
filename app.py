"""


@author: Joseph Kern
"""

import streamlit as st
import pandas as pd
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy import displacy
import sentence_transformers
from sentence_transformers import SentenceTransformer, util
from string import punctuation
from collections import Counter
from heapq import nlargest
from PIL import Image
import torch
import pickle as pkl
from tqdm import tqdm
import os



embedder = SentenceTransformer('all-MiniLM-L6-v2')

st.title("Joseph Kern")
st.markdown("This is Hotel Finder for San Francisco.")


def run():
    with open("sanfran_df.pkl" , "rb") as file_1, open("sanfran_corpus.pkl" , "rb") as file_2, open("sanfran_corpus_embeddings.pkl" , "rb") as file_3:
        df = pkl.load(file_1)
        corpus = pkl.load(file_2)
        corpus_embeddings = pkl.load(file_3)

    query = ''
    query = st.text_input("Enter what you want from your hotel:")

    top_k = min(5, len(corpus))

    query_embedding = embedder.encode(query, convert_to_tensor=True)

    cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)

    st.markdown("\n\n======================\n\n")
    st.write("Query: ", query)
    st.write("\nTop 5 most similar hotels:")

    for score, idx in zip(top_results[0], top_results[1]):
        row_dict = df.loc[df['all_review']== corpus[idx]]
        st.write("\n\nHotel: ", row_dict['Hotel'].values[0])





if __name__ == '__main__':
    run()
