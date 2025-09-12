# -*- coding: utf-8 -*-
"""
Created on Sat Jul 26 10:20:23 2025

@author: Norat
"""
#  seems like spacy 3.8.7 needs numpy >= 2.0.0 which is not supported by Spyder

import natural_language_processing as nlp
import spacy
import pandas as pd
import numpy as np

pt_large = spacy.load('pt_core_news_lg')

fr_large = spacy.load('fr_core_news_lg')
de_large = spacy.load('de_core_news_lg')

pt_path01 = r"C:/Users/Norat/OneDrive/D_Code/Python/Python NLP/NLP 01/NLP 08_VocabList/O Google CRIOU um ROBÔ CONSCIENTE.txt"
pt_path02 = r"C:/Users/Norat/OneDrive/D_Code/Python/Python NLP/NLP 01/NLP 08_VocabList/02 Por que os Chineses estudam Tanto.txt"

pt_word_count01 = nlp.word_freq_all(data_path = pt_path01, model = pt_large)
pt_word_count02 = nlp.word_freq_all(data_path = pt_path02, model = pt_large)


pt_word_count_concat = nlp.concat_vocab_df(pt_word_count01, pt_word_count02)
