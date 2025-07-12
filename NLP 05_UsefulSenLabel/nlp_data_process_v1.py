#%%
# use env: latest_python for spacy
# Seems like LogisticRegression is better than MultinomialNB

# hours spend on this project
# Jan 20, 24: 7.5 hrs
    # about 40 mins on repackage ml_upsampling,ml_upsampling
# Jan 21, 24: 4 hrs
    # doing scoring and model_analyze
    
# Jan 28, 24: 1.5 hrs
# work on nlp_predict_prob


#%%

"""
# NEXT STEP: 

    2) try xgboost, lightgbm, autogluon see if there's an approvement
    3) write a code to see the distribution of the sentences containing specific word or bigram
    
 """

#%%
# import spacy
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.sparse import vstack
from pathlib import Path
from sklearn.model_selection import cross_val_score
import joblib

from nltk.corpus import stopwords
from pathlib import Path

portuguese_stop_words = stopwords.words('portuguese')


# For downloading stop words
# import nltk
# nltk.download('stopwords')
# Load data
# Load the Portuguese language model for spaCy
# nlp = spacy.load('pt_core_news_sm')

# Function to perform lemmatization
#%%%
def lemmatize(text,model):
    doc = model(text)
    lemmatized = " ".join([token.lemma_ for token in doc])
    return lemmatized

def nlp_predict(data,model,tfidf_vectorizer,col_input = 'portuguese_lemma', inplace = True):
    import pandas as pd
    # vocab03 = tfidf_vectorizer.vocabulary_

    if isinstance(data, pd.Series):
        data_in = data.copy()
    elif isinstance(data, pd.DataFrame):
        data_in = data[col_input]
        
    data_tfidf = tfidf_vectorizer.transform(data_in)
    prediction = model.predict(data_tfidf)
    
    # vocab04 = tfidf_vectorizer.vocabulary_
    if isinstance(data, pd.Series):
        out_df = pd.DataFrame({'sentence':data, 'prediction':prediction})
        return out_df
    
    elif isinstance(data, pd.DataFrame):
        if inplace:
            data['prediction'] = prediction
            return data
        else:
            out_data = data.copy()
            out_data['prediction'] = prediction
            return out_data
    
    
    return out_df


def nlp_predict_prob(data,model,tfidf_vectorizer,col_input = 'portuguese_lemma', inplace = True):
    # !!! TOFIX when inplace = True, it still doesn't change the original df
    # doesn't seem to be useful if the prob predict is very low and I want to flag it as not sure
    
    import pandas as pd
    # vocab03 = tfidf_vectorizer.vocabulary_

    if isinstance(data, pd.Series):
        data_in = data.copy()
    elif isinstance(data, pd.DataFrame):
        data_in = data[col_input]
        
    data_tfidf = tfidf_vectorizer.transform(data_in)
    prediction = model.predict_proba(data_tfidf)
    
    labels = model.classes_.tolist()
    
    prob_df = pd.DataFrame(prediction, columns=[label + '_prob' for label in labels]).set_index(data.index)
    out_df = nlp_predict(data, model, tfidf_vectorizer,col_input,inplace)
    
    # vocab04 = tfidf_vectorizer.vocabulary_
    if isinstance(data, pd.Series):
        data = pd.concat([out_df,prob_df], axis = 1)
        return out_df
    
    elif isinstance(data, pd.DataFrame):
        if inplace:
            data = pd.concat([out_df,prob_df ], axis = 1)
            return data
        else:
            out_data = pd.concat([out_df,prob_df ], axis = 1)
            return out_data



def ml_upsampling(X_df, y_df, verbose = 1):
    
    import pandas as pd
    import numpy as np
    
    """
    Perform manual upsampling on a dataset to balance class distribution.

    This function upsamples the minority classes in a dataset to match the 
    number of instances in the majority class. It operates by randomly 
    duplicating instances of the minority classes.

    Parameters:
    X_df (pd.DataFrame): DataFrame containing the feature set.
    y_df (pd.Series): Series containing the target variable with class labels.
    verbose: 
        0 print nothing
        1 print out before & after upsampling
    

    Returns:
    list: Contains two elements:
        - pd.DataFrame: The upsampled feature DataFrame.
        - pd.Series: The upsampled target Series.

    Note:
    The function does not modify the input DataFrames directly. Instead, it 
    returns new DataFrames with the upsampled data. The indices of the 
    returned DataFrames are reset to maintain uniqueness.
    """
    
    
    # Determine the majority class and its count
    # TOFIX01: when y_df is dataframe, it still wrong when do the sampling
    
    if verbose == 0:
        pass
    elif verbose == 1:
        print("Before upsampling: ")
        print()
        print(y_df.value_counts())
        print()
    
    y_df_copy = y_df.copy()
    # Initialize the upsampled DataFrames
    X_train_oversampled = X_df.copy()
    y_train_oversampled = y_df.copy()
    
    if isinstance(y_train_oversampled, pd.DataFrame):
        y_train_oversampled = y_df_copy.iloc[:,0]
        
    majority_class = y_train_oversampled.value_counts().idxmax()
    majority_count = y_train_oversampled.value_counts().max()
    
    # Perform manual oversampling for minority classes
    for label in y_train_oversampled.unique():
        if label != majority_class:
            samples_to_add = majority_count - y_df.value_counts()[label]
            indices = y_df_copy[y_df_copy == label].index
            random_indices = np.random.choice(indices, samples_to_add, replace=True)
            X_train_oversampled = pd.concat([X_train_oversampled, X_df.loc[random_indices]], axis=0)
            y_train_oversampled = pd.concat([y_train_oversampled, y_df.loc[random_indices]])

    # Reset index to avoid duplicate indices
    X_train_oversampled.reset_index(drop=True, inplace=True)
    y_train_oversampled.reset_index(drop=True, inplace=True)
    
    if verbose == 0:
        pass
    elif verbose == 1:
        print("After upsampling: ")
        print()
        print(y_train_oversampled.value_counts())
        print()
    
    return [X_train_oversampled, y_train_oversampled]


def nlp_make_tfidf_matrix(X,text_col, ngram_range =(1,1),stop_words = [], max_df = 0.7):
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    if isinstance(X,pd.Series):
        X_in = X.copy()
    elif isinstance(X,pd.DataFrame):
        X_in = X[text_col]
    else:
        raise TypeError("X should only pd.Series or pd.DataFrame as of now")
    
    tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words,ngram_range=ngram_range,max_df=max_df)
    X_tfidf = tfidf_vectorizer.fit_transform(X_in)
    X_out_df = pd.DataFrame(X_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out(), index=X.index)
    
    return [X_out_df,tfidf_vectorizer]

def os_add_extension(ori_path, added_extension, inplace = True):
    # still doesn't work
    # still can't modify the text direclty
    # imported from "C:\Users\Heng2020\OneDrive\Python NLP\NLP 05_UsefulSenLabel\sen_useful_GPT01.py"
    ori_path_in = [ori_path] if isinstance(ori_path, str) else ori_path
    
    # for now I only write added_extension to support only string
    
    outpath = []

    
    if isinstance(added_extension, str):
        added_extension_in = added_extension if "." in added_extension else "." + added_extension
        
        for i,curr_path in enumerate(ori_path):
            if inplace:
                curr_path = curr_path if added_extension in curr_path else curr_path + added_extension_in
                ori_path[i] = curr_path

                
            else:
                curr_path_temp = curr_path if added_extension in curr_path else curr_path + added_extension_in
                outpath.append(curr_path_temp)
    
    if inplace:
        return ori_path
    else:
        # return the string if outpath has only 1 element, otherwise return the whole list
        if len(outpath) == 1:
            return outpath[0]
        else:
            return outpath
        


def confusion_matrix_adj(y_true, y_accept, y_pred, labels=None):
    from sklearn.metrics import confusion_matrix
    """
    Compute a confusion matrix with adjustments.

    Parameters:
    y_true: Array-like of true class labels.
    y_accept: Array-like of acceptable class labels.
    y_pred: Array-like of predicted class labels.
    labels: List of label names corresponding to the classes (optional).

    Returns:
    Confusion matrix as a 2D array.
    """
    # Adjust predictions
    adjusted_pred = []
    for true, accept, pred in zip(y_true, y_accept, y_pred):
        if pred == true or pred == accept:
            adjusted_pred.append(pred)
        else:
            adjusted_pred.append(true)  # Considered as predicted 'true', actual 'true'

    # Compute confusion matrix
    return confusion_matrix(y_true, adjusted_pred, labels=labels)


def plot_confusion_matrix(y_true, y_pred, title,labels = None):
    
    if labels is None:
        cm = confusion_matrix(y_true, y_pred)
    else:
        cm = confusion_matrix(y_true, y_pred, labels = labels)
    plt.figure(figsize=(8, 6))
    
    if labels is None:
        sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=y_true.unique(), yticklabels=y_true.unique())
    else:
        sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=labels, yticklabels=labels)
    
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()


#%%
folder_path = r"C:/Users/Norat/OneDrive/D_Code/Python/Python NLP/NLP 01/NLP 05_UsefulSenLabel"
folder_Path = Path(folder_path)
data_out_folder = r"C:\Users\Norat\OneDrive\D_Code\Python\Python NLP\NLP 01\NLP 05_UsefulSenLabel\data"

out_X_tfidf_name = "BigBangSentenceS06_Tfidf_v01.parquet"
out_X_ngram_name = "BigBangSentenceS06_ngram_v01.parquet"
out_y_name = "BigBangSentenceS06_y_v01.parquet"

out_X_tfidf_path = data_out_folder + "/" + out_X_tfidf_name
out_X_ngram_path = data_out_folder + "/" +  out_X_ngram_name
out_y_path = data_out_folder + "/" +  out_y_name

del out_X_tfidf_name
del out_X_ngram_name
del out_y_name


df_name = 'BigBangSentenceS06_label_ChatGPT.csv'
RANDOM_STATE = 42
UPSAMPLING = True
df_path = folder_Path / df_name
Y_COL_NAME = 'usefulness'
X_COL_NAME = 'portuguese'
NGRAM_RANGE = (1, 2)
CV = 5

#%%
####################################
saved_model_folder = Path(r'C:/Users/Norat/OneDrive/D_Code/Python/Python NLP/NLP 01/NLP 05_UsefulSenLabel/saved_models')

lr_model_name = "Linear_Regression_balanced.joblib"
nb_model_name = "Naive Bayes_balanced"
vectorizer_tfidf_name = "TfidfVectorizer"
vectorizer_ngram_name = "NgramVectorizer"

if ".joblib" not in lr_model_name:
    lr_model_name += ".joblib"
    
if ".joblib" not in nb_model_name:
    nb_model_name += ".joblib"
    
if ".joblib" not in vectorizer_tfidf_name:
    vectorizer_tfidf_name += ".joblib"

if ".joblib" not in vectorizer_ngram_name:
    vectorizer_ngram_name += ".joblib"


lr_model_path = saved_model_folder / lr_model_name
nb_model_path = saved_model_folder / nb_model_name
vectorizer_tfidf_path = saved_model_folder / vectorizer_tfidf_name
vectorizer_ngram_path = saved_model_folder / vectorizer_ngram_name
#------------------------------

#%%
alarm_path = r"H:\D_Music\Sound Effect positive-massive-logo.mp3"


data_ori = pd.read_csv(df_path)
# Drop specified columns and rows with null 'usefulness'
data = data_ori.drop(columns=['episode', 'translation'])

# remove row that has no label
data = data[data[Y_COL_NAME].notnull()]

# data['portuguese_lemma' ] = data['portuguese'].apply(lemmatize)



X_data = data[X_COL_NAME]
y_data = data[[Y_COL_NAME]]



X_tfidf, tfidf_vectorizer = nlp_make_tfidf_matrix(X_data,text_col=X_COL_NAME)
X_ngram, tfidf_vectorizer_ngram = nlp_make_tfidf_matrix(X_data, text_col=X_COL_NAME,ngram_range=NGRAM_RANGE)


X_tfidf.to_parquet(out_X_tfidf_path)
X_ngram.to_parquet(out_X_ngram_path)
y_data.to_parquet(out_y_path)

joblib.dump(tfidf_vectorizer, vectorizer_tfidf_path)
joblib.dump(tfidf_vectorizer_ngram, vectorizer_ngram_path)
# vocab01 = tfidf_vectorizer.vocabulary_

