import argparse 

ap = argparse.ArgumentParser()
ap.add_argument("--validation-dataset", type = str, required = True, help = "Path to validation dataset")

args = vars(ap.parse_args())


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
import pickle

nltk.download('stopwords')
english_stopwords = stopwords.words('english')
hindi_stop_word_file = open("hindi_stopwords.txt")
hindi_stopwords = hindi_stop_word_file.read()
hindi_stopwords = list(hindi_stopwords.split("\n"))
hindi_stop_word_file.close()


VALIDATION_DATASET_PATH = args['validation_dataset']

# Listing categories, subcategories, and missing subcategories in the training dataset.
# Creating a index map for categories and subcategories.
# Categories in the training data. 
categories_list = [
       'Online and Social Media Related Crime', 'Online Financial Fraud',
       'Online Gambling  Betting',
       'RapeGang Rape RGRSexually Abusive Content',
       'Any Other Cyber Crime', 'Cyber Attack/ Dependent Crimes',
       'Cryptocurrency Crime', 'Sexually Explicit Act',
       'Sexually Obscene material',
       'Hacking  Damage to computercomputer system etc',
       'Cyber Terrorism',
       'Child Pornography CPChild Sexual Abuse Material CSAM',
       'Online Cyber Trafficking', 'Ransomware',
       'Report Unlawful Content'
]

# Sub categories in the training data.
sub_categories = [
            'Cyber Bullying  Stalking  Sexting', 'Fraud CallVishing',
           'Online Gambling  Betting', 'Online Job Fraud',
           'UPI Related Frauds', 'Internet Banking Related Fraud',
           'Other', 'Profile Hacking Identity Theft',
           'DebitCredit Card FraudSim Swap Fraud', 'EWallet Related Fraud',
           'Data Breach/Theft', 'Cheating by Impersonation',
           'Denial of Service (DoS)/Distributed Denial of Service (DDOS) attacks',
           'FakeImpersonating Profile', 'Cryptocurrency Fraud',
           'Malware Attack', 'Business Email CompromiseEmail Takeover',
           'Email Hacking', 'Hacking/Defacement',
           'Unauthorised AccessData Breach', 'SQL Injection',
           'Provocative Speech for unlawful acts', 'Ransomware Attack',
           'Cyber Terrorism', 'Tampering with computer source documents',
           'DematDepository Fraud', 'Online Trafficking',
           'Online Matrimonial Fraud', 'Website DefacementHacking',
           'Damage to computer computer systems etc', 'Impersonating Email',
           'EMail Phishing', 'Ransomware', 'Intimidating Email',
           'Against Interest of sovereignty or integrity of India'
    ]
additonal_subcat = [
        'Sexually Explicit Act',
        'Sexually Obscene material',
        'Child Pornography CPChild Sexual Abuse Material CSAM',
        'RapeGang Rape RGRSexually Abusive Content'
    ]


subcategory_list = sub_categories
subcategory_list.extend(additonal_subcat)
subcat_index_map = dict(
        zip(
            subcategory_list,
            range(len(subcategory_list))
        )
    )

cat_index_map = dict(
    zip(
        categories_list,
        range(len(categories_list))
        
    )
)

# Data preparation : Dropping duplicate training and testing data with respect to crimeaditionalinfo column.
# Removing rows with blank crimeaditionalinfo value.
# Filling missing subcategories with categories
# Ensure all text data is in string format
# ***** We are only considering the sub categories, and sub categories in the training data. *******

def get_info_and_labels(fpath : str, flag : bool = False):
    df = pd.read_csv(fpath)
    print(f"[*] Total Number of samples :: {df.shape[0]}")
    if flag:
        df = df.drop_duplicates(['crimeaditionalinfo'])
    df = df[df['crimeaditionalinfo'].notnull()]
    print(f"[*] Number of samples after removing empty, duplicate *Crime Additional Info*  :: {df.shape[0]}")

    x, y_category, y_subcategory = [], [], []
    for sub_cat in sub_categories:
        sub_cat_df = df.query(f"sub_category == '{sub_cat}'")
        x_content = sub_cat_df['crimeaditionalinfo'].values
        y_sub_cat_values = sub_cat_df['sub_category'].values
        y_cat_values = sub_cat_df['category'].values
        x.extend(x_content)
        y_subcategory.extend(y_sub_cat_values)
        y_category.extend(y_cat_values)
        
        
    for sub_cat in additonal_subcat:
        sub_cat_df = df.query(f"category == '{sub_cat}'")
        x_content = sub_cat_df['crimeaditionalinfo'].values
        y_sub_cat_values = sub_cat_df['category'].values
        y_cat_values = sub_cat_df['category'].values
        x.extend(x_content)
        y_subcategory.extend(y_sub_cat_values)
        y_category.extend(y_cat_values)

    return x, y_category, y_subcategory

def filter_fxn(x):
    if x == '' or len(x) <= 2 or len(x) > 15:
        return False
    if (x in english_stopwords) or (x in hindi_stopwords):
        return False
    return True


def transform_text(text):
    text = text.lower()
    valid_char = "abcdefghijklmnopqrstuvwxyz" + " "
    escape_char_list = ["\r", "\n", "\b", "\t"]
    for escape_char in escape_char_list:
        text = text.replace(escape_char, " ")
        
    for c in text:
        if c not in valid_char:
            text = text.replace(c, "")
    text_list = text.split(" ")
    text_list = list(filter(filter_fxn, text_list))
    return " ".join(text_list)

x_val, y_val_cat, y_val_subcat = get_info_and_labels(VALIDATION_DATASET_PATH, True)
model = tf.keras.models.load_model("CrimeReportTextClassification.keras")
from_disk = pickle.load(open("text_vector.pkl", "rb"))
text_vectorization = tf.keras.layers.TextVectorization.from_config(from_disk['config'])
text_vectorization.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))
text_vectorization.set_weights(from_disk['weights'])
x_val = list(map(transform_text, x_val))
y_val_cat_labels = []
y_val_subcat_labels = []
for sub_cat in y_val_subcat:
    y_val_subcat_labels.append(subcat_index_map[sub_cat])
for cat in y_val_cat:
    y_val_cat_labels.append(cat_index_map[cat])
y_val_cat = tf.keras.utils.to_categorical(y_val_cat_labels, num_classes = len(cat_index_map))
y_val_subcat = tf.keras.utils.to_categorical(y_val_subcat_labels, num_classes = len(subcat_index_map))   
x_val = text_vectorization(x_val)
pred_cat, pred_subcat = model.predict(x_val)

print("Category classification Accuracy: ", accuracy_score(np.argmax(y_val_cat, axis = 1), np.argmax(pred_cat, axis = 1)))
print("Category classification Precision: ", precision_score(np.argmax(y_val_cat, axis = 1), np.argmax(pred_cat, axis = 1), average="micro"))
print("Category classification Recall: ", recall_score(np.argmax(y_val_cat, axis = 1), np.argmax(pred_cat, axis = 1), average="micro"))
print("Category classification F1-score: ", f1_score(np.argmax(y_val_cat, axis = 1), np.argmax(pred_cat, axis = 1), average="micro"))
print()
print("Subcategory classification Accuracy: ", accuracy_score(np.argmax(y_val_subcat, axis = 1), np.argmax(pred_subcat, axis = 1)))
print("Subcategory classification Precision: ", precision_score(np.argmax(y_val_subcat, axis = 1), np.argmax(pred_subcat, axis = 1), average="micro"))
print("Subcategory classification Recall: ", recall_score(np.argmax(y_val_subcat, axis = 1), np.argmax(pred_subcat, axis = 1), average="micro"))
print("Subcategory classification F1-score", f1_score(np.argmax(y_val_subcat, axis = 1), np.argmax(pred_subcat, axis = 1), average="micro"))

