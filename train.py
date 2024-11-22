import os
os.chdir(os.getcwd())

import argparse
parser = argparse.ArgumentParser(description='training script')
parser.add_argument('--train-data', dest='train_dataset', default="./TrainingData/training_dataset.csv", required = False)
parser.add_argument('--save-model', dest='save_model', default="CrimeReportTextClassification.keras", required = False)
parser.add_argument('--drop-duplicate', dest='drop_duplicate',default = True, required = False)
parser.add_argument("--text-vocab-path", dest = "text_vocab_path", default = "text_vector.pkl", required = False)
args = parser.parse_args()

# Importing all the necessary libraries
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


# Stopwords for english and hindi language
nltk.download('stopwords')
english_stopwords = stopwords.words('english')
hindi_stop_word_file = open("hindi_stopwords.txt")
hindi_stopwords = hindi_stop_word_file.read()
hindi_stopwords = list(hindi_stopwords.split("\n"))
hindi_stop_word_file.close()

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

def get_info_and_labels(fpath : str, flag : bool = True):
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



# Preparing string datasets for train
x_train, y_train_cat, y_train_subcat = get_info_and_labels(args.train_dataset, args.drop_duplicate)


# Text pre-processing : Lowercase the strings.
# Removing certain escape characters.
# Removing all characters other than lowercase alphabets(a-z).
# Removing extra white spaces between words.
# Removing stopwords from the strings.
# Note : Additional complex pre-processing like lemmatization, stemming and other standard techniques on english dataset is not applied because of Hinglish words being present in the dataset which causes unforeseen results

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

x_train = list(map(transform_text, x_train))


# Preparing appropriate labels for categories and subcategories, for train data.

y_train_cat_labels = []
y_train_subcat_labels = []

for sub_cat in y_train_subcat:
    y_train_subcat_labels.append(subcat_index_map[sub_cat])
for cat in y_train_cat:
    y_train_cat_labels.append(cat_index_map[cat])


y_train_cat = tf.keras.utils.to_categorical(y_train_cat_labels, num_classes = len(cat_index_map))# labels
y_train_subcat = tf.keras.utils.to_categorical(y_train_subcat_labels, num_classes = len(subcat_index_map))   


# Creating Bag-of-words corpus vectors for pre-processing
# Transforming the train data to text vectors based on bag-of-words approach.

text_vectorization = tf.keras.layers.TextVectorization(
    max_tokens = 10000,
    output_mode = 'multi_hot',
    ngrams = 2,
    pad_to_max_tokens=True
    
)
text_vectorization.adapt(x_train)
x_train = text_vectorization(x_train)
pickle.dump({'config': text_vectorization.get_config(),
             'weights': text_vectorization.get_weights()}
            , open(args.text_vocab_path, "wb"))

# Building deep learning model architecture from scratch.
# Architecture contains 1 Input layer, 1 hidden layer, 2 classification heads(for category and subcategory classification each)
# Lr schedular was used for dynamic learning rate.
# Adam optimizer is used

input = tf.keras.Input(shape = (x_train.shape[1]), name = 'input_seq')
output = Dense(1024, activation = 'relu',kernel_regularizer = tf.keras.regularizers.l2(0.01))(input)
cat_output = Dense(y_train_cat.shape[1], activation = 'softmax', name = 'cat_output')(output)
subcat_output = Dense(y_train_subcat.shape[1], activation = 'softmax', name = 'sub_cat_output')(output)
model = tf.keras.models.Model(input, [cat_output, subcat_output])


lr_schedule =tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=4000,
    decay_rate=0.9)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

model.compile(
    optimizer=optimizer,
    loss = {
        'cat_output': "categorical_crossentropy",
        'sub_cat_output': "categorical_crossentropy"
    },
    metrics = {
        'cat_output': ["acc"],
        'sub_cat_output': ["acc"]
    }
)
print(model.summary())


# Creating proper dataloader
def format_datset(x_seq, y_cat, y_subcat):
    return (
        x_seq,
        {
            "cat_output": y_cat,
            "sub_cat_output": y_subcat
        }
    )

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train_cat, y_train_subcat)).batch(32).shuffle(2028)
train_dataset = train_dataset.map(format_datset)


# Creating checkpoint callback.
# Training the model on train dataset while validating on test dataset.
# Evaluating model on test dataset
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        args.save_model,
        save_best_only = True,
    )
    
]

history = model.fit(train_dataset, epochs=100, callbacks = callbacks,verbose=1)



# Evaluating accuracy score, precision score, recall score, f1 score for both category and subcategory classification.

pred_cat, pred_subcat = model.predict(format_datset(x_train, y_train_cat, y_train_subcat)[0])

print("Category classification Accuracy: ", accuracy_score(np.argmax(y_train_cat, axis = 1), np.argmax(pred_cat, axis = 1)))
print("Category classification Precision: ", precision_score(np.argmax(y_train_cat, axis = 1), np.argmax(pred_cat, axis = 1), average="micro"))
print("Category classification Recall: ", recall_score(np.argmax(y_train_cat, axis = 1), np.argmax(pred_cat, axis = 1), average="micro"))
print("Category classification F1-score: ", f1_score(np.argmax(y_train_cat, axis = 1), np.argmax(pred_cat, axis = 1), average="micro"))
print()
print("Subcategory classification Accuracy: ", accuracy_score(np.argmax(y_train_subcat, axis = 1), np.argmax(pred_subcat, axis = 1)))
print("Subcategory classification Precision: ", precision_score(np.argmax(y_train_subcat, axis = 1), np.argmax(pred_subcat, axis = 1), average="micro"))
print("Subcategory classification Recall: ", recall_score(np.argmax(y_train_subcat, axis = 1), np.argmax(pred_subcat, axis = 1), average="micro"))
print("Subcategory classification F1-score", f1_score(np.argmax(y_train_subcat, axis = 1), np.argmax(pred_subcat, axis = 1), average="micro"))

plt.plot(history.history['cat_output_loss'], label='Category Training Loss')
plt.plot(history.history['sub_cat_output_loss'], label='Sub Category Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.plot(history.history['cat_output_acc'], label='Category Training Accuracy')
plt.plot(history.history['sub_cat_output_acc'], label='Sub Category Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
