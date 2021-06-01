import pandas as pd
import numpy as np
import pickle
import logging
import json

import os
import transformers
import torch
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import torch.utils.data as data_utils

# Change directory
os.chdir(r"C:/Users/Joel\Downloads/cs229_kickstarter-main/")

import data_clean_for_model
import PipelineHelper

logger = logging.getLogger(__name__)

## 1. Load Data
df = pd.read_parquet("data/all_processed_df.parquet.gzip")

k = 5
rseed = 229
df["outcome"] = np.where( df["state"]=="successful", 1, 0, )
df["un_id"] = np.arange(0, df.shape[0], 1 )
df["name_len"] = df["name"].str.len()
df["cv_group"] = np.random.choice( np.arange(0, k), size=df.shape[0] )
df["binned_usd_goal"] = pd.qcut( np.log(df["usd_goal"]+1), 20 )

with open("model_config.json", 'r') as j:
     model_params = json.loads(j.read())
model_params['naive_bayes']['ngram_range'] = tuple(model_params['naive_bayes']['ngram_range'])


## load project metadata
logger.info("Loading features")
try:
    f = open("data/features.pkl", "rb")
    ft_dict = pickle.load(f)
    f.close()
    X_train, y_train, X_test, y_test = ft_dict.values()
except:
    X_train, X_test, y_train, y_test = data_clean_for_model.data_clean_for_model(df, "outcome", model_params, cv=model_params["cv"])

# load text
logger.info("Processing text data")
blurb_train, blurb_test, y_blurb_train, y_blurb_test    = data_clean_for_model.process_blurb(df, model_params)

## BERT
# Set up GPU for parallel processing
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

vocab = 'C:/Users/Joel/Documents/Personal Development/Academic/Computer Science/venv/cs_229_proj/Data/vocab.txt'

# Prep blurbs
sentences_train = ["[CLS] " + blurb + " [SEP]" for blurb in blurb_train]
sentences_test = ["[CLS] " + blurb + " [SEP]" for blurb in blurb_test]

# Clean and Tokenize blurbs using BERT tokenizer
# See
#     https://huggingface.co/transformers/_modules/transformers/models/bert/tokenization_bert.html#BertTokenizer
# for source code
tokenizer = transformers.BertTokenizer(vocab)

tokenized_blurbs_train = [tokenizer.tokenize(sent) for sent in sentences_train]
tokenized_blurbs_test = [tokenizer.tokenize(sent) for sent in sentences_test]

# Set the maximum sequence length.
MAX_LEN = 128

input_ids_train = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_blurbs_train],
    maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
input_ids_test = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_blurbs_test],
    maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

# Use the BERT tokenizer to convert the tokens to their index numbers in the BERT vocabulary
input_ids_train = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_blurbs_train]
input_ids_train = pad_sequences(input_ids_train, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

input_ids_test = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_blurbs_test]
input_ids_test = pad_sequences(input_ids_test, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")


# Create attention masks
attention_masks_train = []
attention_masks_test = []

# Create a mask of 1s for each token followed by 0s for padding
for seq in input_ids_train:
  seq_mask_train = [float(i>0) for i in seq]
  attention_masks_train.append(seq_mask_train)

for seq in input_ids_test:
    seq_mask_test = [float(i>0) for i in seq]
    attention_masks_test.append(seq_mask_test)

# Convert all of our data into torch tensors, the required datatype for our model
train_inputs = torch.tensor(input_ids_train)
test_inputs = torch.tensor(input_ids_test)
train_labels = torch.tensor(y_blurb_train)
test_labels = torch.tensor(y_blurb_test)
train_masks = torch.tensor(attention_masks_train)
test_masks = torch.tensor(attention_masks_test)

# Select a batch size for training. 
batch_size = 32

# Create an iterator of our data with torch DataLoader
train_data = data_utils.TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = data_utils.RandomSampler(train_data)
train_dataloader = data_utils.DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
test_data = data_utils.TensorDataset(test_inputs, test_masks, test_labels)
test_sampler = data_utils.SequentialSampler(test_data)
test_dataloader = data_utils.DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

# Load BertForSequenceClassification, the pretrained BERT model with a single linear classification layer on top. 
model = transformers.BertForSequenceClassification.from_pretrained("bert-base-uncased")
#, num_labels=nb_labels)
model.cuda()

# BERT fine-tuning parameters
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]

optimizer = transformers.BertAdam(optimizer_grouped_parameters,
                     lr=2e-5,
                     warmup=.1)

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)
  
# Store our loss and accuracy for plotting
train_loss_set = []
# Number of training epochs 
epochs = 4

# BERT training loop
for _ in trange(epochs, desc="Epoch"):  
  
  ## TRAINING
  
  # Set our model to training mode
  model.train()  
  # Tracking variables
  tr_loss = 0
  nb_tr_examples, nb_tr_steps = 0, 0
  # Train the data for one epoch
  for step, batch in enumerate(train_dataloader):
    # Add batch to GPU
    batch = tuple(t.to(device) for t in batch)
    # Unpack the inputs from our dataloader
    b_input_ids, b_input_mask, b_labels = batch
    # Clear out the gradients (by default they accumulate)
    optimizer.zero_grad()
    # Forward pass
    loss = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
    train_loss_set.append(loss.item())    
    # Backward pass
    loss.backward()
    # Update parameters and take a step using the computed gradient
    optimizer.step()
    # Update tracking variables
    tr_loss += loss.item()
    nb_tr_examples += b_input_ids.size(0)
    nb_tr_steps += 1
  print("Train loss: {}".format(tr_loss/nb_tr_steps))
       
  ## TEST

  # Put model in evaluation mode
  model.eval()
  # Tracking variables 
  eval_loss, eval_accuracy = 0, 0
  nb_eval_steps, nb_eval_examples = 0, 0
  # Evaluate data for one epoch
  for batch in test_dataloader:
    # Add batch to GPU
    batch = tuple(t.to(device) for t in batch)
    # Unpack the inputs from our dataloader
    b_input_ids, b_input_mask, b_labels = batch
    # Telling the model not to compute or store gradients, saving memory and speeding up test
    with torch.no_grad():
      # Forward pass, calculate logit predictions
      logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)    
    # Move logits and labels to CPU
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()
    tmp_eval_accuracy = flat_accuracy(logits, label_ids)    
    eval_accuracy += tmp_eval_accuracy
    nb_eval_steps += 1
  print("test Accuracy: {}".format(eval_accuracy/nb_eval_steps))

# plot training performance
plt.figure(figsize=(15,8))
plt.title("Training loss")
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.plot(train_loss_set)
plt.show()

