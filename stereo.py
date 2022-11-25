import json
import pandas as pd
import numpy as np
import os
import torch
import tensorflow as tf
from transformers import BertTokenizer, BertModel

path = '/home/zek/School/Information Ecosystem Threats/Network-Of-Deciet' +  "/stereotypes.json"

##########################
## Functions            ##
##########################

#open up the json file and make a data frame out of it
#reformat the data into something more palatable, decomposing the sentences list
def parse_And_Make_DataFrame(which):
    f = open(path)
    data = json.load(f)

    temp_dict_list = []
    inter = data["data"][which]
    for i in inter:
        j = 0
        lis = i.pop("sentences")
        for k in lis:
            i[f"response {j}"] = k["sentence"]
            i[f"label {j}"] = k["gold_label"]
            j += 1
        temp_dict_list.append(i)
    return pd.DataFrame(temp_dict_list)

#make all data tokenized
def tokenize_all_data(df, inputs, tokenizer):
    df_index = 0
    new_df = 0
    df_len = len(df)
    #tokenize the sentences at the following index
    def tok(name, index):
        return tokenizer.encode_plus(
            df[name][index], #text to be tokenized
            max_length = 128,
            truncation=True, #Truncate scentences that are longer than 128 digit representation
            padding='max_length', #pads all sentences to 128 digit representations
            add_special_tokens=True, #Adds special tokens relative to their model
            return_tensors='tf' #returns the algebraic object that describes a multilinear relationship between sets of algebraic objects (aka sentence summary)
        )
        
    while df_index != df_len: #because df_index starts from 0
        first_sentence = tok("response 0", df_index)
        second_sentence = tok("response 1", df_index)
        third_sentence = tok("response 2", df_index)

		#at index n, fill up that whole row
        #print(f"Df_len: {df_len}, df_index: {df_index}, new_df: {new_df}")
        #ids[new_df, :], masks[new_df, :] = first_sentence.input_ids, first_sentence.attention_mask
        #ids[new_df+1, :], masks[new_df+1, :] = second_sentence.input_ids, second_sentence.attention_mask
        #ids[new_df+2, :], masks[new_df+2, :] = third_sentence.input_ids, third_sentence.attention_mask
        
        inputs.append(first_sentence)
        

        df_index+= 1
        new_df+=3
    return inputs, None

# def get_bert_embeddings(inputs, model):
#     """
#     Obtains BERT embeddings for tokens.
#     """
#     # gradient calculation id disabled
#     with torch.no_grad():
#       # obtain hidden states
#       outputs = model(**inputs)
#       hidden_states = outputs[2]
        
#     # concatenate the tensors for all layers use "stack" to create new dimension in tensor
#     token_embeddings = torch.stack(hidden_states, dim=0)
    
#     # remove dimension 1, the "batches"
#     token_embeddings = torch.squeeze(token_embeddings, dim=1)  
      
#     # swap dimensions 0 and 1 so we can loop over tokens
#     token_embeddings = token_embeddings.permute(1,0,2)    
    
#     # intialized list to store embeddings
#     token_vecs_sum = []    
    
#     # "token_embeddings" is a [Y x 12 x 768] tensor
#     # where Y is the number of tokens in the sentence    
#     # loop over tokens in sentence
#     for token in token_embeddings:    
#     # "token" is a [12 x 768] tensor, sum the vectors from the last four layers
#         sum_vec = torch.sum(token[-4:], dim=0)
#         token_vecs_sum.append(sum_vec)
#     return token_vecs_sum

        

##########################
## Data Preparation     ##
##########################

#Make data frames
df_inter = parse_And_Make_DataFrame("intersentence")
df_intra = parse_And_Make_DataFrame("intrasentence")
df_inter.info()


#--[Setting up BERT]--#

#https://huggingface.co/bert-base-uncased?text=The+goal+of+life+is+%5BMASK%5D.
tokenizer = BertTokenizer.from_pretrained('/home/zek/School/Information Ecosystem Threats/Network-Of-Deciet/bert-base-uncased')
bert_model = BertModel.from_pretrained('/home/zek/School/Information Ecosystem Threats/Network-Of-Deciet/bert-base-uncased', output_hidden_states=True)

#Input IDs correlate to the index of tokens within a large lexicon
#attention mask is rows describing which indexes have tokens that should be analyized, which ones are padding and which ones aren't
#Segement type IDs are for context, it gives back the type of grammar it is

#gernerate token and mask IDs
#makes a matrix that is n*3 by 128
#input_ids = np.zeros((len(df_inter) * 3, 128))
#att_masks = np.zeros((len(df_inter) * 3, 128))
inputs = []
input_ids, att_masks = tokenize_all_data(df_inter, inputs, tokenizer)

# response = get_bert_embeddings(inputs[0], bert_model)

# print(response[0].stride())
# print(type(response[0]))





