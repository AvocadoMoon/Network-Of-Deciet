import json
import pandas as pd
import numpy as np
import tensorflow as tf
import os
from transformers import BertTokenizer, BertModel

path = os.getcwd() +  "\\stereotypes.json"

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
            i[f"sentence {j}"] = k["sentence"]
            i[f"label {j}"] = k["gold_label"]
            j += 1
        temp_dict_list.append(i)
    return pd.DataFrame(temp_dict_list)

def generate_training_data(df, ids, masks, tokenizer):
    i = 0
    n = 0
    df_len = len(df)
    def tok(name, index):
        return tokenizer.encode_plus(
            df_inter[name][index],
            max_length = 128,
            truncation=True,
            padding='max_length',
            add_special_tokens=True,
            return_tensors='tf'
        )
    while i != (df_len * 3):
        first_sentence = tok("sentence 0", i)
        second_sentence = tok("sentence 1", i)
        third_sentence = tok("sentence 2", i)

        ids[n, :], masks[n, :] = first_sentence.input_ids, first_sentence.attention_mask
        ids[n+1, :], masks[n+1, :] = second_sentence.input_ids, second_sentence.attention_mask
        ids[n+2, :], masks[n+2, :] = third_sentence.input_ids, third_sentence.attention_mask

        i+= 1
        n+=3
    return ids, masks

        

##########################
## Data Preparation     ##
##########################

#Make data frames
df_inter = parse_And_Make_DataFrame("intersentence")
df_intra = parse_And_Make_DataFrame("intrasentence")
df_inter.info()


#--[Setting up BERT]--#

#https://huggingface.co/bert-base-uncased?text=The+goal+of+life+is+%5BMASK%5D.
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

#Input IDs correlate to the index of tokens within a large lexicon
#attention mask is rows describing which indexes have tokens that should be analyized, which ones are padding and which ones aren't
#Segement type IDs are for context, it gives back the type of grammar it is
# test = tokenizer.encode_plus(
#     df_inter["sentence 0"][0],
#     max_length = 128,
#     truncation=True,
#     padding='max_length',
#     add_special_tokens=True,
#     return_tensors='tf'
# )
# print(test.input_ids)
# print(test.attention_mask)
# print(test)

#gernerate token and mask IDs
input_ids = np.zeros(len(df_inter * 3), 128)
att_masks = np.zeros(len(df_inter * 3), 128)
input_ids, att_masks = generate_training_data(df_inter, input_ids, att_masks, tokenizer)
