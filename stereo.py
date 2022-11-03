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
