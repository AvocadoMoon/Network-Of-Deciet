import json
import pandas as pd
import numpy as np
import os
import torch
from torch import nn
from trainModel import BertClassifier, ModelHelper
from transformers import BertTokenizer, BertModel
from torch.optim import Adam
from tqdm import tqdm
import re
import preprocessor

class ScanTweets(ModelHelper):
    def __init__(self, model_path, tokinizer_path):
        self.model = torch.load(model_path)
        # self.model.load_state_dict(torch.load(model_path))
        #self.model.eval()
        self.tokenizer = BertTokenizer.from_pretrained(tokinizer_path)
    
    def srubTweets(self, tweet):
        if type(tweet) == np.float:
            return ""
        temp = preprocessor.clean(tweet) #deals with URL's, mentions, and emojis

        temp = re.sub('[()]', ' ', temp) #remove paranthesis 

        #WHAT IS THIS 
        temp = re.sub('\[.*?\]',' ', temp) 
        temp = re.sub("[^a-z0-9]"," ", temp)
        ##

        temp = re.sub(r'[0-9]', '', temp) #removes digits
        return temp

    def oneDaysTweets(self):
        pass

    #The data type input is the same so WHY IS IT NOT WORKING
    #The output is just [nan, nan] wtf???
    def testScentence(self, scentence):
        self._set_device()
        tokinized_text = self._tokinize_text(self.tokenizer, scentence)
        
        self.model.to(self.device)

        with torch.no_grad():
            tokinized_text, mask, input_id = self._prep_input(tokinized_text)
            output = self.model(input_id, mask)

            print(output)
            print(f'{"Stereotype" if output.argmax(dim=1).item() == 0 else "Not stereotype"}')

cwd = os.getcwd()
tweetFolder = f'{cwd}/ScrapedTweets'
# um = UseModel(f'{cwd}/stereotype_detection_model.pt', f'{cwd}/bert-base-uncased')
tweetScan = ScanTweets(f'{cwd}/pickledModel.pickle', f'{cwd}/bert-base-uncased')

test_tweet = '''How to not practice emotional distancing during social distancing. @HarvardHealth https://www.t.co/dSXhPqwywW #HarvardHealth https://t.co/H9tfffNAo0'''