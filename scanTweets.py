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
import pandas as pd
import csv

class ScanTweets(ModelHelper):
    def __init__(self, model_path, tokinizer_path):
        self.model = torch.load(model_path)
        # self.model.load_state_dict(torch.load(model_path))
        #self.model.eval()
        self.tokenizer = BertTokenizer.from_pretrained(tokinizer_path)
    
    def scrubAndTokinize(self, tweet):
        tweet = str(tweet) #for some reason I need to do this even though it says its type is STRING beforehand
        temp = preprocessor.clean(tweet) #deals with URL's, mentions, and emojis
        temp = re.sub("[^A-Za-z. ]","", temp) #if its not A-Z, a-z, or a period than REMOVE IT
        return self._tokinize_text(self.tokenizer, temp)

    def oneDaysTweets(self, path):
        tweetDF = pd.read_csv(f'{path}/tweets.csv')
        columnNames = list(tweetDF.columns)
        columnNames.append("Certainity")
        stereoDF = pd.DataFrame(columns=columnNames)
        stereoIndex = 0
        for rIndex in tqdm(range(len(tweetDF))):
            tweet = tweetDF.loc[rIndex]["Tweet"]
            tweet = self.scrubAndTokinize(tweet)
            catagory = self.catagorizeTweet(tweet)
            if (catagory.argmax(dim=1).item() == 0):
                print(tweet)
                stereoDF.loc[stereoIndex] = tweetDF.loc[rIndex]
                stereoDF[stereoIndex]["Certainity"] = catagory.tolist()
                stereoIndex += 1
        stereoDF.to_csv(f'{path}/stereo.csv')
        
    def catagorizeTweet(self, tweet):
        self._set_device()
        tokinized_text = self.scrubAndTokinize(tweet)
        self.model.to(self.device)

        with torch.no_grad():
            tokinized_text, mask, input_id, token_id = self._prep_input(tokinized_text)
            output = self.model(input_id, mask, token_id)

            return output


cwd = os.getcwd()
tweetFolder = f'{cwd}/ScrapedTweets'
# um = UseModel(f'{cwd}/stereotype_detection_model.pt', f'{cwd}/bert-base-uncased')
tweetScan = ScanTweets(f'{cwd}/pickledModel.pickle', f'{cwd}/bert-base-uncased')
for i in os.listdir(tweetFolder):
    tweetScan.oneDaysTweets(i)