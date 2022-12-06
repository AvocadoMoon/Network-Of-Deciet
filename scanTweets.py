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
    
    def scrubTweet(self, tweet):
        tweet = str(tweet) #for some reason I need to do this even though it says its type is STRING beforehand
        temp = preprocessor.clean(tweet) #deals with URL's, mentions, and emojis
        temp = re.sub("[^A-Za-z ]","", temp) #if its not A-Z, a-z, or a period than REMOVE IT
        temp = ' '.join(temp.split())
        return temp

    def oneDaysTweets(self, path):
        print(f"On file: {path}/tweets.csv")
        tweetDF = pd.read_csv(f'{path}/tweets.csv')
        columnNames = list(tweetDF.columns)
        columnNames.append("Certainity")
        stereoDF = pd.DataFrame(columns=columnNames)
        stereoIndex = 0
        for rIndex in tqdm(range(len(tweetDF))):
            tweet = tweetDF.loc[rIndex]["Tweet"]
            tweet = self.scrubTweet(tweet)
            catagory = self.catagorizeTweet(tweet)
            catagory = catagory.tolist()
            catagory = catagory[0]
    
            i = catagory.index(max(catagory))
            if (i == 0 and (catagory[0] - catagory[1]) > 3):
                #print(tweet)
                s = tweetDF.loc[rIndex].to_list()
                s.append(catagory[i])
                stereoDF.loc[stereoIndex] = s
                stereoIndex += 1
        stereoDF.to_csv(f'{path}/stereo.csv')
        
    def catagorizeTweet(self, tweet):
        self._set_device()
        self.model.to(self.device)
        tokinizedTweet = self._tokinize_text(self.tokenizer, tweet)

        with torch.no_grad():
            tokinized_text, mask, input_id, token_id = self._prep_input(tokinizedTweet)
            output = self.model(input_id, mask, token_id)

            return output
    
    def calculateRatios(self):
        pass


cwd = os.getcwd()
tweetFolder = f'{cwd}/ScrapedTweets'
# um = UseModel(f'{cwd}/stereotype_detection_model.pt', f'{cwd}/bert-base-uncased')
tweetScan = ScanTweets(f'{cwd}/pickledModel.pickle', f'{cwd}/bert-base-uncased')
for i in os.listdir(tweetFolder):
    tweetScan.oneDaysTweets(f'{tweetFolder}/{i}')