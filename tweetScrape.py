import configparser
import tweepy
import requests
import pandas as pd
import json
import csv
import os
from datetime import date

##########
## Plan ##
##########
#1). Collect streamed unique tweets.
#2). After collection, determine which streamed tweets are stereotypes / hate speech
#3). The tweets that are stereos, get their number of retweets and do impact analysis on them


###############################
## Get Keys and authenticate ##
###############################
config = configparser.ConfigParser()
config.read("config.ini")

api_key = config["twitter"]["api_key"]
api_key_secret = config["twitter"]["api_key_secret"]

access_token_secret = config["twitter"]["access_token_secret"]
access_token = config["twitter"]["access_token"]

bearer_token = config["twitter"]["bearer_token"]

auth = tweepy.OAuthHandler(api_key, api_key_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)
client = tweepy.Client(auth)


##########################
## Tweet Scraping Logic ##
##########################
cwd = os.getcwd()
directory = f"{cwd}/ScrapedTweets/{date.today()}_Tweets"
path = directory + "/tweets.csv"

csvWriter = None
columns = ["Time", "Tweet", "TweetID", "AuthorID", "Username", "Location"]
if (not(os.path.exists(path))):
	os.mkdir(directory)
	csvWriter = csv.writer(open(path, 'w'), delimiter=",")
	csvWriter.writerow(columns)
if (os.path.exists(path)):
	csvWriter = csv.writer(open(path, 'a'), delimiter=",")

class MyStream(tweepy.StreamingClient):

    limit = 50000
    nTweets = 0
    nRetweets = 30000
    def on_connect(self):
        print("Connected!")
	
	#Get only original english tweets then add them to the CSV file
	#Original means no retweets
    def on_data(self, raw_data):
        data = json.loads(raw_data.decode())
        is_retweet = False if "referenced_tweets" not in data["data"] else data["data"]["referenced_tweets"][0]["type"] == "retweeted"
        is_english = False if "lang" not in data["data"] else data["data"]["lang"] == 'en'
        has_location = False if "includes" not in data else "location" in data["includes"]["users"][0]
        
        if (self.nTweets == self.limit):
            print("Disconnecting")
            self.disconnect()
        
        elif is_english and has_location and not is_retweet:
            #make sure that the location taken is from the tweeter
            #print(data)
            assert(data["includes"]["users"][0]["id"] == data["data"]["author_id"])
            row = [data["data"]["created_at"], data["data"]["text"], data["data"]["id"], data["data"]["author_id"], data["includes"]["users"][0]["username"], data["includes"]["users"][0]["location"]]
            try:
                csvWriter.writerow(row)
                self.nTweets += 1
                print(f"Number of tweets recorded is {self.nTweets}")
            except:
                print("Could not add this row")
            


###################################
## Use Of Twitter Scraping Logic ##
###################################
if __name__ == "__main__":
	tweet_fields, user_fields, expansions = ["lang,created_at,id,author_id,referenced_tweets"],["location"], ["author_id"] 

	stream = MyStream(bearer_token)
	stream.sample(tweet_fields=tweet_fields,user_fields=user_fields, expansions=expansions)


#################
## Limitations ##
#################
# Can not confirm location due to the location being grabed from users bio which may be false

# Would be better to sample random tweets within time frame but not possible unless researcher

#Tweets handle is given, but when going back for reference they may be deleted

################
## References ##
################

#https://developer.twitter.com/en/docs/twitter-api/tweets/likes/api-reference/get-users-id-liked_tweets
#https://docs.tweepy.org/en/stable/client.html#tweets
#https://developer.twitter.com/apitools/api?endpoint=%2F2%2Ftweets%2Fsample%2Fstream&method=get
#https://developer.twitter.com/en/docs/twitter-api/tweets/volume-streams/api-reference/get-tweets-sample-stream#tab1
