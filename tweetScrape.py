import configparser
import random
import tweepy
import requests
import pandas as pd
import numpy
import json


#Initialize Twitter Scraper

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

tweet_fields = ["lang,created_at,id,author_id"]
place_fields = ["country_code"]
user_fields = ["location"]
expansions = ["author_id"]

df = pd.DataFrame(columns=["Time", "Tweet", "ID", "AuthorID"])

def getCountryCode(ip):
    endpoint = f'https://ipinfo.io/{ip}/json'
    response = requests.get(endpoint, verify = True)

    if response.status_code != 200:
        return 'Status:', response.status_code, 'Problem with the request. Exiting.'
        exit()

    data = response.json()
    return data['country']

class MyStream(tweepy.StreamingClient):

    limit = 100
    def on_connect(self):
        print("Connected!")
    
    #Get only english tweets, and then 
    def on_tweet(self, tweet):
        #print(tweet.data)
        
        # print(api.get_user(tweet.author_id))
        if(len(df.index) > self.limit):
            print("Disconnected")
            self.disconnect()
        elif tweet.lang == "en": #and tweet.geo.country_code == "US":
            twit = [tweet.created_at, tweet.text, tweet.id, tweet.author_id]

            #add new row to end of data base
            df.loc[len(df.index)] = twit
            print(len(df.index))

    

stream = MyStream(bearer_token)
stream.sample(tweet_fields=tweet_fields,place_fields=place_fields, user_fields=user_fields, expansions=expansions)

df.info()

#Due to difficulty of randomly sampling tweets from certain dates, using Tweet ID, streaming random Tweets to get current sentiment


#tweet data structure https://developer.twitter.com/en/docs/twitter-api/tweets/likes/api-reference/get-users-id-liked_tweets
#tweet look up https://docs.tweepy.org/en/stable/client.html#tweets

  
