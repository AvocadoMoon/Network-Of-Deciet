import configparser
import tweepy


#Initialize Twitter Scraper

config = configparser.ConfigParser()
config.read("config.ini")

api_key = config["twitter"]["api_key"]
api_key_secret = config["twitter"]["api_key_secret"]

access_token_secret = config["twitter"]["access_token_secret"]
access_token = config["twitter"]["access_token"]

# bearer_token = config["twitter"]["bearer_token"]

auth = tweepy.OAuthHandler(api_key, api_key_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)
client = tweepy.Client(auth)


#Utilize twitter scraper

res = api.search_tweets("t", )
client.twee
print(len(res))

print(res[0].text) #text of the tweet
print(res[0].id) #unique author ID of the tweet
print(res[0].created_at) #time in which tweet was created



#tweet data structure https://developer.twitter.com/en/docs/twitter-api/tweets/likes/api-reference/get-users-id-liked_tweets
#tweet look up https://docs.tweepy.org/en/stable/client.html#tweets

  
