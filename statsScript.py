import os
import pandas as pd

cwd = os.getcwd()

scrapedTweets = f"{cwd}/ScrapedTweets"

totalNTweets = 0
totalNStereo = 0

for k in os.listdir(scrapedTweets):
    totalTweets = pd.read_csv(f"{scrapedTweets}/{k}/tweets.csv")
    totalStereo = pd.read_csv(f"{scrapedTweets}/{k}/stereo.csv")
    sLen = len(totalStereo)
    tLen = len(totalTweets)
    print(f'''------------------\n Total sterotypes: {sLen} || Total Tweets: {tLen} ||
Ratio: {sLen / tLen}''')
    totalNTweets += tLen
    totalNStereo += sLen

print(f"Total Number of Tweets: {totalNTweets} || Total Number of Stereos: {totalNStereo} \n Ratio: {totalNStereo/totalNTweets}")