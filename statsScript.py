import os
import pandas as pd
import matplotlib.pyplot as pyplot

cwd = os.getcwd()
nEpochs = 8
columnsBeingPlotted = {
        "Precision" : "green", 
        "Recall" : "blue", 
        "F1-Score" : "red", 
        "Accuracy" : "black"
}

def ratioOfStereotypes():
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



def modelTestGraph():
    pass


# Load DF, sort rows by Epoch number
# Have a graph where X axis is Epoch number
# The Y axis is the percentage
# Different colors represent different categories (validation, F1 score, etc...)
# Have validation be dotted lines, training solid lines

def extractCSVInformationEpochs(dataFrame):
    epochResultsDF = pd.DataFrame(columns=columnsBeingPlotted.keys())
    #Gets all rows 
    accuracy = None
    for rindex, row in dataFrame.iterrows():
        if (type(row["Catagory"]) == float):
            accuracy = row["Accuracy"]
        if (row["Catagory"] == "Average"):
            l = row.tolist()[1:5]
            l[3] = accuracy #replace the null with the accuracy
            epochResultsDF.loc[len(epochResultsDF)] = l
    print(epochResultsDF)
    return epochResultsDF


def modelEpochGraph():
    trainDF = pd.read_csv(f"{cwd}/Epoch_Results/Training.csv")
    validationDF = pd.read_csv(f"{cwd}/Epoch_Results/Validation.csv")

    trainResultsDF = extractCSVInformationEpochs(trainDF)
    validationResultsDF = extractCSVInformationEpochs(validationDF)

    #Columns: Catagory,Precision,Recall,F1-Score,Accuracy,Support,Epoch

    #The X axis being Epoch numbers
    columns = trainDF.columns
    columns = columns.to_list()

    graph = pyplot

    for i in range(len(columns)):
        if (columns[i] in columnsBeingPlotted.keys()):
            
            #Get all values for 
            print(i)
            training_Y_axis = trainResultsDF[columns[i]]
            validation_Y_axis = validationResultsDF[columns[i]]

            epoch_X_axis = [(j + 1) for j in range(nEpochs)]

            graph.plot(epoch_X_axis, training_Y_axis, color=columnsBeingPlotted[columns[i]],
            linewidth=2, label=columns[i])

            graph.plot(epoch_X_axis, validation_Y_axis, color=columnsBeingPlotted[columns[i]], 
            linestyle='dashed', linewidth = 3, label=f"Validation {columns[i]}")
    
    graph.xlabel("Epochs")
    graph.ylabel("")
    graph.title("Training Results")
    graph.legend()
    graph.savefig('results.jpg')


if __name__ == "__main__":
    #extractCSVInformationEpochs(pd.read_csv(f"{cwd}/Epoch_Results/Training.csv"))
    modelEpochGraph()