import json
import pandas as pd
import numpy as np
import os
import torch
from torch import nn
from transformers import BertTokenizer, BertModel
from torch.optim import Adam
from tqdm import tqdm

path = os.getcwd()

#responsible for accessing and processing single instances of data
class Dataset(torch.utils.data.Dataset):

    def __init__(self, df, labels=None, tokenizer=None):

        #each label that is within the data frame is converted to its numeric representation, and the index within this
        #list corresponds to the index within the self.texts list
        self.labels = [labels[label] for label in df['label']]

        #tokenize all text that is given, and store it within a singular list
        self.texts = [tokenizer(text, 
                               padding='max_length', max_length = 512, truncation=True,
                                return_tensors="pt") for text in df['text']]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y

class BertClassifier(nn.Module):

    def __init__(self, dropout=0.5):

        super(BertClassifier, self).__init__()

        #bert model
        self.bert = BertModel.from_pretrained(f'{path}/bert-base-uncased')

        #dropout probability set
        self.dropout = nn.Dropout(dropout)

        #input of neural network is a 768 dimensional vector and output is two dimensions, in accordiance to number of labels
        self.linear = nn.Linear(768, 2)

        #the nerual network activation function (ReLU curve)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        #get the response from the bert model, first variable is embedding vectors for tokens in sentence,
        #second is for embedding vector of the [CLS] token which is sentence-level classification and what matters for this classifier
        sentence_vectors, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)

        #first take the input and run it through the dropout function, determine which information is masked
        dropout_output = self.dropout(pooled_output)

        #run the returned output through the single layer neural network
        linear_output = self.linear(dropout_output)

        #take the output and run it through the activation function
        final_layer = self.relu(linear_output)

        return final_layer

class TrainAndEvaluate():
    def __init__(self, model, train_data, test_data, val_data, learning_rate=1e-6, epochs=5, batch_size=2):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.test_data = test_data
        self.batch_size = batch_size


    def train(self):

        #collects data in batches, and returns them for consumption in the training loop
        train_dataloader = torch.utils.data.DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
        val_dataloader = torch.utils.data.DataLoader(self.val_data, batch_size=self.batch_size)

        #check if cuda cores are available to utilize
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

        #Cross entropy loss function
        criterion = nn.CrossEntropyLoss()

        #Adam based optimizing function that alters the weights
        optimizer = Adam(self.model.parameters(), lr= self.learning_rate)

        if use_cuda:
                self.model = self.model.cuda()
                criterion = criterion.cuda()

        for epoch_num in range(self.epochs):

                total_acc_train = 0
                total_loss_train = 0

                #tdqm makes a progress bar for every item itterated
                for train_input, train_label in tqdm(train_dataloader):

                    train_label = train_label.to(device)
                    mask = train_input['attention_mask'].to(device)
                    input_id = train_input['input_ids'].squeeze(1).to(device)

                    #output is a probability matrix 
                    output = self.model(input_id, mask)
                    
                    batch_loss = criterion(output, train_label.long())
                    total_loss_train += batch_loss.item()
                    
                    acc = (output.argmax(dim=1) == train_label).sum().item()
                    total_acc_train += acc

                    self.model.zero_grad()
                    batch_loss.backward()
                    optimizer.step()
                
                total_acc_val = 0
                total_loss_val = 0

                #disable gradient calculations for validation data set
                with torch.no_grad():

                    for val_input, val_label in val_dataloader:

                        val_label = val_label.to(device)
                        mask = val_input['attention_mask'].to(device)
                        input_id = val_input['input_ids'].squeeze(1).to(device)

                        output = self.model(input_id, mask)

                        batch_loss = criterion(output, val_label.long())
                        total_loss_val += batch_loss.item()
                        
                        acc = (output.argmax(dim=1) == val_label).sum().item()
                        total_acc_val += acc
                
                print(
                    f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(self.train_data): .3f} \
                    | Train Accuracy: {total_acc_train / len(self.train_data): .3f} \
                    | Val Loss: {total_loss_val / len(self.val_data): .3f} \
                    | Val Accuracy: {total_acc_val / len(self.val_data): .3f}')

    def evaluate(self):


        test_dataloader = torch.utils.data.DataLoader(self.test_data, batch_size=2)

        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

        if use_cuda:

            self.model = self.model.cuda()

        total_acc_test = 0
        with torch.no_grad():

            for test_input, test_label in test_dataloader:

                test_label = test_label.to(device)
                mask = test_input['attention_mask'].to(device)
                input_id = test_input['input_ids'].squeeze(1).to(device)

                output = self.model(input_id, mask)

                acc = (output.argmax(dim=1) == test_label).sum().item()
                total_acc_test += acc
        
        print(f'Test Accuracy: {total_acc_test / len(self.test_data): .3f}')




if __name__ == "__main__":
    #______-Functions-______#
    dataset_path = path +  "/stereotypes.json"

    #open up the json file and make a data frame out of it
    #reformat the data into something more palatable, decomposing the sentences list
    def parse_And_Make_DataFrame(which):
        f = open(dataset_path)
        data = json.load(f)

        temp_dict_list = []
        inter = data["data"][which]
        for bundle in inter:
            lis = bundle.pop("sentences")
            context = bundle.pop("context")
            for k in lis:
                if k["gold_label"] != "anti-stereotype":
                    t = k["sentence"]
                    temp_dict_list.append({
                        "text": f"{context} {t}",
                        "label" : k["gold_label"]
                    })
        return pd.DataFrame(temp_dict_list)


    #______-Data Prep-______#
    labels = {
        'stereotype':0,
        'unrelated':1
    }
    tokenizer = BertTokenizer.from_pretrained(f'{path}/bert-base-uncased')

    #Make data frames
    df_inter = parse_And_Make_DataFrame("intersentence")
    df_intra = parse_And_Make_DataFrame("intrasentence")
    df_inter.info()

    #Split df into df[:.8], df[.8:.9], df[.9:]
    #Esentially 80%, 10%, 10%
    df_train, df_val, df_test = np.split(df_inter, [int(.8*len(df_inter)), int(.9*len(df_inter))])
    true_dataset = lambda x: Dataset(x, labels, tokenizer)
    df_train, df_val, df_test = true_dataset(df_train), true_dataset(df_val), true_dataset(df_test)

    #______-Classifier Creation, Training, and Evaluation-______#
    classifierModerl = BertClassifier()
    tt = TrainAndEvaluate(classifierModerl, df_train, df_test, df_val, epochs=3)
    tt.train()
    tt.evaluate()

    #______-Save the trained model-______#
    #This method makes it so that 
    save_path = f'{path}/stereotype_detection_model.pt'
    pickle_path = f'{path}/pickledModel.pickle'
    torch.save(classifierModerl.state_dict(), save_path)
    torch.save(classifierModerl, pickle_path)



###############
## Knowledge ##
###############
# Pytourch is utilized for low level operations with ML, and is typically used in research, annoying but at least every detail is known
# Tensorflow is more oriented towards high level use, is a pain in the ass to use due to its high level, don't know whats happening
# Conda is a package manager for R and python for developing data scientests
# Published papers are a good source for technical detail once a basis is understood, but before reading papers make sure to get a good basis

#======BERT======#
#https://huggingface.co/bert-base-uncased?text=The+goal+of+life+is+%5BMASK%5D.
#Input IDs correlate to the index of tokens within a large lexicon
#attention mask is rows describing which indexes have tokens that should be analyized, which ones are padding and which ones aren't
#Segement type IDs are for context, it gives back the type of grammar it is
#<---- BERT Offline ---->
'''BERT offline is significantly less annoying to use on a local machine. This way it does not need to download
the BERT setup everytime the script is ran. If BERT gets updated then thats an issue but :/'''


#======Crash course on NN======#
#These series slap HARD
#https://www.youtube.com/watch?v=gZmobeGL0Yg&list=PLZbbT5o_s2xq7LwI2y8_QtvuXZedL6tQU
#https://www.youtube.com/watch?v=CqOfi41LfDw&list=PLblh5JKOoLUIxGDQs4LFFD--41Vzf-ME1

#<---- Activation Function ---->
'''Transforms the sum of weights into some value which is sandwhiched between an upper and lower limit'''
'''The idea behind it is to simulate the activation of a neuron within your brain, so either on or off.
Some activation functions don't follow this principle such as ReLU which allows the input value to go beyond 1.'''

#<---- Training Neural Network ---->
'''Weights are constantly being optimized to find the optimal values which allow for the least amount of loss (error).'''
'''A method to do this is through derivatives. Take the derivative weights within the model with respect to the loss,
then multiple it by the learning rate. Finally add that new value to the weight within the NN'''

#<---- Learning Rate ---->
'''The steps taken to minimize the loss within the NN'''

#<---- Training, Testing, Validation Set ---->
'''The training set is the data that would be utilized to train the neural network during each epoch'''

'''The validation set is used to make sure that the model is not overfitting. Used to make sure that the NN is not
just getting really good with the training set. If it is able to peform well with both the training set and validation
set then we know we're good.'''

'''Used to test model after it is done training, is unlabled when entering'''

#<---- Loss Functions ---->
'''Loss functions are used to measure the eror between the predicted values vs the provided target values '''
'''Can have different type of loss functions such as, regression (used for predicting continous values),
classification (used when predicting discrete values), ranking (used when predicting relative values)'''
'''Affects the weights within the NN'''

#<---- Epoch ---->
'''One single pass of all the data to the network'''

#<---- Gradient Decent ---->
'''The process of minimizing our loss (or error) by tweaking the weights and biases in our model'''

#<---- Batch Size ---->
'''Number of samples that will be passed through to the network at one time'''
'''Used to help make the NN run faster in certain cases but can also cause for worse predictions'''

#<---- Drop out ---->
'''During training, randomly zero out some of the elements of the input tensor with probability p using samples from
a bernoulli distribution.'''
'''Has been proven to be an effective technique for regularization and preventing co-adaptation of neurons https://arxiv.org/abs/1207.0580'''


#<---- Long Short Term Memory ---->
'''A form of Neural Network that is used to solve the exploding/vanashing gradient problem within recurent nueral networks(RNN)'''
'''They utilize two different channels for long term memory and short term memory so that long term patterns can be recognized'''
'''Is super slow and does well with medium length text but anything past 1000 words or so it does not do the best'''

#<---- Transformer ---->
'''Similar to LSTM but it is much faster since it can run in parallel'''
'''Typically is used for text translation. It has multiple layers, that being an attention head and feed foward.'''
'''Linear layer of the transformer is utilized to help map a probabilites of vector output to a possible output'''