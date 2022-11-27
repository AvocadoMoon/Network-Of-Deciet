import torch
import os
from trainModel import BertClassifier, Dataset
from transformers import BertTokenizer, BertModel



class UseModel():
    def __init__(self, model_path, tokinizer_path):
        self.model = torch.load(model_path)
        # self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.tokenizer = BertTokenizer.from_pretrained(tokinizer_path)
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
    

    def testScentence(self, scentence):
        tokinized_text = self.tokenizer(scentence, padding='max_length', max_length = 512, 
        truncation=True, return_tensors="pt")
        
        self.model.to(self.device)
        print(tokinized_text)

        with torch.no_grad():
            mask = tokinized_text['attention_mask'].to(self.device)
            input_id = tokinized_text['input_ids'].squeeze(1).to(self.device)
            print(self.device)
            output = self.model(mask, input_id)

            print(output)

cwd = os.getcwd()
# um = UseModel(f'{cwd}/stereotype_detection_model.pt', f'{cwd}/bert-base-uncased')
um = UseModel(f'{cwd}/pickledModel.pickle', f'{cwd}/bert-base-uncased')

um.testScentence("Many people live in Ethiopia.The people are very thin and good at distance running.")
um.testScentence("Black people smell and love watermellon")