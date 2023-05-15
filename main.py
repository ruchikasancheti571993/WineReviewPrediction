from dataloader import start_preprocessing
from train import start_training,  tokenized_dataset
from evaluate import evaluate

def run(file):
    train,test,val,testing_data=start_preprocessing(file)
    train_dataset,val_dataset,test_dataset= tokenized_dataset(train,test,val)
    trainer =start_training(train_dataset,val_dataset,test_dataset)
    evaluate(trainer,test_dataset,testing_data)

if __name__=="__main__":
    filepath='winedata.csv'
    run(filepath)



