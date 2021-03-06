import torch
from classifier import Classifier
from bilstm.train import Train
from bilstm.test import Test
from bilstm.preprocessing import PreProcesseData
from bilstm.bilstm_random import BilstmRandom, BilstmEnsemble
from bilstm.bilstm_pretrain import BilstmPretrain
from bilstm.eval import get_accuracy_bilstm

config = {
    "batch_size": 200,
    "embed_dim": 300,
    "bilstm_hidden_size": 100,
    "hidden_size": 300,
    "lr": 1,
    "momentum": 0,
    "epoch": 30,
    "early_stop": 500,
    "use_pretrained": False,
    "use_ensemble" : False,
    "n_models": 3,
    "path_model": "data/bilstm.model",
    "freeze": False,
    "train_data_file_path": "res/train_5500.label",
    "test_data_file_path": "res/TREC_10.label",
    "pre_train_file_path": "data/glove.small",
}

class BiLSTM(Classifier):
    def __init__(self, config):
        Classifier.__init__(self, config)
        self.config = config

    def collate_fxn(self, input_dataset):
        data, label, length = [],[],[]
        for dataset in input_dataset:
            data.append(dataset[0])
            label.append(dataset[1])
            length.append(len(dataset[0]))
        data = torch.nn.utils.rnn.pad_sequence(data, padding_value=0)
        return data, label, length


    def train(self):
        # preprocess data
        preProcessedData = PreProcesseData(file_path=config["train_data_file_path"], pre_train_file_path=config["pre_train_file_path"], is_train=True)

        # initialise train
        trainer = Train(config=config, preProcessedData=preProcessedData, collate_fn=self.collate_fxn)

        # choose model to be trained
        if self.config['use_pretrained']:
            model = BilstmPretrain(
                embed=torch.FloatTensor(preProcessedData.vocabulary_embed),
                hidden_zie=config['bilstm_hidden_size'],
                forward_hidden_zie=config['hidden_size'],
                forward_output_size=len(preProcessedData.label_index),
                enable_grad=not config['freeze']
            )
        else:
            if self.config['use_ensemble']:
                model = BilstmEnsemble(
                    n_models=config['n_models'],
                    input_size=config['embed_dim'],
                    hidden_zie=config['bilstm_hidden_size'],
                    vocabulary_size=len(preProcessedData.vocabulary) + 1,
                    forward_hidden_zie=config['hidden_size'],
                    forward_output_size=len(preProcessedData.label_index),
                    enable_grad=config['freeze'],
                )
            else:
                model = BilstmRandom(
                    input_size=config['embed_dim'],
                    hidden_zie=config['bilstm_hidden_size'],
                    vocabulary_size=len(preProcessedData.vocabulary) + 1,
                    forward_hidden_zie=config['hidden_size'],
                    forward_output_size=len(preProcessedData.label_index),
                    enable_grad=config['freeze'],
                )
        # choose loss function and optimser
        loss_fxn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"])

        print("BiLSTM: Training begun...")
        accuracy, y_pred = trainer.doTraining(model=model, model_name="bilstm", loss_fxn=loss_fxn, optimizer=optimizer, accuracy_fxn=get_accuracy_bilstm)
        print("BiLSTM: Training complete!")
        print("BiLSTM: Train results:")
        print("Accuracy: ", accuracy)

    def test(self):
        preProcessedData = PreProcesseData(file_path="res/TREC_10.label", pre_train_file_path=config["pre_train_file_path"], is_train=False)
        model = torch.load(config["path_model"])
        tester = Test(preProcessedData=preProcessedData, model=model, model_type="bilstm")
        accuracy, y_pred = tester.doTesting()
        print("BiLSTM: Test results:")
        print("Accuracy: ", accuracy)
