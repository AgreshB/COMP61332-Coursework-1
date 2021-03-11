import torch
from classifier import Classifier
from bilstm.train import Train
from bilstm.test import Test
from bilstm.preprocessing import PreProcesseData
from bilstm.bilstm_random import BilstmRandom, BilstmEnsemble
from bilstm.bilstm_pretrain import BilstmPretrain
from bilstm.eval import get_accuracy_bilstm


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
        preProcessedData = PreProcesseData(file_path=self.config["path_data"], pre_train_file_path=self.config["path_pre_emb"], unk_token=self.config["unk_token"], is_train=True)

        # initialise train
        trainer = Train(config=self.config, preProcessedData=preProcessedData, collate_fn=self.collate_fxn)

        # choose model to be trained
        if self.config['use_pretrained']:
            model = BilstmPretrain(
                embed=torch.FloatTensor(preProcessedData.vocabulary_embed),
                hidden_zie=self.config['bilstm_hidden_size'],
                forward_hidden_zie=self.config['hidden_size'],
                forward_output_size=len(preProcessedData.label_index),
                enable_grad=not self.config['freeze']
            )
        else:
            if self.config['use_ensemble']:
                model = BilstmEnsemble(
                    n_models=self.config['n_models'],
                    input_size=self.config['word_embedding_dim'],
                    hidden_zie=self.config['bilstm_hidden_size'],
                    vocabulary_size=len(preProcessedData.vocabulary) + 1,
                    forward_hidden_zie=self.config['hidden_size'],
                    forward_output_size=len(preProcessedData.label_index),
                    enable_grad=self.config['freeze'],
                )
            else:
                model = BilstmRandom(
                    input_size=self.config['word_embedding_dim'],
                    hidden_zie=self.config['bilstm_hidden_size'],
                    vocabulary_size=len(preProcessedData.vocabulary) + 1,
                    forward_hidden_zie=self.config['hidden_size'],
                    forward_output_size=len(preProcessedData.label_index),
                    enable_grad=self.config['freeze'],
                )
        # choose loss function and optimser
        loss_fxn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.config["lr"], momentum=self.config["momentum"])

        print("BiLSTM: Training begun...")
        accuracy, y_pred = trainer.doTraining(model=model, model_name="bilstm", loss_fxn=loss_fxn, optimizer=optimizer, accuracy_fxn=get_accuracy_bilstm)
        print("BiLSTM: Training complete!")
        print("BiLSTM: Train results:")
        print("Accuracy: ", accuracy)

    def test(self):
        preProcessedData = PreProcesseData(file_path=self.config["path_test"], pre_train_file_path=self.config["path_pre_emb"], unk_token=self.config["unk_token"], is_train=False)
        model = torch.load(self.config["model_path"])
        tester = Test(preProcessedData=preProcessedData, model=model, model_type="bilstm")
        accuracy, y_pred = tester.doTesting()
        print("BiLSTM: Test results:")
        print("Accuracy: ", accuracy)
