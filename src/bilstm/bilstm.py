import torch
import numpy as np
from classifier import Classifier
from bilstm.train import Train
from bilstm.test import Test
from bilstm.preprocessing import PreProcesseData
from bilstm.bilstm_random import BilstmRandom, BilstmRandomEnsemble
from bilstm.bilstm_pretrain import BilstmPretrain, BilstmPretrainEnsemble
from bilstm.eval import get_confusion_matrix, get_micro_f1, get_macro_f1


class BiLSTM(Classifier):
    def __init__(self, config):
        Classifier.__init__(self, config)
        self.config = config
        self.model = None

    def collate_fxn(self, input_dataset):
        data, label, length = [],[],[]
        for dataset in input_dataset:
            data.append(dataset[0])
            label.append(dataset[1])
            length.append(len(dataset[0]))
        data = torch.nn.utils.rnn.pad_sequence(data, padding_value=0)
        return data, label, length

    def accuracy_fxn(self, model, loader):
        y_pred = list()
        y_actual = list()
        with torch.no_grad():
            for x, y, lengths in loader:
                y_pred.extend(model(x,lengths).argmax(dim=1).numpy().tolist())
                y_actual.extend(y)
        return np.sum(np.array(y_pred) == y_actual) / len(y_actual), y_actual, y_pred

    def output_results(self, accuracy, confusion_matrix, micro_f1, macro_f1, fp):
        print("BiLSTM: Train results:", file=fp)
        print("Accuracy: ", accuracy, file=fp)
        print("Confusion Matrix:\n", confusion_matrix, file=fp)
        print("Micro F1: ", micro_f1, file=fp)
        print("Macro F1: ", macro_f1, file=fp)


    def train(self, processedData=None, trainer_obj=None, save_file_name=None):
        # preprocess data
        if processedData == None:
            preProcessedData = PreProcesseData(file_path=self.config["path_data"], pre_train_file_path=self.config["path_pre_emb"], unk_token=self.config["unk_token"], is_train=True)
        else:
            preProcessedData = processedData

        # initialise train
        if trainer_obj == None:
            trainer = Train(config=self.config, preProcessedData=preProcessedData, collate_fn=self.collate_fxn)
        else:
            trainer = trainer_obj

        # choose model to be trained
        if self.config['use_pretrained']:
            if self.config["use_ensemble"]:
                model = BilstmPretrainEnsemble(
                    n_models=self.config["n_models"],
                    embed=torch.FloatTensor(preProcessedData.vocabulary_embed),
                    hidden_zie=self.config['bilstm_hidden_size'],
                    forward_hidden_zie=self.config['hidden_size'],
                    forward_output_size=len(preProcessedData.label_index),
                    enable_grad=not self.config['freeze']
                )
            else:
                model = BilstmPretrain(
                    embed=torch.FloatTensor(preProcessedData.vocabulary_embed),
                    hidden_zie=self.config['bilstm_hidden_size'],
                    forward_hidden_zie=self.config['hidden_size'],
                    forward_output_size=len(preProcessedData.label_index),
                    enable_grad=not self.config['freeze']
                )
        else:
            if self.config["use_ensemble"]:
                model = BilstmRandomEnsemble(
                    n_models=self.config["n_models"],
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
        fp = open(self.config["output_file"], "w")
        accuracy, y_pred = trainer.doTraining(model=model, model_name="bilstm", loss_fxn=loss_fxn, optimizer=optimizer, accuracy_fxn=self.accuracy_fxn, save_file_name=save_file_name, fp=fp)
        confusion_matrix = get_confusion_matrix(trainer.y_validation, y_pred, len(preProcessedData.label_index))
        micro_f1 = get_micro_f1(confusion_matrix)
        macro_f1,f1 = get_macro_f1(confusion_matrix)
        print("BiLSTM: Training complete!")
        print("BiLSTM: Train results:")
        print("Accuracy: ", accuracy)
        print("Confusion Matrix:\n", confusion_matrix)
        print("Micro F1: ", micro_f1)
        print("Macro F1: ", macro_f1)
        self.output_results(accuracy, confusion_matrix, micro_f1, macro_f1, fp)
        self.model = model
        fp.close()

    def test(self):
        print("BiLSTM: Testing begun...")
        fp = open(self.config["output_file"], "w")
        preProcessedData = PreProcesseData(file_path=self.config["path_test"], pre_train_file_path=self.config["path_pre_emb"], unk_token=self.config["unk_token"], is_train=False)
        model = torch.load(self.config["model_path"])
        tester = Test(preProcessedData=preProcessedData, model=model, model_type="bilstm")
        accuracy, y_pred = tester.doTesting()
        confusion_matrix = get_confusion_matrix(tester.y_actual, y_pred, len(preProcessedData.label_index))
        micro_f1 = get_micro_f1(confusion_matrix)
        macro_f1,f1 = get_macro_f1(confusion_matrix)
        print("BiLSTM: Testing complete!")
        print("BiLSTM: Test results:")
        print("Accuracy: ", accuracy)
        print("Confusion Matrix:\n", confusion_matrix)
        print("Micro F1: ", micro_f1)
        print("Macro F1: ", macro_f1)
        self.output_results(accuracy, confusion_matrix, micro_f1, macro_f1, fp)
        fp.close()