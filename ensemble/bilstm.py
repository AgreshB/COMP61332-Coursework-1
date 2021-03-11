import torch
from bilstm.bilstm import BiLSTM
from bilstm.preprocessing import PreProcesseData
from ensemble.train import Train
import numpy as np

class BistmEnsemble:
    def __init__(self, config):
        self.config = config
        self.n_models = config["n_models"]

    def get_accuracy(self, models, x, y, lengths):
        accs = list()
        y_preds_sum = list()
        for i in range(len(y)):
            y_preds_sum.append(dict())
        with torch.no_grad():
            for model_index in range(len(models)):
                y_preds = models[model_index](x,lengths).argmax(dim=1)
                accs.append(np.sum(y_preds.numpy()==y)/len(y))
                for i in range( len(y_preds.numpy().tolist()) ):
                    if y_preds.numpy().tolist()[i] in y_preds_sum[i]:
                        y_preds_sum[i][y_preds.numpy().tolist()[i]] += 1
                    else:
                        y_preds_sum[i][y_preds.numpy().tolist()[i]] = 1
        y_preds_ens = list()
        for i in range(len(y_preds_sum)):
            sort_list = list(y_preds_sum[i].items())
            sort_list.sort(key=lambda x:x[1],reverse=True)
            y_preds_ens.append(sort_list[0][0])
        accs.append(np.sum(np.array(y_preds_ens)==y)/len(y))
        return accs,y_preds_ens

    def train(self):
        print("BiLSTM Ensemble: Training begun...")
        preProcessedData = PreProcesseData(file_path=self.config["path_data"], pre_train_file_path=self.config["path_pre_emb"], unk_token=self.config["unk_token"], is_train=True)

        # define train test split
        train_qty = int(0.9 * len(preProcessedData.labels))
        validation_qty = len(preProcessedData.labels) - train_qty

        torch.manual_seed(0)

        labelled_data = [[s, l] for s,l in zip(preProcessedData.sentence_representation, preProcessedData.label_representation)]
        train_data, validation_data = torch.utils.data.random_split(labelled_data, [train_qty, validation_qty])

        all_models = []
        for m_no in range(0, self.n_models):
            model = BiLSTM(self.config)
            trainer = Train(config=self.config, dataset=train_data, collate_fn=model.collate_fxn, model_no=m_no)
            model.train(processedData=preProcessedData, trainer_obj=trainer, save_file_name=f"data/bilstm_{m_no}.model")
            all_models.append(model.model)

        x_validation, y_validation = [], []
        for i in range(0, validation_qty):
            x_validation.append(validation_data[i][0])
            y_validation.append(validation_data[i][1])

        l = []
        for s in x_validation:
            l.append(len(s))
        
        x_validation = torch.nn.utils.rnn.pad_sequence(x_validation, padding_value=0)
        accuracy, y_pred = self.get_accuracy(all_models, x_validation, y_validation, l)
        print("BiLSTM Ensemble: Train results:")
        print("Accuracy: ", accuracy[-1])
