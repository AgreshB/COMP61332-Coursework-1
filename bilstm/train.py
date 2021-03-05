import torch
from concat_dataset import ConcatDataset
from torch.utils.data.dataloader import DataLoader

class Train:
    def __init__(self, config, preProcessedData, collate_fn):
        super().__init__()

        self.batch_size = config["batch_size"]
        self.epochs = config["epoch"]
        self.early_stop = config["early_stop"]

        # define train test split
        train_qty = int(0.9 * len(preProcessedData.labels))
        validation_qty = len(preProcessedData.labels) - train_qty

        torch.manual_seed(0)

        labelled_data = [[s, l] for s,l in zip(preProcessedData.sentence_representation, preProcessedData.label_representation)]
        train_data, validation_data = torch.utils.data.random_split(labelled_data, [train_qty, validation_qty])

        self.x_train, self.y_train = [], []

        for train_dp in train_data:
            self.x_train.append(train_dp[0])
            self.y_train.append(train_dp[1])

        self.x_validation, self.y_validation = [], []

        for val_dp in validation_data:
            self.x_validation.append(val_dp[0])
            self.y_validation.append(val_dp[1])

        # initialise dataloaders
        self.concat_train = ConcatDataset((self.x_train, self.y_train))
        self.dataloader_train = DataLoader(self.concat_train, batch_size=self.batch_size, collate_fn=collate_fn)
        self.concat_validation = ConcatDataset((self.x_validation, self.y_validation))
        self.dataloader_validation = DataLoader(self.concat_validation, batch_size=self.batch_size, collate_fn=collate_fn)

    def doTraining(self, model, model_name, loss_fxn, optimizer, accuracy_fxn):
        model.train()
        early_stop, best_accuracy = 0, 0

        for epoch in range(self.epochs):
            batch_count = 1

            for data, label, length in self.dataloader_train:
                optimizer.zero_grad()
                y_pred = model(data, length)
                loss = loss_fxn(y_pred, torch.tensor(label))
                batch_count += 1
                loss.backward()
                optimizer.step()

                accuracy, _, _ = accuracy_fxn(model, self.dataloader_validation)

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    early_stop = 0
                    torch.save(model, f"data/{model_name}.model")
                    print(f"epoch: {epoch + 1}\tbatch: {batch_count}\taccuracy: {best_accuracy}")
                else:
                    early_stop += 1
                if early_stop >= self.early_stop:
                    print("early stop condition met")
                    break
        final_model = torch.load(f"data/{model_name}.model")
        accuracy, y_actual, y_pred = accuracy_fxn(final_model, self.dataloader_validation)
        return accuracy, y_pred