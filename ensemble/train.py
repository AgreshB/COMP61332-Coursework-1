import torch
from concat_dataset import ConcatDataset
from torch.utils.data.dataloader import DataLoader

class Train:
    def __init__(self, config, dataset, collate_fn, model_no):
        super().__init__()

        self.batch_size = config["batch_size"]
        self.epochs = config["epoch"]
        self.early_stop = config["early_stop"]

        # define train test split
        train_qty = int(0.9 * len(dataset))
        validation_qty = len(dataset) - train_qty

        torch.manual_seed(model_no)

        train_data, validation_data = torch.utils.data.random_split(dataset, [train_qty, validation_qty])

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

    def doTraining(self, model, model_name, loss_fxn, optimizer, accuracy_fxn, save_file_name=None):
        model.train()
        early_stop, best_accuracy = 0, 0

        if save_file_name == None:
            model_save_file_name = f"data/{model_name}.model"
        else:
            model_save_file_name = save_file_name

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
                    torch.save(model, model_save_file_name)
                    print(f"epoch: {epoch + 1}\tbatch: {batch_count}\taccuracy: {best_accuracy}")
                else:
                    early_stop += 1
                if early_stop >= self.early_stop:
                    print("early stop condition met")
                    break
        final_model = torch.load(model_save_file_name)
        accuracy, y_actual, y_pred = accuracy_fxn(final_model, self.dataloader_validation)
        return accuracy, y_pred