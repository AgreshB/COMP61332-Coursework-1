from classifier import Classifier

class BiLSTM(Classifier):
    def __init__(self, config):
        Classifier.__init__(self,config)
        self.config = config

    def train(self):
        print("BiLSTM: Training begun...")
        #TODO: BoW
        print("BiLSTM: Training complete!")

    def test(self):
        print("BiLSTM: Test results:")
