
from classifier import Classifier

class BagOfWords(Classifier):
    def __init__(self, config):
        Classifier.__init__(self,config)
        self.config = config

    def train(self):
        print("BoW: Training begun...")
        #TODO: BoW
        print("BoW: Training complete!")

    def test(self):
        print("BoW: Test results:")
