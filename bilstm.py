from classifier import Classifier
from preprocessing import preprocess_pipeline

class BiLSTM(Classifier):
    def __init__(self, config):
        Classifier.__init__(self,config)
        self.config = config

    def train(self):
        print("BiLSTM: Training begun...")
        # NOTE: replace call to preprocess_pipeline with name in config file
        labels, sentences, vocabulary, vocabulary_embed, sentence_representation, label_index, label_representation = preprocess_pipeline("res/train_5500.label")
        print("BiLSTM: Training complete!")

    def test(self):
        print("BiLSTM: Test results:")
