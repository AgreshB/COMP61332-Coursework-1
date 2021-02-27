class Classifier(object):
    
    def __init__(self, config):
        self.config = config
        self.train_filename = config.get('MODEL', 'train_filename')
        self.test_filename = config.get('MODEL', 'test_filename')