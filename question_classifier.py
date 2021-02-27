
import sys, getopt
import configparser
from bilstm import BiLSTM
from bag_of_words import BagOfWords


def main(argv):
    train = False
    test = False
    configfile = ''
    try:
        opts, args = getopt.getopt(argv,"hc:tr",["config=","train","test"])
    except getopt.GetoptError:
        print ('question_classifier.py -c <configfile>')
        sys.exit(2)
    for opt, arg in opts:
        arg = arg.strip() # remove whitespace
        if opt == '-h':
            print ('test.py -c <configfile>')
            sys.exit()
        elif opt in ("-c", "--config"):
            configfile = arg
        elif opt in ("-t", "--test"):
            test = True
        elif opt in ("-r", "--train"):
            train = True
    print(configfile)
    
    config = configparser.ConfigParser()
    config.read(configfile)
    modelString = config.get('MODEL', 'Model')
    
    if modelString == 'bow':
        #BagOfWords(config, train, test)
        model = BagOfWords(config)
    elif modelString == 'bilstm':
        model = BiLSTM(config)
        #BiLSTM(config, train, test)
    else:
        print('Error choosing model! Model is currently: ',model)

    if(train):
        model.train()
    if(test):
        model.test()
if __name__ == "__main__":
   main(sys.argv[1:])
