
import sys, getopt
import configparser
from bow.bow import BagOfWords
from bow.cbow import ContBagOfWords
from bilstm.bilstm import BiLSTM


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
    
    config = configparser.ConfigParser()
    config.read(configfile)
    modelString = config.get('MODEL', 'Model')

    def Convert(tup, di): 
        for a, b in tup:
            if(RepresentsInt(b)):
                di.setdefault(a, int(b))
            if(RepresentsFloat(b)):
                di.setdefault(a, float(b))
            elif(RepresentsBool(b)):
                di.setdefault(a, bool(b))
            else:
                di.setdefault(a, b)
        return di
    def RepresentsBool(s):
        if(s == "True" or s == "False"):
            return True
        return False
    def RepresentsInt(s):
        try: 
            int(s)
            return True
        except ValueError:
            return False
    def RepresentsFloat(s):
        try: 
            float(s)
            return True
        except ValueError:
            return False
    dictionary = {} 
    config = Convert(config.items('MODEL'), dictionary)

    if modelString == 'bow':
        model = BagOfWords(config)
    elif modelString == 'cbow':
        model = ContBagOfWords(config)
    elif modelString == 'bilstm':
        model = BiLSTM(config)
    else:
        print('Error choosing model! Model is currently: ',modelString)
    
    if(train):
        model.train()
    if(test):
        model.test()
if __name__ == "__main__":
   main(sys.argv[1:])
