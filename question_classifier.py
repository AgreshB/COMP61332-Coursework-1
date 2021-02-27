
import sys, getopt
import configparser

def BagOfWords(config, train, test):
    if(train):
        print("BoW: Training begun...")
        #TODO: BoW
        print("BoW: Training complete!")
    if(test):
        print("BoW: Test results:")
        #TODO: Print results

def BiLSTM(config, train, test):
    if(train):
        print("BiLSTM: Training begun...")
        #TODO: BiLSTM
        print("BiLSTM: Training complete!")
    if(test):
        print("BiLSTM: Test results:")
        #TODO: Print results


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
    model = config.get('MODEL', 'Model')
    if model == 'bow':
        print(model)
        BagOfWords(config, train, test)
    elif model == 'bilstm':
        BiLSTM(config, train, test)
    else:
        print('Error choosing model! Model is currently: ',model)

if __name__ == "__main__":
   main(sys.argv[1:])
