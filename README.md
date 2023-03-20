# Text Mining Classification

## LSTM

### python setup
- the following steps ensure everyone runs the versions of the packages being used
- use python3+
- install pip (https://pypi.org/project/pip/)
- run the commands below in the terminal
- run `pip install virtualenv`
- run `virtualenv venv`
- run `source venv/bin/activate`
- run `pip install -r requirements.txt`
- whenever you install a pip package, remember to run `pip freeze > requirements.txt`


### BiLSTM training (for development)
- open bilstm.py
- configure/change the config dictionary as needed
- call `train` fxn of object bilstm_classifier (last line), ie, `bilstm_classifier.train()`
- run `python bilstm.py`

### BiLSTM testing (for development)
- open bilstm.py
- configure/change the config dictionary as needed
- call `test` fxn of object bilstm_classifier (last line), ie, `bilstm_classifier.test()`
- run `python bilstm.py`

### TODO
- load config from config file
- metrics (f1 score, confusion matrix)
