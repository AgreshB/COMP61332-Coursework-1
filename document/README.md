# COMP61332-Coursework-1

## python setup
- the following steps ensure everyone runs the versions of the packages being used
- use python3+
- install pip (https://pypi.org/project/pip/)
- run the commands below in the terminal
- run `pip install virtualenv`
- run `virtualenv venv`
- run `source venv/bin/activate`
- run `pip install -r requirements.txt`
- whenever you install a pip package, remember to run `pip freeze > requirements.txt`

## running 
- `cd src` to go to the directory containg all the code
- make sure you are in the src directory before training/testing

### training
- `python question_classifier.py --train -c ../data/<config_file_name>.ini` 
- example, `python question_classifier.py --train -c ../data/bilstm.ini` 

### testing
- `python question_classifier.py --test -c ../data/<config_file_name>.ini` 
- example, `python question_classifier.py --test -c ../data/bilstm.ini` 