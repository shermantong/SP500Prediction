# SP500Prediction

## To launch the app server, run the following command:
python3 SP500_app.py. it will show 
* Running on local URL: http://127.0.0.1:7680

## To train the model, run the following command:
python3 model_training.py <train_datafile_path> <test_datafile_path>

## To test the model, run the following command:
python3 model_predicting.py <datafile_path>

## To initialize the database, run the following command:
python3 initialize_database.py <datafile_path>

## To clean the database, run the following command:
python3 clean_database.py --db <database_path> --ops <operation1> <operation2> ...

