#Disaster Response Pipeline

##Overview


A machine learning application pipeline that is capable of curating the text messages and classify them. This application has a flask based web UI to assist emergency worker to classify caller's input into several categories so they can route them to its relevant specialized organization. The training data is provided by Figure Eight.

Dependencies
Interpreter: Python 3.6+
Processing libraries: Numpy, Pandas, Scikit-learn, NLTK, Pickle, Re
DB connect library: SQLalchemy
For Web app: Flask, Ploty

 File Descriptions
data:
disaster_categories.csv: A csv file containing the 36 different message categories
disaster_messages.csv: A csv file containing the disaster messages
process_data.py: ETL pipeline to process messages and categories into a single SQLite database
DisasterResponse.db: SQLite database that contains both messages and categories
model:
train_classifier.py: ML pipeline to build, train, evaluate, and save a classifer
classifier.pkl: Pick file of trained model
app:
run.py: Runs the Flask web app
templates: HTML files for web app

Usage

Clone the repo using:
git clone https://github.com/



Run the following commands in the project's root directory to set up your database and model.

To run ETL pipeline that cleans data and stores in database
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db

ML pipeline that trains classifier and saves the model

python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

Run the following command to run your web app.
python app/run.py

Go to the URL:

http://0.0.0.0:3001/ or http://localhost:3001

Acknowledgements

Figure Eight for providing pre-labeled messages dataset

Author
Naresh Rayala

Copyright and License
Code is released under the MIT License.
