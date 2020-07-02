USAGE OF THE QUESTION RANKER
*************************************
AM Questions and Models of Discourse
author: Luise Schricker

(Obtained from https://github.com/QUD-comp/ranking-potential-questions on 8 April 2020)

Dependencies and package-versions are 
noted in the respective Python-files.
*************************************

General notes: 
In order to use the code which is dependent on a word vector model, get a model of word vectors (e.g. at https://code.google.com/archive/p/word2vec/) and place it in the directory ‚word2vec‘. 
The interview excerpt used in data/snowden_qud_annotations was taken from https://www.ndr.de/nachrichten/netzwelt/Snowden-Interview-Transcript,snowden277.html. 

extract_data.py:
————————————————————————————
This file contains methods for extracting the training and development data. Usage examples of the individual methods are indicated at the bottom of the file. 

Usage:
python3 extract_data.py


extract_features.py:
————————————————————————————
This file contains methods for extracting features given an assertion-question pair. Usage examples of the individual methods are indicated at the bottom of the file. This file does not have to be executed individually for ranking question.


ranking.py:
————————————————————————————
This file contains the QuestionRanker class, which can be used to rank potential questions. In order to retrain the classifier that is used in the machine learning mode (‚ml’) of the ranker, the parameter ‚train‘ has to be set to True and the path to the training data has to be given as ‚train_path‘ when initializing the QuestionRanker. In case the training data or the Feature Extractor (extract_features.py) is changed, the file swda_train_features.json has to be deleted before training. Then the features of the training set are extracted before training starts. WARNING: this takes quite some time (> 1h on a Macbook Pro, 2015).

ATTENTION: Retraining the classifier overwrites the existing model. To prevent this, rename the file of the existing model (model/model.pkl).


Usage examples of the QuestionRanker’s individual methods are indicated at the bottom of the file.
