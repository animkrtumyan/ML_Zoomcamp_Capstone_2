**Problem description**

The **dataset** refers to the Titanic shipwreck [datasets](https://www.kaggle.com/c/titanic/data?select=train.csv). 

The **purpose** of the project is to predict which passengers will survived based on given critera. There are 891 passengers' information in the dataset. 
The total number of columns are 12, where 7 are numeric, 5 are categorical features.  


The features (columns) are the following:

PassengerID: An unique index for passeningers' rows.

Survived: States if the passenger survived or not.

Pclass: Ticket class. 1st, 2nd, and 3rd.

Name: Passenger’s name.

Sex: Passenger’s gender (Male or Female).

SibSP: Number of siblings/spouse living aboad.

Parch: Number of parents/children.

Ticket: Passenger Ticket Number.

Fare: The cost of the ticket.

Cabin: Passenger Cabin.

Embarked: Port of boarding.

Age: Passenger’s age.

We can dismiss the 'PassengerId', 'Cabin','Name','Ticket' features, as they will not have big influence on targed variable 'Survived'.

The problem of survival prediction is vital from security perspective. The insights learned from the project can be used in passenger distribution during cruise or designing of the ships.

As _objectives_ of this project were selected:
-identify data feauters,
-conduct EDA for finding out patterns and correlations of the data features,
-use various types of modeling for prediction of survival, that worth to be approved,
-use trained models for further analysis and predictions.

The results of the project can be used for designing models to solving transportation secuirity issues. For example, given a new passenger and using the models, it could be possible to identify the possibility of survival.

**Problem solution**
For solving loan approval problem, there were used predictions based on various models: Logistic regression, Decision Tree Classifier, RandomForest Classifier, Xgboost classifier, etc. Hyperparameter tuning based on GridSearch was implemented for the models. All models perform more than 77 % accuracy score on testing dataset. However, RandomForestClassifier performed slightly better (81 %) than the other models.

**Models with best parameters and accuracies**

-Logistic Regression -'logisticregression__C': 1, accuracy = 0.78
ROC curve performs better of the positive class. 
-KNeighborsClassifier() - {'kneighborsclassifier__n_neighbors': 19}, accuracy =0.77
ROC curve performs worser of the positive class in comparision with the Logistic Regression.
-DecisionTreeClassifier- {'max_depth': 3}, accuracy = 0.78 
ROC and PR-curves perform better than in case of KNeighborsClassifier
-RandomForestClassifier- {'max_features': 5, 'n_estimators': 50}, accuracy = 0.81
ROC curve performs pretty well of the positive class when comparing with the other models.
-XGBClassifier- {'learning_rate': 0.0001, 'n_estimators': 50}, accuracy = 0.79
ROC curve performs better of the positive class.
-GradientBoostingClassifier- {'learning_rate': 1.6667333333333334, 'n_estimators': 50}, accuracy = 0.79
ROC curve performs better of the positive class.
As an outlier detection technique was used DBSCAN.


**Virtual environment**
For activating virtual environment, I used "python -m waitress --listen=*:9696 predict:app" command in the terminal. Parallely, the jupyter notebook was used to send POST requests.

For creating virtual envrinment  pipnev was used. The all versions of necessary libraries were taken via pip freeze command, that outputs all libraries and their versions.
After, I selected those liraries that are used for the Capstone project. This list was savd in the requirements.txt. Later, the requirements.txt was used for creating virtual environment.
 
pipenv install -r requirements.txt
If this will not work, just try the bellow command in the terminal (cmd Windows).
pipenv install numpy pandas requests scikit-learn seaborn  waitress xgboost matplotlib  flask .
To activate this project's virtualenv, run pipenv shell.
Alternatively, run a command inside the virtualenv with pipenv run.

**Docker**
Docker commands were used based on the Windows Docker Desctop. Details are in the file.


