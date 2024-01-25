import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import  RandomForestClassifier

from sklearn.preprocessing import LabelEncoder
import pickle


from sklearn.model_selection import train_test_split, KFold, cross_val_score

from sklearn.metrics import roc_curve, auc, precision_recall_curve

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline


def train():
    output_file = 'model_rand.bin'
    label_encoder = LabelEncoder()
    scaling = MinMaxScaler()

    data =  pd.read_csv('train.csv')
    df= data.drop(['PassengerId','Cabin','Name','Ticket'], axis = 1)
    df["Sex"] = label_encoder.fit_transform(df["Sex"])
    df["Embarked"] = label_encoder.fit_transform(df["Embarked"])

    df_full_train, df_test  = train_test_split(df, test_size= 0.2, random_state = 1)
    df_train, df_val  = train_test_split(df_full_train, test_size= 0.25, random_state = 1)

    df_train = df_train.reset_index(drop = True)
    df_val = df_train.reset_index(drop = True)
    df_test = df_test.reset_index(drop = True)

    y_train = (df_train['Survived'] == 1).astype(int).values
    y_val = (df_val['Survived'] == 1).astype(int).values
    y_test = (df_test['Survived'] == 1).astype(int).values

    del df_train['Survived']
    del df_val['Survived']
    del df_test['Survived']

    model_rand = RandomForestClassifier(max_features= 5, n_estimators =10 )
    model_rand.fit(df_train, y_train)

    print("The best parameter is : ", model_rand.best_params_ )
    print("The score for the testing set is: ", round(model_rand.score(df_test, y_test),2 ))

    y_predict_rforest = model_rand.predict(df_test)
    y_predict_proba_rforest = model_rand.predict_proba(df_test)

    r_acc = accuracy_score(y_predict_rforest,y_test)

    print('Random forest accuracy: {:.2f}%'.format(r_acc*100))

    with open(output_file_dt, 'wb') as f_out:
        pickle.dump(model_rand, f_out)

    print("Model trained and saved successfully!")


if __name__ == '__main__':
    train()
