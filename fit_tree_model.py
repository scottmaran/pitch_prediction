import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import optuna
import pickle
import os

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import cross_val_predict

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

import xgboost as xgb
from xgboost import plot_importance

def objective(trial, train_X, train_y, val_X, val_y, train_weights=None, val_weights=None):
    param = {
             "max_depth": trial.suggest_int('max_depth', 2, 100,step=1),
              "learning_rate": trial.suggest_float('learning_rate', 0.01, 0.5, log=True),
              "n_estimators": trial.suggest_int('n_estimators', 100, 10000,step=100),
              "subsample" : trial.suggest_float('subsample', 0.1, 1, step=0.1),
              "min_child_weight" : trial.suggest_int('min_child_weight', 1, 10, step=1), 
              "colsample_bytree" : trial.suggest_float('subsample', 0.1, 1, step=0.1),
            }
    clf = xgb.XGBClassifier(tree_method="hist", enable_categorical=True, early_stopping_rounds=10,
                            objective='objective=multi:softprob', eval_metric=['merror','mlogloss'], **param)
    
    if train_weights is None:
      clf.fit(train_X, train_y, eval_set=[(val_X, val_y)], verbose=0)
    else:
       clf.fit(train_X, train_y, eval_set=[(val_X, val_y)], 
               sample_weight=train_weights, sample_weight_eval_set=val_weights, verbose=0)
    
    best_merror = clf.evals_result()['validation_0']['merror'][clf.best_iteration]
    return best_merror

def main():
    model_path_name = "weight_adv_no_pitch_id_fulled_tuned"
    DIRECTORY_PATH = f"models/{model_path_name}"
    MODEL_INFO_FILEPATH = f"{DIRECTORY_PATH}/info.txt"

    if not os.path.exists(DIRECTORY_PATH):
        os.makedirs(DIRECTORY_PATH)
    with open(MODEL_INFO_FILEPATH, "w") as text_file:
        print(f"##### MODEL INFO #####", file=text_file)
        print(f"Weighted, no pitcher id, advanced_dataset (no fully properly tuned)", file=text_file)
    
    dataset = pd.read_pickle("data/advanced_dataset_final.pkl")

    train_index_stop = 569484 # this is the first index of a new pitcher
    val_index_stop = 640690 # make sure no overlapping plays

    training_set = dataset.iloc[0:train_index_stop, :]
    validation_set = dataset.iloc[train_index_stop:val_index_stop, :]
    test_set = dataset.iloc[val_index_stop:,]

    train_X = training_set.drop(["uid", "pitch_type", "type_confidence", "pitcher_id"],axis=1)
    train_y = training_set['pitch_type']
    train_weights = training_set['type_confidence']

    val_X = validation_set.drop(["uid", "pitch_type", "type_confidence", "pitcher_id"],axis=1)
    val_y = validation_set['pitch_type']
    val_weights = validation_set['type_confidence']

    test_X = test_set.drop(["uid", "pitch_type", "type_confidence", "pitcher_id"],axis=1)
    test_y = test_set['pitch_type']
    test_weights = test_set['type_confidence']

    mapping = {'FF' : 0,
           'SL': 1, 
           'CU': 2, 
           'SI': 3, 
           'FC': 4, 
           'FT': 5, 
           'KC': 6, 
           'CH': 7, 
           'KN': 8, 
           'FS': 9, 
           'FO': 10, 
           'EP': 11, 
           'SC': 12}
    
    train_y = train_y.map(mapping)
    val_y = val_y.map(mapping)
    test_y = test_y.map(mapping)
    print(f"Beginning Optuna Study...")

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda x : objective(x, train_X, train_y, val_X, val_y, train_weights, val_weights.values.reshape(1,-1)), n_trials=30)
    print(f"Best Params = {study.best_params}")
    with open(MODEL_INFO_FILEPATH, "a") as text_file:
        print(f"Best Params = {study.best_params}", file=text_file)
    
    clf = xgb.XGBClassifier(tree_method="hist", enable_categorical=True, early_stopping_rounds=10,
                                        objective='objective=multi:softprob', eval_metric=['merror','mlogloss'], **study.best_params)
    clf.fit(train_X, train_y, eval_set=[(val_X, val_y)], sample_weight=train_weights, sample_weight_eval_set=val_weights.values.reshape(1,-1))

    pickle.dump(clf, open(f"{DIRECTORY_PATH}/xgb_model.pkl", "wb"))

    best_val_merror = clf.evals_result()['validation_0']['merror'][clf.best_iteration]
    best_val_mlogloss = clf.evals_result()['validation_0']['mlogloss'][clf.best_iteration]

    print(f"Best Val merror = {best_val_merror}")
    print(f"Best Val mlogloss = {best_val_mlogloss}")
    with open(MODEL_INFO_FILEPATH, "a") as text_file:
        print(f"Best Val merror = {best_val_merror}", file=text_file)
        print(f"Best Val mlogloss = {best_val_mlogloss}", file=text_file)

    y_pred = clf.predict(val_X)

    print('\n------------------ Confusion Matrix -----------------\n')
    cmatrix = confusion_matrix(val_y, y_pred)
    print(f"{pd.DataFrame(cmatrix, columns=list(mapping.keys()), index=list(mapping.keys()))}")

    with open(MODEL_INFO_FILEPATH, "a") as text_file:
        print('\n------------------ Confusion Matrix for Full Model -----------------\n', file=text_file)
        print(f"{pd.DataFrame(cmatrix, columns=list(mapping.keys()), index=list(mapping.keys()))}", file=text_file)
    
    print(classification_report(val_y, y_pred, target_names=list(mapping.keys())))
    with open(MODEL_INFO_FILEPATH, "a") as text_file:
        print(classification_report(val_y, y_pred, target_names=list(mapping.keys())), file=text_file)

    ''' Test Stats '''
    y_test_pred = clf.predict(test_X)
    print('\n------------------ TEST DATA Confusion Matrix -----------------\n')
    test_cmatrix = confusion_matrix(test_y, y_test_pred)
    print(f"{pd.DataFrame(test_cmatrix, columns=list(mapping.keys()), index=list(mapping.keys()))}")
    with open(MODEL_INFO_FILEPATH, "a") as text_file:
        print('\n------------------ TEST DATA Confusion Matrix for Full Model -----------------\n', file=text_file)
        print(f"{pd.DataFrame(test_cmatrix, columns=list(mapping.keys()), index=list(mapping.keys()))}", file=text_file)
    print(classification_report(test_y, y_test_pred, target_names=list(mapping.keys())))
    with open(MODEL_INFO_FILEPATH, "a") as text_file:
        print(classification_report(test_y, y_test_pred, target_names=list(mapping.keys())), file=text_file)
    
    plt.style.use('fivethirtyeight')
    plt.rcParams.update({'font.size': 16})

    fig, ax1 = plt.subplots(1,1, figsize=(10,6))
    plot_importance(clf, importance_type='gain', max_num_features=20, ax=ax1)
    plt.savefig(f"{DIRECTORY_PATH}/feature_importance.png")
    plt.show()

if __name__ == "__main__":
    main()