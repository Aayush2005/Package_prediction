import pandas as pd
import numpy as np

df = pd.read_csv('Travel_Cleaned.csv')

from sklearn.model_selection import train_test_split

X = df.drop(['ProdTaken'],axis=1)
y = df['ProdTaken']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

##Encoding
cat_features = X.select_dtypes(include = "object").columns
num_features = X.select_dtypes(exclude = "object").columns

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

numericTransformer = StandardScaler()
ohTransformer = OneHotEncoder(drop='first')

preprocessor = ColumnTransformer(
    [
        ('OneHotEncoder',ohTransformer,cat_features),
        ('StrandardScaler',numericTransformer,num_features)
    ]
)

##Applying Transformation

X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)


##Training 
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score,confusion_matrix,precision_score,\
recall_score,roc_auc_score

models = {
    "Decision Tree" : DecisionTreeClassifier(),
    "Random Forest" : RandomForestClassifier()
}

for i in range(len(list(models))):
    model = list(models.values())[i]
    model.fit(X_train,y_train) ##Training Model

    #Make Prediction
    y_training_pred = model.predict(X_train)
    y_pred = model.predict(X_test)

    ##Training set performence
    model_train_accuracy = accuracy_score(y_train,y_training_pred)
    model_train_f1 = f1_score(y_train,y_training_pred,average="weighted")
    model_train_precision = precision_score(y_train,y_training_pred)
    model_train_recall = recall_score(y_train,y_training_pred)
    model_train_roc = roc_auc_score(y_train,y_training_pred)

    ##Test Set Performence
    model_test_accuracy = accuracy_score(y_test,y_pred)
    model_test_f1 = f1_score(y_test,y_pred,average="weighted")
    model_test_precision = precision_score(y_test,y_pred)
    model_test_recall = recall_score(y_test,y_pred)
    model_test_roc = roc_auc_score(y_test,y_pred)

    print(list(models.keys())[i])

    print("Training Score")
    print('- Accuracy {:.4f}'.format(model_train_accuracy))
    print('- F1 Score {:.4f}'.format(model_train_f1))
    print('- Precision {:.4f}'.format(model_train_precision))
    print('- Recall {:.4f}'.format(model_train_recall))
    print('- Roc Auc {:.4f}'.format(model_train_roc))

    print('_'*35)

    print("Test Score")
    print('- Accuracy {:.4f}'.format(model_test_accuracy))
    print('- F1 Score {:.4f}'.format(model_test_f1))
    print('- Precision {:.4f}'.format(model_test_precision))
    print('- Recall {:.4f}'.format(model_test_recall))
    print('- Roc Auc {:.4f}'.format(model_test_roc))

    print('='*35)
    print('\n')

##Hyperparameter Tuning
rf_parms = {
    "max_depth" : [5,8,15,None,10],
    'max_features' : [5,7,"auto",8],
    'min_samples_split' : [2,8,15,20],
    "n_estimators" : [100,200,500,1000]
}

# ##Model List for Hyperparameter Tuning
# randomCvModels = [('RF',RandomForestClassifier(),rf_parms)]

# from sklearn.model_selection import RandomizedSearchCV

# model_param = {}
# for name,model,params in randomCvModels:
#     random = RandomizedSearchCV(estimator=model,param_distributions=params,n_iter=100,cv=3,verbose=2,n_jobs=-1)
#     random.fit(X_train,y_train)
#     model_param[name] = random.best_params_

# for model_name in model_param:
#     print(f"--------------------------Best params for {model_name}------------------------")
#     print(model_param[model_name])


models_tuned = {
    "Random Forest" : RandomForestClassifier(n_estimators=1000,min_samples_split=2,max_features=8,max_depth=None)
}

for i in range(len(list(models_tuned))):
    model = list(models_tuned.values())[i]
    model.fit(X_train,y_train) ##Training Model

    #Make Prediction
    y_training_pred = model.predict(X_train)
    y_pred = model.predict(X_test)

    ##Training set performence
    model_train_accuracy = accuracy_score(y_train,y_training_pred)
    model_train_f1 = f1_score(y_train,y_training_pred,average="weighted")
    model_train_precision = precision_score(y_train,y_training_pred)
    model_train_recall = recall_score(y_train,y_training_pred)
    model_train_roc = roc_auc_score(y_train,y_training_pred)

    ##Test Set Performence
    model_test_accuracy = accuracy_score(y_test,y_pred)
    model_test_f1 = f1_score(y_test,y_pred,average="weighted")
    model_test_precision = precision_score(y_test,y_pred)
    model_test_recall = recall_score(y_test,y_pred)
    model_test_roc = roc_auc_score(y_test,y_pred)

    print(list(models_tuned.keys())[i])

    print("Training Score after Hyperparameter tuning")
    print('- Accuracy {:.4f}'.format(model_train_accuracy))
    print('- F1 Score {:.4f}'.format(model_train_f1))
    print('- Precision {:.4f}'.format(model_train_precision))
    print('- Recall {:.4f}'.format(model_train_recall))
    print('- Roc Auc {:.4f}'.format(model_train_roc))

    print('_'*35)

    print("Test Score after Hyperparameter Tuning")
    print('- Accuracy {:.4f}'.format(model_test_accuracy))
    print('- F1 Score {:.4f}'.format(model_test_f1))
    print('- Precision {:.4f}'.format(model_test_precision))
    print('- Recall {:.4f}'.format(model_test_recall))
    print('- Roc Auc {:.4f}'.format(model_test_roc))

    print('='*35)
    print('\n')