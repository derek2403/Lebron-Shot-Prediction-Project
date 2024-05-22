import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import pickle


df = pd.read_csv("lebron_shot_data.csv")
df['PERIOD'] = df['PERIOD'].astype('object')
categorical=df.select_dtypes(include='object')
numerical=df.select_dtypes(exclude='object')

columns_to_drop = ['PLAYER_NAME', 'TEAM_NAME', 'SEASON', 'GAME_ID', 'GAME_EVENT_ID',
                   'PLAYER_ID', 'TEAM_ID', 'GAME_DATE', 'LOC_X', 'LOC_Y', 'SHOT_ATTEMPTED_FLAG', 'EVENT_TYPE']
df = df.drop(columns=columns_to_drop)
df['TIME_REMAINING'] = df['MINUTES_REMAINING'] + df['SECONDS_REMAINING'] / 60
df['TIME_REMAINING'] = df['TIME_REMAINING'].apply(lambda x: '{:.0f}.{:02.0f}'.format(*divmod(x * 60, 60))) 
df = df.drop(columns=['MINUTES_REMAINING', 'SECONDS_REMAINING'])
label_encoder = LabelEncoder()
df['ACTION_TYPE_ENCODED'] = label_encoder.fit_transform(df['ACTION_TYPE'])
df['HTM_ENCODED'] = label_encoder.fit_transform(df['HTM'])
df['VTM_ENCODED'] = label_encoder.fit_transform(df['VTM'])
one_hot_columns = ['SHOT_TYPE', 'SHOT_ZONE_BASIC', 'SHOT_ZONE_AREA', 'SHOT_ZONE_RANGE', 'PERIOD']
one_hot_encoder = OneHotEncoder(drop='first', sparse_output=False)  # Keep all categories
one_hot_encoded = one_hot_encoder.fit_transform(df[one_hot_columns])
one_hot_encoded_df = pd.DataFrame(one_hot_encoded, columns=one_hot_encoder.get_feature_names_out(one_hot_columns))

df = df.drop(columns=one_hot_columns)

df = pd.concat([df, one_hot_encoded_df], axis=1)

numerical_cols = ['SHOT_DISTANCE']
numerical_transformer = StandardScaler()
df[numerical_cols] = numerical_transformer.fit_transform(df[numerical_cols])
columnsdropped = ['ACTION_TYPE', 'HTM', 'VTM']
df = df.drop(columns=columnsdropped)  # Reassign the dropped DataFrame to df

X = df.drop('SHOT_MADE_FLAG', axis=1)  # Features
y = df['SHOT_MADE_FLAG']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
params = {'n_estimators': 100,
          'max_depth': 5,
          'learning_rate': 0.05,}
tuned_model = GradientBoostingClassifier(**params)
tuned_model.fit(X_train, y_train)
tuned_predictions = tuned_model.predict(X_test)

pickle.dump(tuned_model, open("model.pkl", "wb"))