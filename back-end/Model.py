#import pandas
import pandas as pd
import os
col_names = ['profile pic', 'nums/length username', 'fullname words', 'nums/length fullname', 'name==username', 'description length', 'external URL', 'private', '#posts', '#followers', '#follows', 'fake']

directoryName = os.path.dirname(__file__)
TrainFile = 'back-end//CSVFiles//train.csv'
TestFile = 'back-end//CSVFiles//test.csv'

# load train dataset
train = pd.read_csv(TrainFile, header=0, names=col_names)

# load test dataset
test = pd.read_csv(TestFile, header=0, names=None)

#split train dataset in features and target variable
feature_columns = ['profile pic', 'nums/length username', 'fullname words', 'nums/length fullname', 'name==username', 'description length', 'external URL', 'private', '#posts', '#followers', '#follows']
X_train = train[feature_columns] # Features
y_train = train.fake # Target variable

# import the class
from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
logreg = LogisticRegression(random_state=16, max_iter=1000)

# fit the model with data
logreg.fit(X_train, y_train)

# retrieve a specific row using iloc
sample_row = test.iloc[0]
sample_df = pd.DataFrame([sample_row])
sample_pred = logreg.predict(sample_df[feature_columns])
sample_proba = logreg.predict_proba(sample_df[feature_columns])

# display the prediction and confidence
if sample_pred[0] == 1:
    result = "Bot"
else:
    result = "Not a bot"

print('Prediction:', result)
print('Probability of being a bot:','{:.1%}'.format(sample_proba[0][1]))