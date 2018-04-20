import xgboost as xgb
import pandas as pd
import numpy as np

#sum(1 for line in open('train.csv'))

train = pd.read_csv("train.csv", dtype={'ip':int, 'app':int,
        'device':int,'os':int,'channel':int, 'is_attributed':int}, nrows=50000000, low_memory = True)
target = train['is_attributed']
train = train.drop(['is_attributed', 'attributed_time', 'click_time'],axis=1)

xgtrain = xgb.DMatrix(train.values, target.values)

param = {'max_depth': 5, 'eta': 1, 'silent': 0, 'objective': 'binary:logistic'}
param['nthread'] = 8
param['eval_metric'] = 'auc'

plst = param.items()

num_round = 10
bst = xgb.train(plst, xgtrain, num_round)

bst.save_model('5depth470.model')

test = pd.read_csv("test.csv")
test = test.drop(['click_id', 'click_time'],axis=1)
xgtest = xgb.DMatrix(test.values)

ypred = bst.predict(xgtest)

ypred = np.where(ypred >= 0.5, 1, 0)

#Prepare Kaggle Submission
click_id = test.iloc[:, 0].values
predictions = ypred
submission = "MySubmission1.csv"
d = {'click_id': click_id, 'is_attributed': predictions}
df = pd.DataFrame(data=d)
df.to_csv(submission, index=False)