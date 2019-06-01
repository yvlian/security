import pandas as pd
import xgboost as xgb
import numpy as np
d = pd.read_csv('./data/train.csv')
train_id = d.pop('file_id').tolist()
n = len(d)
split = 0.75
m = int(split * n)
y_train = d['label'][:m].tolist()
y_val = d['label'][m:].tolist()
print('y_train',y_train)
print('y_val',y_val)
d.pop('label')
x_train = d[:m]
x_val = d[m:]

# x_test = pd.read_csv('./data/test.csv')
# test_id = x_test.pop('file_id').tolist()

param = {}
param['learning_rate'] = 0.1
param['n_estimators'] = 1000
param['objective'] = 'multi:softprob'
# scale weight of positive examples
param['eta'] = 0.1
param['max_depth'] = 6
param['min_child_weight'] = 1
param['scale_pos_weight'] = 1
param['gamma'] = 0
param['silent'] = 1
param['nthread'] = 4
param['num_class'] = 8
param['subsample'] = 0.8
param['colsample_bytree'] = 0.8
param['seed'] = 27


model = xgb.train(param, xgb.DMatrix(x_train, y_train))
softprob_pred = model.predict(xgb.DMatrix(x_val))
print(softprob_pred)
predictions = softprob_pred.argmax(axis=1)
accuracy = sum([1 for i in range(len(predictions)) if predictions[i] == y_val[i]])/len(predictions)
print(accuracy)

# softprob_pred = model.predict(xgb.DMatrix(x_test))
# data1 = pd.DataFrame(softprob_pred,columns=['prob0', 'prob1', 'prob2', 'prob3', 'prob4', 'prob5' ,'prob6','prob7'])
# data1['file_id'] = test_id
# data1 = data1.set_index('file_id')
# data1.to_csv('./data/security_submit.csv')
