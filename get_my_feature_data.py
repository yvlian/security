import pandas as pd
print('---')
d1 = pd.read_csv('./data/security_train.csv')
d1['file_id'] = d1['file_id'].apply(lambda x:'train'+str(x))
y = d1[['file_id','label']].groupby('file_id').mean()
d1.pop('label')

d2 = pd.read_csv('./data/security_test.csv')
d2['file_id'] = d2['file_id'].apply(lambda x:'test'+str(x))

d = pd.concat([d1,d2],axis=0)
data = pd.DataFrame()

print('get my data feature,including file_id,APIs_count,threadi_api_length,total_api_length,threads_num,label')
for r in d.groupby('file_id'):
    fid = r[0]
    row = r[1]
    new_row = {'file_id': fid,'total_api_length':row.shape[0]}
    temp = row['tid'].value_counts()
    temp.sort_values(ascending=False)
    new_row['threads_num'] = temp.shape[0]
    for i in range(temp.shape[0]):
        new_row['thread'+str(i)+'_api_length'] = temp.iloc[i]

    temp = row['api'].value_counts()
    for k,v in temp.items():
        new_row[k] = v
    data = data.append(new_row, ignore_index=True)

n = y.shape[0]

print('test.csv')
data = data.fillna(0)

data1 = data[:n]
data1['label'] = y['label'].tolist()
data1['file_id'] = data1['file_id'].apply(lambda x:x.replace('test','')).tolist()
data1 = data1.set_index('file_id')
data1 = data1.astype('int')
data1.to_csv('./data/test.csv')

print('train.csv')
data2 = data[n:]
data2['file_id'] = data2['file_id'].apply(lambda x:x.replace('train','')).tolist()
data2 = data2.set_index('file_id')
data2 = data2.astype('int')
data2.to_csv('./data/train.csv')

