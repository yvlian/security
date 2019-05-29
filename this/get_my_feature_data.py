import pandas as pd
print('---')
d1 = pd.read_csv('../data/security_train.csv')
d1['file_id'] = ['train'+str(fid) for fid in d1['file_id']]
y = d1[['file_id','label']].groupby('file_id').mean()
d1.pop('label')

d2 = pd.read_csv('../data/security_test.csv')
d2['file_id'] = ['test'+str(fid) for fid in d2['file_id']]

d = pd.concat([d1,d2],axis=0)
d.reset_index(inplace=True)

api_dictionary = list(set(d['api']))

tids_per_file = dict()
data = pd.DataFrame(columns=api_dictionary)
file_ids = list(set(d['file_id']))
file_ids.sort(key = list(d['file_id']).index)

print('get my data feature,including file_id,APIs_count,api_length,total_api_length,n_threads,label')
for fid in file_ids:
    mask1 = (d['file_id'] == fid)
    tid_mask1 = d['tid'][mask1]
    tids = list(set(tid_mask1))
    new_row = {'file_id':fid,'n_threads':len(tids)}
    api_lengths = []
    for i in range(len(tids)):
        tid = tids[i]
        mask2 = (d['tid'] == tid)
        api_lengths.append(len(tid_mask1[mask2]))
        apis = d['api'][mask1][mask2]
        for api in apis:
            if api not in new_row.keys():
                new_row[api] = 1
                continue
            new_row[api] += 1
    api_lengths.sort(reverse=True)
    for i in range(len(api_lengths)):
        new_row['api_length' + str(i)] = api_lengths[i]
    new_row['total_api_length'] = sum(api_lengths)
    data = data.append(new_row, ignore_index=True)

print('train.csv')
data = data.fillna(0)
data1 = data[:len(y)]
data1['label'] = list(y['label'])
data1['file_id'] = data1['file_id'].apply(lambda x:x.replace('train','')).tolist()
data1 = data1.set_index('file_id')
data1.to_csv('../data/train.csv')

print('test.csv')
data2 = data[len(y):]
data2['file_id'] = data2['file_id'].apply(lambda x:x.replace('test','')).tolist()
data2 = data2.set_index('file_id')
data2.to_csv('../data/test.csv')

