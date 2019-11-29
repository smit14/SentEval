import h5py
import numpy as np
import json
import time
from apply import apply_logician

def decode_text(itow, ans):
    ans_list = []
    for i in range(ans.shape[0]):
        temp_ans = []
        for j in range(ans.shape[1]):
            if ans[i][j] == 8963 or ans[i][j] == 0:
                break
            else:
                temp_ans.append(itow[str(ans[i][j])])

        ans_list.append(temp_ans)
    return ans_list

input_json = '../data/visdial_params.json'
f = json.load(open(input_json, 'r'))
itow = f['itow']

file_path = '../data/visdial_data.h5'
f = h5py.File(file_path,'r')


print(f.keys())
print(f['opt_train'].shape)
print(f['ans_index_train'].shape)
n = 10 # datasize
opt_list_train = f['opt_list_train'][:]                    # total_ans x 8
opt_train = f['opt_train'][:n,:,:]                  # datasize x 10 x 100
ans_index_train = f['ans_index_train'][:n,:]      # datasize x 10


# datasize x 10 x 100 x 3

gt_list = []
opt_list = []


ans_index_train_flattened = ans_index_train.reshape(-1)
ans_index_train_flatten_relative = np.zeros(ans_index_train_flattened.shape[0], dtype=np.int32)
for i in range(n*10):
    ans_index_train_flatten_relative[i] = opt_train.reshape(-1)[100*i + ans_index_train_flattened[i]]


gt_ans = opt_list_train[ans_index_train_flatten_relative,:]
opt_ans = opt_list_train[opt_train.reshape(-1),:]

gt_list = decode_text(itow, gt_ans)
gt2_list = [gt for gt in gt_list for j in range(100)]
opt_list = decode_text(itow, opt_ans)

batch_size = 2000
n = len(opt_list)

#open file - visdial_data_probs.h5
file_name = '../data/visdial_data_prob.h5'
f = h5py.File(file_name,'r+')
probs_data = f['opt_train']

ii = 0
t = time.time()
for i in range(0, n, batch_size):
    t = time.time()
    start = i
    end = min(n,start+batch_size)
    gt = gt2_list[start:end]
    opt = opt_list[start:end]

    probs = apply_logician(gt,opt, is_list=True)
    probs = probs.cpu().detach().numpy()
    log_probs = 10000*((probs[:,0]*10000).astype(int)) + (probs[:,1]*10000).astype(int)
    log_probs = log_probs.reshape(-1,10,100)
    probs_data[ii:int(ii+batch_size/1000),:,:] = log_probs
    ii = ii+int(batch_size/1000)
    print((time.time()-t)*(1000/batch_size))
    t= time.time()
