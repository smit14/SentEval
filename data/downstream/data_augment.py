import os, sys

with open('good_yes') as f:
    yes_list = f.readlines()

with open('good_no') as f:
    no_list = f.readlines()

with open('good_cntell') as f:
    cnt_list = f.readlines()

with open('labels_testing.py') as f:
    lbls_list = f.readlines()

len_yes = len(yes_list)
len_no = len(no_list)
len_cnt = len(cnt_list)

# final_len = min(len_yes,min(len_no,len_cnt))
final_len = 65

data_dict = {}
data_dict['yes'] = yes_list[:final_len]
data_dict['no'] = no_list[:final_len]
data_dict['cnt'] = cnt_list[:final_len]

s1,s2,labels = [],[],[]
def get_label(str1, str2):
    if (str1=='yes' and str2=='yes') or (str1=='no' and str2=='no') or (str1=='cnt' and str2=='cnt'):
        return 'entailment'
    if (str1=='yes' and str2=='no') or (str2=='yes' and str1=='no'):
        return 'contradiction'
    if (str1=='yes' and str2=='cnt') or (str1=='no' and str2=='cnt') or (str2=='yes' and str1=='cnt') or (str2=='no' and str1=='cnt'):
        return 'neutral'

keys_list = ['yes','no','cnt']
for i in range(3):
    for j in range(3):
        list1 = data_dict[keys_list[i]]
        list2 = data_dict[keys_list[j]]
        label = get_label(keys_list[i], keys_list[j])

        n1 = len(list1)
        n2 = len(list2)
        n = n1*n2

        s1_list = [x for x in list1 for k in range(n2)]
        assert (len(s1_list) == n)
        s2_list = list2*n1
        assert (len(s2_list) == n)

        if label == 'entailment':
            labels_list = [lbls_list[0]]*n
        elif label == 'neutral':
            labels_list = [lbls_list[1]]*n
        else:
            labels_list = [lbls_list[2]]*n

        s1 += s1_list
        s2 += s2_list
        labels += labels_list


first_n = 65000

with open('./SNLI/s1.train') as f:
    init_s1 = f.readlines()

with open('./SNLI/s2.train') as f:
    init_s2 = f.readlines()

with open('./SNLI/labels.train') as f:
    init_labels = f.readlines()

init_s1 = init_s1[:first_n]
init_s2 = init_s2[:first_n]
init_labels = init_labels[:first_n]

init_s1 += s1
init_s2 += s2
init_labels += labels


w = open("./SNLI/s1_augment_ordered.train", 'w')
w.writelines([items for items in init_s1])
w.close()

w = open("./SNLI/s2_augment_ordered.train", 'w')
w.writelines([items for items in init_s2])
w.close()

w = open("./SNLI/labels_augment_ordered.train", 'w')
w.writelines([items for items in init_labels])
w.close()
