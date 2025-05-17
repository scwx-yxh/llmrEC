import numpy as np
uid = open('/home/share/yangxuanhui/embedding/mmgcn_item.npy', 'rb')
user_item_dict = np.load(uid, allow_pickle=True)
print(user_item_dict.shape)
item_user_dict = {}
for user, items in user_item_dict:
    for item in items:
        if item not in item_user_dict:
            item_user_dict[item] = []
        item_user_dict[item].append(user)

np.save('/home/share/yangxuanhui/embedding/item_user.npy', item_user_dict) 