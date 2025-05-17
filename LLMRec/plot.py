import pickle
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib
from sklearn import manifold
from sklearn.preprocessing import normalize
import seaborn as sns 
from math import pi 
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42 

user_cnt = 2000
item_cnt = 500
popu = open('/home/share/yangxuanhui/dataset/movielens/user_item_dict.npy', 'rb')
pop_u = np.load(popu, allow_pickle=True).item()
popi = open('/home/share/yangxuanhui/embedding/item_user_dict.npy', 'rb')
pop_i = np.load(popi, allow_pickle=True).item()
embeds = ['/home/share/yangxuanhui/embedding/mmgcn_user.npy', '/home/share/yangxuanhui/embedding/mgcn_user.npy', '/home/share/yangxuanhui/embedding/lgcn_user.npy']
i_embeds = ['/home/share/yangxuanhui/embedding/mmgcn_item.npy', '/home/share/yangxuanhui/embedding/mgcn_item.npy', '/home/share/yangxuanhui/embedding/lgcn_item.npy']
models = ['modal', 'modal_purified', 'pre-processed']

data = {}

# user_indices = list(user.keys())
# udx = np.random.choice(user_indices, 2000)

popularity = np.array(sorted(len(value) for value in pop_u.values()))
hot = popularity[int(len(popularity) * 0.8)]
medium = popularity[int(len(popularity) * 0.5)]
low = popularity[int(len(popularity) * 0.2)]
hot_users = []
long_tail_users = []
while len(hot_users) < user_cnt:
    udx = np.random.choice(list(pop_u.keys()))
    if len(pop_u[udx]) >= medium and udx not in hot_users:
        hot_users.append(udx)
while len(long_tail_users) < user_cnt:
    udx = np.random.choice(list(pop_u.keys()))
    if len(pop_u[udx]) <= medium:
        long_tail_users.append(udx)

# popularity = np.array(sorted(pop_i.values()))
popularity = np.array(sorted(len(value) for value in pop_i.values()))
hot = popularity[int(len(popularity) * 0.8)]
medium = popularity[int(len(popularity) * 0.5)]
low = popularity[int(len(popularity) * 0.2)]
hot_items = []
long_tail_items = []
i=0
while len(hot_items)<item_cnt:
    udx = np.random.choice(list(pop_i.keys()))
    if len(pop_i[udx]) >= medium and udx not in hot_items:
        hot_items.append(udx)
while len(long_tail_items)<item_cnt:
    udx = np.random.choice(list(pop_i.keys()))
    if len(pop_i[udx]) <= medium:
        long_tail_items.append(udx)

for uemb,iemb,name in zip(embeds, i_embeds, models):
    ue = open(uemb, 'rb')
    ie = open(iemb, 'rb')
    user_emb = np.load(ue, allow_pickle=True)
    item_emb = np.load(ie, allow_pickle=True)
    selected_user_emb = user_emb[hot_users]
    selected_cold_user_emb = user_emb[long_tail_users]
    selected_item_emb = item_emb[hot_items]
    selected_tail_item_emb = item_emb[long_tail_items]
    selected_emb = np.concatenate([selected_user_emb, selected_cold_user_emb, selected_item_emb, selected_tail_item_emb], axis=0)
    print(selected_emb.shape)
    print(selected_user_emb.shape)
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    emb_2d = tsne.fit_transform(selected_emb)
    data[name] = emb_2d
# for emb,name in zip(embeds,models):
#     ue = open(emb, 'rb')

#     user_emb = np.load(ue, allow_pickle=True)
#     selected_user_emb = user_emb[udx]  # np.concatenate([item_emb[idx],user_emb[udx]],axis=0)
#     tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
#     user_emb_2d = tsne.fit_transform(selected_user_emb)
#     data[name]=user_emb_2d
f,axes = plt.subplots(2,len(embeds), figsize=(15, 4),gridspec_kw={'height_ratios':[3,1]})
for i,name in enumerate(models):

    x_hot_user = [d[0] for d in data[name][:user_cnt]]
    y_hot_user = [d[1] for d in data[name][:user_cnt]]
    x_tail_user = [d[0] for d in data[name][user_cnt:user_cnt * 2]]
    y_tail_user = [d[1] for d in data[name][user_cnt:user_cnt * 2]]

    x_hot_item = [d[0] for d in data[name][2 * user_cnt:2 * user_cnt + item_cnt]]
    y_hot_item = [d[1] for d in data[name][2 * user_cnt:2 * user_cnt + item_cnt]]
    x_tail_item = [d[0] for d in data[name][2 * user_cnt + item_cnt:]]
    y_tail_item = [d[1] for d in data[name][2 * user_cnt + item_cnt:]]

    sns.scatterplot(x_hot_user, y_hot_user, ax=axes[0][i], s=2, edgecolor='#800000', color='#800000', label='Hot Users', alpha=0.6)
    sns.scatterplot(x_tail_user, y_tail_user, ax=axes[0][i], s=2, edgecolor='#0C9BE9', color='#0C9BE9', label='Cold Users', alpha=0.6)
    sns.scatterplot(x_hot_item, y_hot_item, ax=axes[0][i], s=2, edgecolor='#FF7F00', color='#FF7F00', label='Hot Items',alpha=0.6)
    sns.scatterplot(x_tail_item, y_tail_item, ax=axes[0][i], s=2, edgecolor='#B2DF8A', color='#B2DF8A', label='Cold Items',alpha=0.6)
    axes[0][i].set_title(name, fontsize=9, fontweight="bold")

    data[name] = normalize(data[name], axis=1, norm='l2')
    x_hot_user = [d[0] for d in data[name][:user_cnt]]
    y_hot_user = [d[1] for d in data[name][:user_cnt]]
    x_tail_user = [d[0] for d in data[name][user_cnt:2*user_cnt]]
    y_tail_user = [d[1] for d in data[name][user_cnt:2*user_cnt]]
    x_hot_item = [d[0] for d in data[name][2*user_cnt:2*user_cnt+item_cnt]]
    y_hot_item = [d[1] for d in data[name][2*user_cnt:2*user_cnt+item_cnt]]
    x_tail_item = [d[0] for d in data[name][2*user_cnt+item_cnt:]]
    y_tail_item = [d[1] for d in data[name][2*user_cnt+item_cnt:]]
    angles = np.arctan2(y_hot_user, x_hot_user)
    sns.kdeplot(data=angles, bw=0.15, shade=True, legend=True, ax=axes[1][i], color='#800000')
    angles = np.arctan2(y_tail_user, x_tail_user)
    sns.kdeplot(data=angles, bw=0.15, shade=True, legend=True, ax=axes[1][i], color='#0C9BE9')
    angles = np.arctan2(y_tail_item, x_tail_item)
    sns.kdeplot(data=angles, bw=0.15, shade=True, legend=True, ax=axes[1][i], color='#B2DF8A')
    angles = np.arctan2(y_hot_item, x_hot_item)
    sns.kdeplot(data=angles, bw=0.15, shade=True, legend=True, ax=axes[1][i], color='#FF7F00')

for ax in axes[0]:
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.patch.set_facecolor('white')
    #ax.collections[0].set_alpha(0)
    ax.set_xlabel('Features', fontsize=9)
    ax.legend(fontsize=7, frameon=True, facecolor='white',ncol=1,markerscale=3).remove()
axes[0][0].set_ylabel('Features', fontsize=9)


for ax in axes[1]:
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.set_xlabel('Angles', fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_xlim(-pi, pi)
axes[1][0].set_ylim(0, 1.5)
axes[1][0].set_ylabel('Density', fontsize=9)
axes[0][0].legend(bbox_to_anchor=(2.5, 1.3),loc='upper center',fontsize=9,ncol=4,frameon=True,markerscale=3)
plt.savefig('/home/share/yangxuanhui/embedding/hot' + '.svg',format='svg')
plt.show()
