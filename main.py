import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

dataset = np.array(
[[-25, -46], #são paulo
[-22, -43], #rio de janeiro
[-25, -49], #curitiba
[-30, -51], #porto alegre
[-19, -43], #belo horizonte
[-15, -47], #brasilia
[-12, -38], #salvador
[-8, -34], #recife
[-16, -49], #goiania
[-3, -60], #manaus
[-22, -47], #campinas
[-3, -38], #fortaleza
[-21, -47], #ribeirão preto
[-23, -51], #maringa
[-27, -48], #florianópolis
[-21, -43], #juiz de fora
[-1, -48], #belém
[-10, -67], #rio branco
[-8, -63] #porto velho
])


plt.scatter(dataset[:,1], dataset[:,0])
plt.xlim(-75, -30)
plt.ylim(-50, 10)
plt.grid()
plt.savefig('1-cidades.png')

model = KMeans(n_clusters=3, init='k-means++', n_init='auto')
# model = KMeans(n_clusters=3, init='random', n_init='auto')
pred_y = model.fit_predict(dataset)

plt.scatter(dataset[:,1], dataset[:,0], c = pred_y)
plt.xlim(-75, -30)
plt.ylim(-50, 10)
plt.grid()
plt.scatter(model.cluster_centers_[:,1],model.cluster_centers_[:,0], s = 70, c = 'red')
plt.savefig('2-clusters.png')