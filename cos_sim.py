import pickle
import numpy as np
# with open("embeddings.pkl", "wb") as fOut:
#         pickle.dump({'sentences': df['opinions'], 'embeddings': embeddings},fOut)
emb = pickle.load(open('embeddings.pkl', 'rb'))
sentences, embeddings = emb['sentences'], emb['embeddings']
print(embeddings.shape)
print(embeddings[:10])
print('computing cos sim')
cos_sim = np.dot(embeddings, embeddings.T)
print('computing norm')
norm = np.linalg.norm(embeddings, axis = 1).reshape(-1,1)
print('computing final cos_sim')
cos_sim = cos_sim/norm/ norm.T
#cos_sim.shape
print('saving')
most_similar = np.argmax(np.triu(cos_sim, 1), axis = 1)
with open("embeddings_cos_sim.pkl", "wb") as fOut:
        pickle.dump({'sentences': sentences, 'cos_sim': cos_sim},fOut)