import numpy as np
import pandas as pd
import ampligraph
from ampligraph.latent_features import TransE
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Knowledge Graph Construction.
def graph_construction(start, stop, step):
    kg = pd.DataFrame()
    kg['h'] = [i for i in np.arange(start, stop, step)]
    kg['r'] = ['isLessThan'] * int((stop-start) / step)
    kg['t'] = [i for i in np.arange(step, stop+1, step)]
    kg["h"] = kg.h.values.astype(str)
    kg["t"] = kg.t.values.astype(str)
    return kg

start = 0
stop = 10000
step = 50
kg_df = graph_construction(start, stop, step)
kg_df.head()

kg_array = kg_df.to_numpy().astype(str)
model = TransE(epochs=200, k=100, verbose=True)
model.fit(kg_array)

