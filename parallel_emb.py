"""
This example starts multiple processes (1 per GPU), which encode
sentences in parallel. This gives a near linear speed-up
when encoding large text collections.
"""

from sentence_transformers import SentenceTransformer, LoggingHandler
import logging
import pandas as pd
import pickle

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

#Important, you need to shield your code with if __name__. Otherwise, CUDA runs into issues when spawning new processes.
if __name__ == '__main__':

    #Create a large list of 100k sentences
    print('reading file')
    # df = pd.read_excel('full-unlabelled.xlsx')
    df = pd.read_excel('preprocessed_full.xlsx')
    df = df.dropna()
    print('reading file done')
    sentences = list(df['opinion_text'].astype("string"))

    #Define the model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    #Start the multi-process pool on all available CUDA devices
    pool = model.start_multi_process_pool()

    #Compute the embeddings using the multi-process pool
    emb = model.encode_multi_process(sentences, pool)
    print("Embeddings computed. Shape:", emb.shape)
    with open("embeddings.pkl", "wb") as fOut:
        pickle.dump({'sentences': df['opinions'], 'embeddings': emb},fOut)
    #Optional: Stop the proccesses in the pool
    model.stop_multi_process_pool(pool)