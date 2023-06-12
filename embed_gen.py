#python -m nltk.downloader all
# import pandas as pd
# import numpy as np
# # import string
# # import nltk
# # import re
# # from tqdm import tqdm
# # nltk.download('stopwords')
# # from nltk.corpus import stopwords
# # stop_words = stopwords.words('english')
# from sentence_transformers import SentenceTransformer,util
# # from pandarallel import pandarallel
# # pandarallel.initialize()
# # from fast_sentence_transformers import FastSentenceTransformer as SentenceTransformer
# import pickle

# def preprocess(op):
#     txt = op.split("text\':")
#     if len(txt)>1:
#       txt = txt[1]
#     else:
#       return ""
#     txt = re.sub('\\n',' ',txt)#txt.split('\\n')
#     #txt = " ".join(txt)    
#     txt = "".join(c for c in txt if c not in string.punctuation)
#     words = (txt.split())
#     words = [w for w in words if w.lower() not in stop_words and len(w) > 3][:512]
#     final_txt = " ".join(words)    
#     return final_txt


# print("preprocessing")
# # df['opinion_text'] = df['opinions'].parallel_apply(lambda x: preprocess(x))
# print("preprocessing done")

# # encoder = SentenceTransformer('all-MiniLM-L6-v2', device = 'gpu', quantize = True) 
# embeddings = encoder.encode(df['opinion_text'])

if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    from sentence_transformers import SentenceTransformer,util
    import pickle

    print('reading file')
    # df = pd.read_excel('full-unlabelled.xlsx')
    df = pd.read_excel('preprocessed_full.xlsx')
    df = df.dropna()
    print('reading file done')
    model = SentenceTransformer('all-MiniLM-L6-v2')
    #Start the multi-process pool on all available CUDA devices
    print('loading model')
    embeddings=[]
    pool = model.start_multi_process_pool()
    print('computing embeddings...')
        #Compute the embeddings using the multi-process pool
    embeddings = model.encode_multi_process(df['opinion_text'], pool)
    print("Embeddings computed. Shape:", embeddings.shape)

        #Optional: Stop the proccesses in the pool
    model.stop_multi_process_pool(pool)
# if(embeddings==[]):
#     print('no embeddings')
#     return
    with open("embeddings.pkl", "wb") as fOut:
        pickle.dump({'sentences': df['opinions'], 'embeddings': embeddings},fOut)

    cos_sim = np.dot(embeddings, embeddings.T)
    norm = np.linalg.norm(embeddings, axis = 1).reshape(-1,1)
    cos_sim = cos_sim/norm/ norm.T
    #cos_sim.shape
    most_similar = np.argmax(np.triu(cos_sim, 1), axis = 1)
