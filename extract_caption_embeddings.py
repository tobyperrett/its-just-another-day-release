from glob import glob
import torch
import torch.nn.functional as F
from plot_similarity import plot_confusion_matrix

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

all_features = []

images = glob("/Users/toby/Code/vis_ai/sample_vids/2876b375-e848-412c-8a6f-0664cbab6a33/lavilla_xl_1s/*1.txt")
images.sort()

print(len(images))


# #Sentences we want to encode. Example:
# sentence = ['This framework generates embeddings for each input sentence']

# #Sentences are encoded by calling model.encode()
# embedding = model.encode(sentence)

for i in images:
    # read first line
    with open(i) as f:
        first_line = f.readline().strip('\n')
    image_features = model.encode(first_line)
    print(image_features.shape)
    all_features.append(torch.tensor(image_features))

feats = torch.stack(all_features)
feats = feats.squeeze()
print(feats.shape)

cos_sim = F.cosine_similarity(feats.unsqueeze(0), feats.unsqueeze(1), dim=2)
print(cos_sim.shape)

plot_confusion_matrix(cos_sim)