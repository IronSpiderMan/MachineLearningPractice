import glob
import hashlib
import random

import numpy as np
import torch
import chromadb
from tqdm import tqdm
from PIL import Image
from transformers import ViTImageProcessor, ViTModel
from chromadb import Documents, EmbeddingFunction, Embeddings

device = "cuda" if torch.cuda.is_available() else "cpu"


def md5encode(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()


class VitEmbeddingFunction(EmbeddingFunction):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
        self.model = ViTModel.from_pretrained("google/vit-base-patch16-224")

    def __call__(self, images: Documents) -> Embeddings:
        images = [Image.open(image) for image in images]
        inputs = self.processor(images, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            last_hidden_state = outputs.last_hidden_state
        return last_hidden_state[:, 0, :].numpy().tolist()


client = chromadb.PersistentClient(path="image_search")
collection = client.get_or_create_collection(name="images", embedding_function=VitEmbeddingFunction())
files = glob.glob(r"G:\datasets\lbxx\lbxx\*")
# bar = tqdm(total=len(files))
# for file in files:
#     collection.add(
#         documents=[file],
#         metadatas=[{"source": file}],
#         ids=[md5encode(file)]
#     )
#     bar.update(1)
# file = random.choice(files)
file = r"G:\datasets\lbxx\lbxx\10067.jpeg"
similar_images = collection.query(
    query_texts=[file],
    n_results=10,
)
similar_images = similar_images['documents']
images = [file, *similar_images[0]]
image = np.hstack([np.array(Image.open(image).resize((512, 512)))
                   for image in images])
Image.fromarray(image).show()
