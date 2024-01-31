# %%
from PyPDF2 import PdfMerger
import os
# from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
# from tqdm import notebook
import openai
import json
import openai
import pypdfium2 as pdfium
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
import torch
import json
import spacy
from spacy.lang.en import English
from IPython.display import display, Markdown, Latex
# from IPython.display import display, Markdown, Latex

embedder_model = "msmarco-MiniLM-L6-cos-v5"

def flatten(x):
    return [item for sublist in x for item in sublist]


def cleanslide(slide):
    # remove empty lines
    lines = slide.splitlines()
    lines = [l for l in lines if l.strip() != '']
    return ' '.join(lines)

def answer(passage):
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages =  [{"role": "system", "content": "You are a highly intelligent de-noising bot. Prompts from the user are OCR of slide shows which are noisy, you will de-noise the contents of the slide as accurately as possible."},
                     {"role": "user", "content": passage}],
        # prompt= Prompt + " P: " + "\"" + passage + "\"",
        temperature=0,
        # max_tokens= int(len(passage) / 3),
        # top_p=1,
        # frequency_penalty=0.0,
        # presence_penalty=0.0,
        # stop=["end of passage"]
        )
    return response.choices[0].message.content

def search(query, corpus_embeddings, k):
    embedder = SentenceTransformer(embedder_model)
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
    cos_scores = cos_scores.cpu()
    top_results = torch.topk(cos_scores, k=k)
    responses = []
    for score, idx in zip(top_results[0], top_results[1]):
        responses.append(idx.item())
    return responses
class SlideSearcher:
    def __init__(self, machanswers, imgs, corpus_embeddings, ind2ind):
        self.machanswers = machanswers
        self.imgs = imgs
        self.corpus_embeddings = corpus_embeddings
        self.ind2ind = ind2ind
    def get_top_k(self, query, topk = 5):
        previnds = []
        corpus_embeddings = self.corpus_embeddings
        ind2ind = self.ind2ind
        machanswers = self.machanswers
        imgs = self.imgs
        for i in search(query, corpus_embeddings, topk):
            if self.ind2ind[i] not in previnds:
                display(Markdown(machanswers[ind2ind[i]]))
                display(imgs[ind2ind[i]])
                previnds.append(ind2ind[i])
            print(ind2ind[i] + 1)

#%%

def make_slide_searcher(slides_path, tesseract_location, output_location):
    openai.api_key = os.environ['OPENAI_API_KEY']

    path = Path(slides_path)
    tesseract_path = Path(tesseract_location)
    out_path = Path(output_location)
    out_path.mkdir(parents=True, exist_ok=True)

    files = os.listdir(path)
    merger = PdfMerger()
    mergersols = []


    for f in files:
        merger.append(path / f)
        mergersols.append(len(pdfium.PdfDocument(path / f)))

    merger.write(out_path / 'result.pdf')
    
    filepath = out_path / 'result.pdf'
    pdf = pdfium.PdfDocument(filepath)
    page_indices = [i for i in range(len(pdf))]
    br = [pdf.get_page(i).render().to_pil() for i in range(len(pdf))]
    pytesseract.pytesseract.tesseract_cmd = str(tesseract_path.absolute())
    
    if not os.path.exists(out_path / "cleanedsents.json"):
        texts = []
        fails = []
        for (i, b) in enumerate(br):
            print(f'{i}/{len(br)}', end='\r')
            try:
                texts.append(pytesseract.image_to_string(b, lang='eng+equ', timeout=1, config='--psm 11'))
            except:
                texts.append('')
                fails.append(i) 
        cleaned = [cleanslide(t) for t in texts]
        with open(out_path / "cleanedsents.json", 'w+') as f:
            json.dump(cleaned, f)
        
    with open(out_path / "cleanedsents.json", "r") as f:
        cleaned = json.load(f)
    
    
    if not os.path.exists(out_path / "machanswers.json"):
        machanswers = []
        for (i, t) in enumerate(cleaned):
            print(f'{i}/{len(cleaned)}', end='\r')
            machanswers.append(answer(t))
            if os.path.exists('stop.txt'):
                break
        with open(out_path / "machanswers.json", 'w+') as f:
            json.dump(machanswers, f)
            
    with open(out_path / "machanswers.json", "r") as f:
        machanswers = json.load(f)
    
    ind2vec = [list(zip([i] * len(s.split('.')), s.split('.'))) for (i, s) in enumerate(machanswers)]
    ind2ind = {i : iv[0] for (i, iv) in  enumerate(flatten(ind2vec))}
    corpus = [s[1] for s in flatten(ind2vec)]
    imgs = br
    
    embedder = SentenceTransformer(embedder_model)
    nlp = English()
    nlp.add_pipe('sentencizer')
    
    corpus = [str(x) for x in list(nlp('\n'.join(machanswers)).sents)]
    corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)
    
    return SlideSearcher(machanswers, imgs, corpus_embeddings, ind2ind)

