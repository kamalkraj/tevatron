import json
import random
from sentence_transformers import SentenceTransformer, models
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from datasets import load_dataset

random.seed(42)

# Load a model
transformer = models.Transformer("model_msmarco_passage_multiquery")
pooling = models.Pooling(transformer.get_word_embedding_dimension(),pooling_mode="cls")
# norm = models.Normalize()
model = SentenceTransformer(modules=[transformer, pooling])

# Load the Quora IR dataset (https://huggingface.co/datasets/BeIR/quora, https://huggingface.co/datasets/BeIR/quora-qrels)
corpus = load_dataset("BeIR/quora", "corpus", split="corpus")
queries = load_dataset("BeIR/quora", "queries", split="queries")
relevant_docs_data = load_dataset("BeIR/quora-qrels", split="validation")

# Shrink the corpus size heavily to only the relevant documents + 10,000 random documents
required_corpus_ids = list(map(str, relevant_docs_data["corpus-id"]))
required_corpus_ids += random.sample(corpus["_id"], k=10_000)
corpus = corpus.filter(lambda x: x["_id"] in required_corpus_ids)

# Convert the datasets to dictionaries
corpus = dict(zip(corpus["_id"], corpus["text"]))  # Our corpus (cid => document)
queries = dict(zip(queries["_id"], queries["text"]))  # Our queries (qid => question)
relevant_docs = {}  # Query ID to relevant documents (qid => set([relevant_cids])
for qid, corpus_ids in zip(relevant_docs_data["query-id"], relevant_docs_data["corpus-id"]):
    qid = str(qid)
    corpus_ids = str(corpus_ids)
    if qid not in relevant_docs:
        relevant_docs[qid] = set()
    relevant_docs[qid].add(corpus_ids)

# Given queries, a corpus and a mapping with relevant documents, the InformationRetrievalEvaluator computes different IR metrics.
ir_evaluator = InformationRetrievalEvaluator(
    queries=queries,
    corpus=corpus,
    relevant_docs=relevant_docs,
    name="BeIR-quora-dev",
)
results = ir_evaluator(model)
print(json.dumps(results))