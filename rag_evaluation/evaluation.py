from src.ragchain.configure import ConfigLoader, PineconeClient
from src.ragchain.vector_embeddings import PineconeIndexManager
from src.ragchain.rag_pipeline import RAGPipeline
import os
import json
import numpy as np
import pandas as pd
import random
import time
import nltk
from dotenv import load_dotenv
from nltk.translate.bleu_score import sentence_bleu
from sentence_transformers import SentenceTransformer, util
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from nltk.translate.meteor_score import meteor_score
from sklearn.metrics import precision_score, recall_score, f1_score
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from pinecone import ServerlessSpec, Pinecone


class PipelineTester:
    def __init__(self, log_file="evaluation_log.txt"):
        load_dotenv()
        self.api_key = os.getenv("PINECONE_API_KEY")
        self.gem_api_key = os.getenv("GEM_API_KEY")
        self.pinecone_client = Pinecone(api_key=self.api_key)
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.log_file = log_file
        
        # Setup Pinecone and RAG pipeline
        config = ConfigLoader()
        pinecone_client = PineconeClient(api_key=config.pinecone_api_key)
        pinecone_manager = PineconeIndexManager(pinecone_client=pinecone_client)
        retriever = pinecone_manager.get_retriever()
        self.rag_pipeline = RAGPipeline(retriever)
    
    def log(self, message):
        with open(self.log_file, "a") as log:
            log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")
    
    def bleu_score(self, reference, candidate):
        return sentence_bleu([reference.split()], candidate.split())
    
    def rouge_score(self, reference, candidate):
        return self.scorer.score(reference, candidate)
    
    def bert_score(self, references, candidates):
        P, R, F1 = bert_score(candidates, references, lang="en", rescale_with_baseline=True)
        return P.mean().item(), R.mean().item(), F1.mean().item()
    
    def meteor_score(self, reference, candidate):
        reference_tokens = reference.split()  # Tokenizing reference
        candidate_tokens = candidate.split()  # Tokenizing candidate
        return meteor_score([reference_tokens], candidate_tokens)

    
    def semantic_similarity(self, ref, cand):
        ref_embedding = self.sentence_model.encode(ref, convert_to_tensor=True)
        cand_embedding = self.sentence_model.encode(cand, convert_to_tensor=True)
        return util.pytorch_cos_sim(ref_embedding, cand_embedding).item()
    
    def test_pipeline(self, query):
        response = self.rag_pipeline.answer_question(query)
        return response["answer"]
    
    def evaluate_from_json(self, json_file, output_file="evaluation_results.csv"):
        with open(json_file, "r") as f:
            data = json.load(f)
        
        results = []
        for i, item in enumerate(data, 1):
            question = item["question"]
            expected_answer = item["answer"]
            generated_answer = self.test_pipeline(question)
            
            scores = {
                "question": question,
                "expected_answer": expected_answer,
                "generated_answer": generated_answer,
                "bleu": self.bleu_score(expected_answer, generated_answer),
                "rouge1": self.rouge_score(expected_answer, generated_answer)["rouge1"].fmeasure,
                "rouge2": self.rouge_score(expected_answer, generated_answer)["rouge2"].fmeasure,
                "rougeL": self.rouge_score(expected_answer, generated_answer)["rougeL"].fmeasure,
                "bert_precision": self.bert_score([expected_answer], [generated_answer])[0],
                "bert_recall": self.bert_score([expected_answer], [generated_answer])[1],
                "bert_f1": self.bert_score([expected_answer], [generated_answer])[2],
                "meteor": self.meteor_score(expected_answer, generated_answer),
                "semantic_similarity": self.semantic_similarity(expected_answer, generated_answer)
            }
            results.append(scores)
            
            if i % 10 == 0:
                df = pd.DataFrame(results)
                df.to_csv(output_file, index=False)
                self.log(f"Saved results after {i} test cases. Sleeping for 60 seconds.")
                time.sleep(60)
        
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
        self.log(f"Evaluation completed. Results saved to {output_file}.")
    
if __name__ == "__main__":
    tester = PipelineTester()
    json_file = "/home/shtlp_0096/Desktop/coding/rag_project/tests/golden_set.json"
    tester.evaluate_from_json(json_file,'test_case_report.csv')
