import os
import json
import csv
import logging
import numpy as np
from dotenv import load_dotenv
from src.ragchain.configure import ConfigLoader, PineconeClient
from src.ragchain.vector_embeddings import EmbeddingModel, PineconeIndexManager
from src.ragchain.rag_pipeline import RAGPipeline
from ragas import EvaluationDataset, evaluate
from ragas.metrics import (
    LLMContextRecall, Faithfulness, SemanticSimilarity, AnswerCorrectness
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

class JSONLProcessor:
    @staticmethod
    def traverse_jsonl(file_path):
        ui_list, ref_list = [], []
        if not os.path.exists(file_path):
            logging.error(f"File not found: {file_path}")
            return [], []
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                for idx, line in enumerate(file, start=1):
                    try:
                        obj = json.loads(line.strip())
                        user_input, reference = obj.get("user_input", "N/A"), obj.get("reference", "N/A")
                        ui_list.append(user_input)
                        ref_list.append(reference)
                        logging.info(f"Processed line {idx}: {user_input[:50]}...")
                    except json.JSONDecodeError as e:
                        logging.error(f"JSON decode error at line {idx}: {str(e)}")
        except Exception as e:
            logging.error(f"Error processing {file_path}: {str(e)}")
        return ui_list, ref_list

class Evaluator:
    def __init__(self):
        self.eval_llm = LangchainLLMWrapper(ChatGoogleGenerativeAI(model="gemini-2.0-flash"))
        self.embeddings = LangchainEmbeddingsWrapper(EmbeddingModel.get_embeddings())
        self.metrics = [
            LLMContextRecall(max_retries=3), Faithfulness(max_retries=3),
            SemanticSimilarity(), AnswerCorrectness(max_retries=3)
        ]

    def evaluate_responses(self, dataset):
        evaluation_dataset = EvaluationDataset.from_list(dataset)
        return evaluate(dataset=evaluation_dataset, embeddings=self.embeddings, metrics=self.metrics, llm=self.eval_llm)

class ResultHandler:
    @staticmethod
    def replace_nan(value):
        return "N/A" if isinstance(value, (float, np.float32, np.float64)) and np.isnan(value) else value

    @staticmethod
    def save_result(output_csv, output_json, data_item):
        try:
            with open(output_csv, mode="a", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow([
                    data_item["user_input"], " | ".join(data_item["retrieved_contexts"]),
                    data_item["response"], data_item["reference"],
                    data_item["evaluation_result"]["context_recall"],
                    data_item["evaluation_result"]["faithfulness"],
                    data_item["evaluation_result"]["semantic_similarity"],
                    data_item["evaluation_result"]["answer_correctness"]
                ])
            logging.info(f"Saved result to CSV: {output_csv}")
            json_data = [] if not os.path.exists(output_json) else json.load(open(output_json, "r", encoding="utf-8"))
            json_data.append(data_item)
            json.dump(json_data, open(output_json, "w", encoding="utf-8"), indent=4, ensure_ascii=False)
            logging.info(f"Saved result to JSON: {output_json}")
        except Exception as e:
            logging.error(f"Error saving result: {str(e)}")

class RAGASEvaluator:
    def __init__(self, pipeline, evaluator):
        self.pipeline = pipeline
        self.evaluator = evaluator

    def process_folder(self, folder_path, output_csv="evaluation_results.csv", output_json="evaluation_results.json"):
        if not os.path.exists(output_csv):
            with open(output_csv, mode="w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow([
                    "User Input", "Retrieved Contexts", "Response", "Reference",
                    "Context Recall", "Faithfulness", "Semantic Similarity", "Answer Correctness"
                ])
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".jsonl"):
                file_path = os.path.join(folder_path, file_name)
                logging.info(f"Processing file: {file_path}")
                sample_queries, expected_responses = JSONLProcessor.traverse_jsonl(file_path)
                dataset = []
                for query, reference in zip(sample_queries, expected_responses):
                    relevant_docs = self.pipeline.retriever.invoke(query)
                    response = self.pipeline.rag_chain.invoke({"input": query})
                    dataset.append({
                        "user_input": query, "retrieved_contexts": [doc.page_content for doc in relevant_docs],
                        "response": response["answer"], "reference": reference
                    })
                eval_scores = self.evaluator.evaluate_responses(dataset).scores
                for item, score in zip(dataset, eval_scores):
                    item["evaluation_result"] = {k: ResultHandler.replace_nan(v) for k, v in score.items()}
                    ResultHandler.save_result(output_csv, output_json, item)
                logging.info(f"Evaluation completed for: {file_path}")

# Execution
config = ConfigLoader()
pinecone_client = PineconeClient(api_key=config.pinecone_api_key)


pinecone_manager = PineconeIndexManager(pinecone_client=pinecone_client)

retriever = pinecone_manager.get_retriever()
pipeline = RAGPipeline(retriever=retriever,model="gemini-1.5-flash")

evaluator = Evaluator()
ragas_evaluator = RAGASEvaluator(pipeline, evaluator)
ragas_evaluator.process_folder("/home/shtlp_0096/Desktop/coding/rag_project/rag_evaluation/evaluation_v2/ragas/generated_testsets")
