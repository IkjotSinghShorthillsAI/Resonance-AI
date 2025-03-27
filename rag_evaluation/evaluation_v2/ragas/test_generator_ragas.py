import os
from dotenv import load_dotenv
from src.ragchain.configure import ConfigLoader
from src.ragchain.vector_embeddings import EmbeddingModel, DocumentProcessor
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset import TestsetGenerator
from langchain_google_genai import ChatGoogleGenerativeAI

class ModelInitializer:
    def __init__(self):
        self.config_loader = ConfigLoader()
        self.config = {
            "model": "gemini-2.0-flash-lite",
            "temperature": 0.4,
            "max_tokens": None,
            "top_p": 0.8,
        }

    def initialize_models(self):
        generator_llm = LangchainLLMWrapper(ChatGoogleGenerativeAI(
            model=self.config["model"],
            temperature=self.config["temperature"],
            max_tokens=self.config["max_tokens"],
            top_p=self.config["top_p"],
        ))

        generator_embeddings = LangchainEmbeddingsWrapper(EmbeddingModel.get_embeddings())
        
        return generator_llm, generator_embeddings

class TestSetGenerator:
    def __init__(self, llm, embedding_model, batch_size=5, testset_size=10):
        self.generator = TestsetGenerator(llm=llm, embedding_model=embedding_model)
        self.batch_size = batch_size
        self.testset_size = testset_size
    
    def generate_test_sets(self, documents, output_dir="/home/shtlp_0096/Desktop/coding/rag_project/rag_evaluation/evaluation_v2/ragas/dataset"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        for i in range(0, len(documents), self.batch_size):
            batch = documents[i:i + self.batch_size]
            try:
                dataset = self.generator.generate_with_langchain_docs(batch, testset_size=self.testset_size)
                output_path = os.path.join(output_dir, f"dataset_{i}.jsonl")
                dataset.to_jsonl(path=output_path)
                print(f"Successfully processed batch {i//self.batch_size + 1}, saved to {output_path}")
            except Exception as e:
                print(f"Error processing batch {i//self.batch_size + 1}: {e}")
                continue

if __name__ == "__main__":
    data_path = "/home/shtlp_0096/Desktop/coding/rag_project/data/singular_websites"
    output_directory = "rag_evaluation/evaluation_v2/ragas/generated_testsets"
    
    doc_processor = DocumentProcessor(data_path)
    documents = doc_processor.load_txt_files()
    
    
    model_initializer = ModelInitializer()
    llm, embeddings = model_initializer.initialize_models()
    
    test_set_generator = TestSetGenerator(llm, embeddings)
    test_set_generator.generate_test_sets(documents, output_dir=output_directory)
