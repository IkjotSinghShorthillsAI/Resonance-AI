import os
import json
import csv
import time
import logging
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel
from google import generativeai as genai

# Load environment variables
load_dotenv()

def setup_logging():
    logging.basicConfig(
        filename="processing.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

class QA(BaseModel):
    question: str
    answer: str
    expected_context: str
    question_type: str

class QAProcessor:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
    
    def read_file(self, file_path: str) -> str:
        if file_path.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as file:
                return file.read()
        return ""

    def generate_qa(self, text: str) -> List[QA]:
        prompt = f'''You are an expert dataset creator tasked with generating a high-quality golden dataset for evaluating a Retrieval-Augmented Generation (RAG) model.
        ...
        (same prompt as before)
        '''
        
        response = self.client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config={'response_mime_type': 'application/json', 'response_schema': list[QA]}
        )
        
        try:
            qa_pairs = json.loads(response.text)
            return [QA(**qa) for qa in qa_pairs] if isinstance(qa_pairs, list) else []
        except json.JSONDecodeError:
            return []

class FileProcessor:
    def __init__(self, qa_processor: QAProcessor, progress_file: str = "progress.json"):
        self.qa_processor = qa_processor
        self.progress_file = progress_file
        self.processed_files = self.load_progress()
    
    def load_progress(self) -> set:
        if os.path.exists(self.progress_file):
            with open(self.progress_file, "r", encoding="utf-8") as f:
                return set(json.load(f))
        return set()
    
    def save_progress(self):
        with open(self.progress_file, "w", encoding="utf-8") as f:
            json.dump(list(self.processed_files), f, indent=4)
        logging.info("Progress saved: %d files processed.", len(self.processed_files))
    
    def save_qa_data(self, output_json: str, output_csv: str, all_qa: List[QA]):
        with open(output_json, "w", encoding="utf-8") as json_file:
            json.dump([qa.dict() for qa in all_qa], json_file, indent=4, ensure_ascii=False)
        
        with open(output_csv, "w", encoding="utf-8", newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Question", "Answer"])
            for qa in all_qa:
                writer.writerow([qa.question, qa.answer])
        
        logging.info("Q&A data saved: %d pairs generated.", len(all_qa))
    
    def process_folders(self, folder_paths: List[str], output_json: str, output_csv: str):
        all_qa = []
        count = 0
        
        logging.info("Starting processing for folders: %s", ", ".join(folder_paths))
        
        for folder_path in folder_paths:
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                
                if filename in self.processed_files or not os.path.isfile(file_path) or not filename.endswith(".txt"):
                    continue  
                
                logging.info("Processing file: %s from %s", filename, folder_path)
                print(f"Processing: {filename} from {folder_path}")
                
                try:
                    text = self.qa_processor.read_file(file_path)
                    qa_pairs = self.qa_processor.generate_qa(text)
                    all_qa.extend(qa_pairs)
                    self.processed_files.add(filename)
                    count += 1
                    
                    logging.info("Generated %d Q&A pairs for %s", len(qa_pairs), filename)
                    print(f"Generated {len(qa_pairs)} Q&A pairs")
                except Exception as e:
                    logging.error("Error processing %s: %s", filename, str(e))
                    continue
                
                if count % 14 == 0:
                    self.save_qa_data(output_json, output_csv, all_qa)
                    self.save_progress()
                    logging.info("Checkpoint reached: %d files processed. Waiting to avoid timeout.", count)
                    print(f"Checkpoint saved: {count} files processed. Waiting to avoid timeout.")
                    time.sleep(30)
        
        self.save_qa_data(output_json, output_csv, all_qa)
        self.save_progress()
        logging.info("Final save complete. Processed %d new files.", count)
        print(f"Final save complete. Processed {count} new files.")

if __name__ == "__main__":
    setup_logging()
    api_key = os.getenv('GOOGLE_API_KEY')
    qa_processor = QAProcessor(api_key=api_key)
    file_processor = FileProcessor(qa_processor)
    
    folder_paths = ["path/to/folder1", "path/to/folder2"]  # Replace with actual paths
    output_json = "output.json"
    output_csv = "output.csv"
    
    file_processor.process_folders(folder_paths, output_json, output_csv)
