import os
import json
import csv
import time
from dotenv import load_dotenv
from typing import List
from pydantic import BaseModel

# Google GenAI Imports
from google import genai

# Load environment variables
load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

class QA(BaseModel):
    question: str
    answer: str

class FileLoader:
    """Handles file reading from multiple folders."""
    @staticmethod
    def read_file(file_path: str) -> str:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()

class TestCaseGenerator:
    """Generates question-answer pairs using Google Generative AI."""
    def __init__(self):
        pass
    
    def generate_qa(self, text: str) -> List[dict]:
        prompt = f"""
        Generate 5 question-answer pairs based on the following text:
        ---
        {text}
        ---
        Format the output as a list of JSON objects with 'question' and 'answer'.
        """
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config={
                'response_mime_type': 'application/json',
                'response_schema': list[QA]
            }
        )
        try:
            qa_pairs = json.loads(response.text)
            return qa_pairs if isinstance(qa_pairs, list) else []
        except json.JSONDecodeError:
            return []

class QAStorage:
    """Handles storage of generated test cases."""
    @staticmethod
    def save_to_json(qa_list: List[dict], output_path: str):
        with open(output_path, "a", encoding="utf-8") as file:
            json.dump(qa_list, file, indent=4)
            file.write("\n")
    
    @staticmethod
    def save_to_csv(qa_list: List[dict], output_path: str):
        file_exists = os.path.isfile(output_path)
        with open(output_path, "a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["Question", "Answer"])
            for qa in qa_list:
                writer.writerow([qa["question"], qa["answer"]])

# Example usage
def main():
    folders = ["/home/shtlp_0096/Desktop/coding/rag_project/data/singular_websites", "/home/shtlp_0096/Desktop/coding/rag_project/data/scraped_genres", "/home/shtlp_0096/Desktop/coding/rag_project/data/instrument_data"]  # Replace with actual folder paths
    output_json = "/home/shtlp_0096/Desktop/coding/rag_project/tests/qa_output.json"
    output_csv = "/home/shtlp_0096/Desktop/coding/rag_project/tests/qa_output.csv"
    
    generator = TestCaseGenerator()
    count=0
    for folder in folders[:1]:
        for file_name in os.listdir(folder):
            file_path = os.path.join(folder, file_name)
            if file_path.endswith(".txt"):
                text = FileLoader.read_file(file_path)
                qa_pairs = generator.generate_qa(text)
                QAStorage.save_to_json(qa_pairs, output_json)
                QAStorage.save_to_csv(qa_pairs, output_csv)
                print(f"Processed: {file_path}")
                count+=1
                if count%14==0:
                    time.sleep(60)
    
    print("Test cases generation completed.")

if __name__ == "__main__":
    main()
