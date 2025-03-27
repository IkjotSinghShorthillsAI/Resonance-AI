import json
import random
from typing import List, Dict


class TestCaseSelector:
    """Class for selecting top test cases based on a scoring mechanism."""

    def __init__(self, input_file: str, output_file: str, top_n: int = 1000):
        self.input_file = input_file
        self.output_file = output_file
        self.top_n = top_n
        self.test_cases = self._load_test_cases()

    def _load_test_cases(self) -> List[Dict]:
        """Loads test cases from a JSON file."""
        with open(self.input_file, "r", encoding="utf-8") as file:
            return json.load(file)

    @staticmethod
    def _score_case(tc: Dict) -> float:
        """Scores a test case based on question & answer length with slight randomness."""
        q_len = len(tc.get("question", ""))
        a_len = len(tc.get("answer", ""))
        randomness = random.uniform(0, 0.05)  # Add slight randomness to prevent ties
        return q_len + a_len + randomness  # Longer Q&A generally means better depth

    def _rank_cases(self) -> List[Dict]:
        """Ranks test cases based on their scores."""
        return sorted(self.test_cases, key=self._score_case, reverse=True)

    def select_top_cases(self):
        """Selects the top N test cases and saves them to a JSON file."""
        ranked_cases = self._rank_cases()
        golden_cases = ranked_cases[:self.top_n]
        
        with open(self.output_file, "w", encoding="utf-8") as file:
            json.dump(golden_cases, file, indent=4, ensure_ascii=False)

        print(f"✅ Selected top {self.top_n} test cases and saved to {self.output_file}")


if __name__ == "__main__":
    selector = TestCaseSelector("/home/shtlp_0096/Desktop/coding/rag_project/tests/qa_output.json", "/home/shtlp_0096/Desktop/coding/rag_project/tests/golden_set.json", top_n=1200)
    selector.select_top_cases()