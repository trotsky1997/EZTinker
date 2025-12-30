"""GSM8K dataset loader with Math-Verify evaluation."""

import re

from datasets import load_dataset

# Try to import math-verify for robust evaluation
try:
    from math_verify.grader import verify as math_verify
    from math_verify.parser import parse as math_parse

    MATH_VERIFY_AVAILABLE = True
except ImportError:
    MATH_VERIFY_AVAILABLE = False

    def math_verify(gold, pred, **kwargs):
        return str(gold).strip() == str(pred).strip()

    def math_parse(x):
        return x


class GSM8KDataset:
    """GSM8K dataset wrapper with Math-Verify based evaluation."""

    def __init__(
        self, split: str = "train", max_samples: int | None = None, use_math_verify: bool = False
    ):
        """Initialize GSM8K dataset.

        Args:
            split: Dataset split ('train' or 'test')
            max_samples: Maximum number of samples to load (None for all)
            use_math_verify: Whether to use Math-Verify (requires separate installation)
        """
        self.split = split
        self.max_samples = max_samples
        self.use_math_verify = use_math_verify

        # Load dataset from HuggingFace
        print(f"Loading GSM8K {split} split...")
        dataset = load_dataset("openai/gsm8k", "main", split=split)

        if max_samples:
            dataset = dataset.select(range(max_samples))
            print(f"  Selected {max_samples} samples")

        # Parse and store data
        self.data = []
        for item in dataset:
            self.data.append(
                {
                    "question": item["question"],
                    "ground_truth": item["answer"],
                    "extracted_answer": self._extract_answer(item["answer"]),
                }
            )

        print(f"✓ Loaded {len(self.data)} examples")

        # Optional Math-Verify setup (would need additional installation/config)
        self.evaluator = None
        if self.use_math_verify:
            print("  Initializing Math-Verify...")
            try:
                from math_verify.verify_math_answer import instantiate_evaluator  # type: ignore

                self.evaluator = instantiate_evaluator(
                    model="gpt-3.5-turbo",
                    model_dir=None,
                    cache_dir="data/math-verify-cache",
                    api_key=None,
                    data_dir="data/math-verify-results",
                    temperature=0,
                    enable_log=False,
                )
                print("  ✓ Math-Verify initialized")
            except Exception as e:
                print(f"  ⚠ Math-Verify initialization failed: {e}")
                print("  Fallback to basic evaluation")

    def _extract_answer(self, text: str) -> str:
        """Extract numeric answer from GSM8K ground truth format.

        Args:
            text: Ground truth answer text with #### separator

        Returns:
            Extracted numeric answer as string
        """
        # GSM8K uses format: "solution steps... ####123"
        if "####" in text:
            answer = text.split("####")[-1].strip()
        else:
            # Fallback: extract last number
            numbers = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text)
            answer = numbers[-1] if numbers else ""
        return answer

    def get_example_question(self, idx: int) -> tuple[str, str, str]:
        """Get example by index.

        Args:
            idx: Example index

        Returns:
            Tuple of (question, formatted_prompt, ground_truth_answer)
        """
        item = self.data[idx]
        question = item["question"]
        answer = item["extracted_answer"]

        # Format as instruction for Qwen2 model
        prompt = (
            "Please solve the following math problem step by step. "
            "Provide your final answer as a single number.\n\n"
            f"Question: {question}\n\n"
            "Let's think step by step."
        )

        return question, prompt, answer

    def evaluate_answer(
        self, model_response: str, ground_truth_str: str, question: str = ""
    ) -> dict:
        """Evaluate if model response matches ground truth using multiple strategies.

        Now uses math-verify for robust mathematical answer comparison.

        Args:
            model_response: Model's generated response text
            ground_truth_str: Ground truth numerical answer (extracted)
            question: Optional question text for better evaluation context

        Returns:
            Dict with evaluation results:
                - is_correct: bool
                - ground_truth: str
                - pred_answer: str
                - strategy: str (which method was used)
                - confidence: float (0.0 to 1.0, depends on match quality)
                - math_verify_result: bool (True if math-verify passed)
        """
        # Try to extract prediction from model response
        pred_answer = self._extract_prediction(model_response, question)

        result = {
            "ground_truth": ground_truth_str,
            "pred_answer": pred_answer,
            "is_correct": False,
            "strategy": "math_verify",
            "confidence": 0.0,
            "math_verify_available": MATH_VERIFY_AVAILABLE,
        }

        if not pred_answer or not ground_truth_str:
            result["warning"] = "No prediction or ground truth to evaluate"
            return result

        # Use Math-Verify if available
        if MATH_VERIFY_AVAILABLE:
            # Parse both answers (ensures expressions are in correct format)
            try:
                gold_parsed = math_parse(ground_truth_str.strip())
                pred_parsed = math_parse(pred_answer.strip())

                # Verify using math-verify grader
                math_verify_result = math_verify(
                    gold_parsed,
                    pred_parsed,
                    float_rounding=6,
                    numeric_precision=15,
                    strict=False,  # Allow more flexible matching
                    timeout_seconds=3,
                    raise_on_error=False,
                )

                result.update(
                    {
                        "is_correct": math_verify_result,
                        "math_verify_result": math_verify_result,
                        "confidence": 1.0 if math_verify_result else 0.0,
                    }
                )

                return result

            except Exception as e:
                result["math_verify_error"] = str(e)
                result["strategy"] = "math_verify_fallback"

        # Fallback to basic string/numeric matching
        pred_clean = self._normalize_number(pred_answer)
        gold_clean = self._normalize_number(ground_truth_str)

        # Direct string equality
        is_equal = pred_clean == gold_clean

        # Numeric equality
        is_equiv = False
        try:
            pred_float = float(pred_clean.replace(",", ""))
            gold_float = float(gold_clean.replace(",", ""))
            is_equiv = abs(pred_float - gold_float) < 1e-4
        except ValueError:
            pass

        # Combined correctness
        is_correct = is_equal or is_equiv

        # Assign confidence
        confidence = 1.0 if is_equal else (0.9 if is_equiv else 0.0)

        result.update(
            {
                "is_correct": is_correct,
                "confidence": confidence,
                "strategy": result.get("strategy", "basic"),
                "is_equal": is_equal,
                "is_equiv": is_equiv,
            }
        )

        return result

    def _normalize_number(self, text: str) -> str:
        """Clean and normalize number.

        Args:
            text: Text with number

        Returns:
            Cleaned number (e.g., "1", "2.5", "-37")
        """
        text = text.strip()

        # Remove common suffixes and prefixes
        for bad in ["the answer is ", "answer: ", "=", "$", "%", " degrees", ".0"]:
            text = text.replace(bad, "")

        # Remove comma separators
        text = text.replace(",", "")

        # Extract number
        # Try standard number
        match = re.search(r"-?\d+(?:\.\d+)?", text)
        if match:
            return match.group(0)

        return text.strip()

    def _extract_prediction(self, model_response: str, question: str = "") -> str:
        """Extract numerical prediction from model response.

        Args:
            model_response: Model's generated response text
            question: Optional question for context

        Returns:
            Extracted numerical answer as string, or original text if no number found
        """
        # Remove commas from response for number extraction
        response = model_response.replace(",", "")

        # Strategy 1: Look for "answer is NUMBER" pattern
        answer_is_pattern = r"(?:the\s+)?answer\s+is\s+(-?\d+(?:\.\d+)?)"
        match = re.search(answer_is_pattern, response, re.IGNORECASE)
        if match:
            return match.group(1)

        # Strategy 2: Look for boxed answer
        boxed_pattern = r"\\boxed\{(-?\d+(?:\.\d+)?)\}"
        match = re.search(boxed_pattern, response)
        if match:
            return match.group(1)

        # Strategy 3: Extract last number
        numbers = re.findall(r"-?\d+(?:\.\d+)?", response)
        if numbers:
            return numbers[-1]

        # Strategy 4: No clear number, return last word that might be a conversion
        # Fallback: return the whole response for Math-Verify to handle
        return response

    def __len__(self) -> int:
        """Return number of examples."""
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        """Get example by index."""
        return self.data[idx]


if __name__ == "__main__":
    # Quick test
    dataset = GSM8KDataset(split="train", max_samples=10)
    print(f"\nDataset size: {len(dataset)} examples")

    # Test first example
    question, prompt, answer = dataset.get_example_question(0)
    print(f"\nQuestion: {question}")
    print(f"\nAnswer: {answer}")

    # Test evaluation with correct answer
    correct_response = "The answer is 15."
    result = dataset.evaluate_answer(correct_response, answer, question)
    print(f"\nCorrect answer evaluation: {result}")

    # Test evaluation with incorrect answer
    incorrect_response = "The answer is 20."
    result2 = dataset.evaluate_answer(incorrect_response, answer, question)
    print(f"\nIncorrect answer evaluation: {result2}")

    print("\n✓ GSM8K dataset loader test completed")
