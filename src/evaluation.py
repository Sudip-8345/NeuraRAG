"""
Evaluation Module

This module handles evaluation of the RAG system.
"""

from typing import List, Dict, Optional
import json


class EvaluationDataset:
    """Manages the evaluation dataset."""
    
    @staticmethod
    def get_evaluation_questions() -> List[Dict]:
        """
        Get the evaluation dataset with expected answer types.
        
        Returns:
            List of evaluation questions with metadata
        """
        return [
            {
                "id": 1,
                "question": "Can I return a product after 30 days?",
                "expected_answer_type": "answerable",
                "category": "refund",
                "notes": "Clearly stated in refund policy"
            },
            {
                "id": 2,
                "question": "What is the cost of overnight shipping?",
                "expected_answer_type": "answerable",
                "category": "shipping",
                "notes": "Specific price mentioned in shipping policy"
            },
            {
                "id": 3,
                "question": "How long does it take to process a refund?",
                "expected_answer_type": "answerable",
                "category": "refund",
                "notes": "Timeline clearly specified"
            },
            {
                "id": 4,
                "question": "Can I cancel a custom order?",
                "expected_answer_type": "partially_answerable",
                "category": "cancellation",
                "notes": "Policy mentions 2-hour window but conditions apply"
            },
            {
                "id": 5,
                "question": "Do you offer international shipping to Australia?",
                "expected_answer_type": "partially_answerable",
                "category": "shipping",
                "notes": "Policy mentions 100+ countries but not specific countries"
            },
            {
                "id": 6,
                "question": "What is your warranty policy for electronics?",
                "expected_answer_type": "unanswerable",
                "category": "warranty",
                "notes": "No warranty policy document provided"
            },
            {
                "id": 7,
                "question": "Can I use cryptocurrency to pay for shipping?",
                "expected_answer_type": "unanswerable",
                "category": "payment",
                "notes": "Payment methods not discussed in provided policies"
            },
            {
                "id": 8,
                "question": "What happens if I receive a defective product?",
                "expected_answer_type": "answerable",
                "category": "refund",
                "notes": "Detailed in special circumstances of refund policy"
            }
        ]


class Evaluator:
    """Evaluates RAG system responses."""
    
    def __init__(self):
        """Initialize the evaluator."""
        self.evaluation_results = []
    
    @staticmethod
    def evaluate_response(question_data: Dict, 
                         answer: str, 
                         retrieved_chunks: List[Dict]) -> Dict:
        """
        Evaluate a single response.
        
        Args:
            question_data: Question with expected answer type
            answer: Generated answer
            retrieved_chunks: Retrieved context chunks
            
        Returns:
            Evaluation result
        """
        expected_type = question_data['expected_answer_type']
        
        # Check for hallucination indicators
        has_fallback = (
            "don't have enough information" in answer.lower() or
            "contact customer service" in answer.lower() or
            "not available" in answer.lower()
        )
        
        # Scoring criteria
        accuracy_score = "❌"  # Default
        hallucination_score = "✅"  # Assume no hallucination unless detected
        clarity_score = "✅"  # Assume clear unless issues found
        
        # Evaluate based on expected answer type
        if expected_type == "answerable":
            if has_fallback:
                accuracy_score = "⚠️"  # Should answer but didn't
                note = "Failed to answer an answerable question"
            elif len(answer.strip()) > 20:  # Has substantial answer
                accuracy_score = "✅"
                note = "Correctly answered"
            else:
                accuracy_score = "❌"
                note = "Answer too brief or unclear"
                
        elif expected_type == "partially_answerable":
            if has_fallback:
                accuracy_score = "✅"  # Correctly identified incomplete info
                note = "Correctly acknowledged partial information"
            elif len(answer.strip()) > 20:
                accuracy_score = "⚠️"  # Answered but might be incomplete
                note = "Provided partial answer"
            else:
                accuracy_score = "❌"
                note = "Failed to provide useful response"
                
        elif expected_type == "unanswerable":
            if has_fallback:
                accuracy_score = "✅"  # Correctly refused to answer
                note = "Correctly identified lack of information"
            else:
                accuracy_score = "❌"
                hallucination_score = "❌"  # Likely hallucinated
                note = "Potential hallucination - answered unanswerable question"
        
        # Check answer clarity
        if len(answer.strip()) < 10:
            clarity_score = "❌"
        elif len(answer) > 500:
            clarity_score = "⚠️"  # Too verbose
        
        return {
            "question_id": question_data['id'],
            "question": question_data['question'],
            "expected_type": expected_type,
            "category": question_data['category'],
            "answer": answer,
            "num_chunks_retrieved": len(retrieved_chunks),
            "sources": [chunk['source'] for chunk in retrieved_chunks],
            "accuracy": accuracy_score,
            "hallucination_prevention": hallucination_score,
            "clarity": clarity_score,
            "note": note
        }
    
    def run_evaluation(self, rag_system, questions: Optional[List[Dict]] = None) -> List[Dict]:
        """
        Run evaluation on the RAG system.
        
        Args:
            rag_system: RAG system instance
            questions: Optional list of questions (uses default if None)
            
        Returns:
            List of evaluation results
        """
        if questions is None:
            questions = EvaluationDataset.get_evaluation_questions()
        
        results = []
        
        print("\n" + "="*80)
        print("RUNNING EVALUATION")
        print("="*80 + "\n")
        
        for q_data in questions:
            print(f"\nQuestion {q_data['id']}: {q_data['question']}")
            print(f"Expected: {q_data['expected_answer_type']}")
            print("-" * 80)
            
            # Get answer from RAG system
            response = rag_system.answer_question(q_data['question'])
            
            # Evaluate the response
            eval_result = self.evaluate_response(
                q_data,
                response['answer'],
                response.get('retrieved_chunks', [])
            )
            
            results.append(eval_result)
            
            # Print evaluation
            print(f"Answer: {response['answer'][:200]}{'...' if len(response['answer']) > 200 else ''}")
            print(f"\nScores: Accuracy={eval_result['accuracy']} | "
                  f"Hallucination Prevention={eval_result['hallucination_prevention']} | "
                  f"Clarity={eval_result['clarity']}")
            print(f"Note: {eval_result['note']}")
            print(f"Sources: {', '.join(eval_result['sources'])}")
        
        # Print summary
        self._print_summary(results)
        
        return results
    
    def _print_summary(self, results: List[Dict]) -> None:
        """Print evaluation summary."""
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80 + "\n")
        
        # Count scores
        accuracy_scores = [r['accuracy'] for r in results]
        hallucination_scores = [r['hallucination_prevention'] for r in results]
        clarity_scores = [r['clarity'] for r in results]
        
        def count_scores(scores):
            return {
                'pass': scores.count('✅'),
                'warning': scores.count('⚠️'),
                'fail': scores.count('❌')
            }
        
        acc_counts = count_scores(accuracy_scores)
        hall_counts = count_scores(hallucination_scores)
        clar_counts = count_scores(clarity_scores)
        
        print(f"Accuracy:              ✅ {acc_counts['pass']} | ⚠️ {acc_counts['warning']} | ❌ {acc_counts['fail']}")
        print(f"Hallucination Prevention: ✅ {hall_counts['pass']} | ⚠️ {hall_counts['warning']} | ❌ {hall_counts['fail']}")
        print(f"Clarity:               ✅ {clar_counts['pass']} | ⚠️ {clar_counts['warning']} | ❌ {clar_counts['fail']}")
        
        total = len(results)
        overall_pass = sum(1 for r in results if r['accuracy'] == '✅' and r['hallucination_prevention'] == '✅')
        
        print(f"\nOverall Pass Rate: {overall_pass}/{total} ({100*overall_pass/total:.1f}%)")
        print("="*80 + "\n")
    
    def save_results(self, results: List[Dict], filepath: str) -> None:
        """Save evaluation results to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {filepath}")
