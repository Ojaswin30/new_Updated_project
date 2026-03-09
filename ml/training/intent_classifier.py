"""
Intent Classifier: Determines query intent (visual / attribute / hybrid)
Based on parsed constraints and text characteristics
"""
from typing import Dict, Tuple
import re

class IntentClassifier:
    """
    Classifies query intent into three categories:
    - visual: dominated by visual attributes (color, brand, product type only)
    - attribute: dominated by specific constraints (size, material, price, keywords)
    - hybrid: balanced mix of both
    
    This is rule-based and deterministic (no training needed).
    """
    
    def __init__(self):
        pass
    
    def classify(self, text: str, constraints: Dict) -> Tuple[str, Dict[str, float]]:
        """
        Returns: (intent_label, confidence_scores)
        
        intent_label: 'visual' | 'attribute' | 'hybrid'
        confidence_scores: {
            'visual': 0.0-1.0,
            'attribute': 0.0-1.0,
            'hybrid': 0.0-1.0
        }
        """
        visual_score = self._compute_visual_score(constraints)
        attribute_score = self._compute_attribute_score(constraints)
        
        # Compute intent
        total = visual_score + attribute_score
        
        if total == 0:
            # No constraints extracted - default to hybrid with low confidence
            return 'hybrid', {'visual': 0.3, 'attribute': 0.3, 'hybrid': 0.4}
        
        visual_ratio = visual_score / total
        attribute_ratio = attribute_score / total
        
        # Decision thresholds
        VISUAL_THRESHOLD = 0.7
        ATTRIBUTE_THRESHOLD = 0.7
        
        if visual_ratio >= VISUAL_THRESHOLD:
            intent = 'visual'
        elif attribute_ratio >= ATTRIBUTE_THRESHOLD:
            intent = 'attribute'
        else:
            intent = 'hybrid'
        
        # Return confidence scores (normalized)
        confidences = {
            'visual': visual_ratio,
            'attribute': attribute_ratio,
            'hybrid': 1.0 - abs(visual_ratio - attribute_ratio)  # balance measure
        }
        
        return intent, confidences
    
    def _compute_visual_score(self, constraints: Dict) -> float:
        """
        Visual features: category, color
        These are things you can see in an image
        """
        score = 0.0
        
        if constraints.get('category'):
            score += 1.0
        
        if constraints.get('color'):
            score += 1.0
        
        return score
    
    def _compute_attribute_score(self, constraints: Dict) -> float:
        """
        Attribute features: size, material, price, keywords
        These are things you typically need text/metadata to identify
        """
        score = 0.0
        
        if constraints.get('size'):
            score += 1.5  # Size is a strong attribute signal
        
        if constraints.get('material'):
            score += 1.0
        
        if constraints.get('price_min') or constraints.get('price_max'):
            score += 1.5  # Price is a strong attribute signal
        
        keywords = constraints.get('keywords', [])
        if keywords:
            # More keywords = stronger attribute intent
            score += min(2.0, len(keywords) * 0.5)
        
        return score


def classify_intent(text: str, constraints: Dict) -> Tuple[str, Dict[str, float]]:
    """Convenience function"""
    classifier = IntentClassifier()
    return classifier.classify(text, constraints)


# ==================== TESTING ====================
if __name__ == "__main__":
    from ml.src.pipeline.constraint_parser import ConstraintParser
    
    parser = ConstraintParser()
    classifier = IntentClassifier()
    
    test_queries = [
        "red dress",  # visual
        "black shoes",  # visual
        "cotton shirt size M under 1500",  # attribute
        "leather jacket xl",  # hybrid
        "blue denim jeans 32 waist",  # hybrid
        "running shoes",  # visual
        "formal shirt cotton medium price under 2000",  # attribute
    ]
    
    print("Intent Classification Test:\n")
    for query in test_queries:
        constraints = parser.parse(query)
        intent, scores = classifier.classify(query, constraints)
        
        print(f"Query: '{query}'")
        print(f"  Intent: {intent}")
        print(f"  Scores: visual={scores['visual']:.2f}, "
              f"attribute={scores['attribute']:.2f}, "
              f"hybrid={scores['hybrid']:.2f}")
        print(f"  Constraints: {constraints}\n")