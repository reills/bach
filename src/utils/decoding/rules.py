from typing import Dict, List, Optional, Tuple
import torch

class MusicRules:
    def __init__(self, vocab: Dict[str, int]) -> None:
        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in vocab.items()}
        
        # Pre-cache some categories
        self.mel_tokens = {k: v for k, v in vocab.items() if k.startswith("MEL_INT12_")}
        self.harm_class_tokens = {k: v for k, v in vocab.items() if k.startswith("HARM_CLASS_")}
        
        # Consonant classes: 0 (Unison), 3 (m3), 4 (M3), 7 (P5), 8 (m6), 9 (M6)
        self.consonant_classes = {"0", "3", "4", "7", "8", "9"}

    def apply_rules(self, logits: torch.Tensor, alpha: float) -> torch.Tensor:
        """
        Rescores logits based on musical rules.
        alpha: strictness (0.0 = no rules, 1.0 = strict)
        """
        if alpha <= 0:
            return logits

        # We'll use a modified logit set
        new_logits = logits.clone()
        
        for token, idx in self.vocab.items():
            penalty = 0.0
            
            # 1. Melodic Leap Penalty (|MEL_INT12| > 7)
            if token.startswith("MEL_INT12_"):
                try:
                    val = int(token.split("_")[-1].replace("+", ""))
                    if abs(val) > 7:
                        penalty += 2.0 * alpha
                except ValueError:
                    pass
            
            # 2. Consonance Bias (HARM_CLASS)
            if token.startswith("HARM_CLASS_"):
                cls = token.split("_")[-1]
                if cls != "NA" and cls not in self.consonant_classes:
                    penalty += 1.0 * alpha
            
            if penalty > 0:
                new_logits[:, idx] -= penalty
                
        return new_logits
