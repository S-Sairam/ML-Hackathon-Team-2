"""
1_HMM_Training.ipynb - Hidden Markov Model Training for Hangman
This notebook trains HMMs on the corpus to predict letter probabilities
"""

import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import pickle
import matplotlib.pyplot as plt

# ============================================================================
# PART 1: LOAD AND PREPROCESS CORPUS
# ============================================================================

def load_corpus(filename='corpus.txt'):
    """Load words from corpus file"""
    with open(filename, 'r') as f:
        words = [line.strip().upper() for line in f if line.strip()]
    print(f"Loaded {len(words)} words from corpus")
    return words

def group_words_by_length(words):
    """Group words by their length for separate HMM training"""
    word_groups = defaultdict(list)
    for word in words:
        word_groups[len(word)].append(word)
    
    print("\nWord length distribution:")
    for length in sorted(word_groups.keys()):
        print(f"Length {length}: {len(word_groups[length])} words")
    
    return word_groups


# ============================================================================
# PART 2: HMM IMPLEMENTATION
# ============================================================================

class HangmanHMM:
    """
    Hidden Markov Model for Hangman letter prediction
    
    Hidden States: Letter positions (0, 1, 2, ..., word_length-1)
    Observations: Letters at each position
    """
    
    def __init__(self, word_length):
        self.word_length = word_length
        self.alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        self.letter_to_idx = {letter: i for i, letter in enumerate(self.alphabet)}
        
        # Emission probabilities: P(letter | position)
        # Shape: (word_length, 26)
        self.emission_probs = np.zeros((word_length, 26))
        
        # Transition probabilities: P(position_t | position_t-1)
        # For simplicity, transitions are deterministic (sequential)
        self.transition_probs = np.eye(word_length)
        
        # Letter pair frequencies for bigram model
        self.bigram_counts = defaultdict(lambda: defaultdict(int))
        self.bigram_probs = {}
        
    def train(self, words):
        """Train HMM on words of specific length"""
        
        # Count letter occurrences at each position
        position_counts = [Counter() for _ in range(self.word_length)]
        
        for word in words:
            if len(word) != self.word_length:
                continue
                
            for pos, letter in enumerate(word):
                position_counts[pos][letter] += 1
                
            # Build bigram counts
            for i in range(len(word) - 1):
                self.bigram_counts[word[i]][word[i+1]] += 1
        
        # Convert counts to probabilities with Laplace smoothing
        alpha = 0.5  # Smoothing parameter
        
        for pos in range(self.word_length):
            total = sum(position_counts[pos].values()) + alpha * 26
            for letter in self.alphabet:
                count = position_counts[pos][letter] + alpha
                letter_idx = self.letter_to_idx[letter]
                self.emission_probs[pos][letter_idx] = count / total
        
        # Build bigram probabilities
        for letter1 in self.bigram_counts:
            total = sum(self.bigram_counts[letter1].values()) + alpha * 26
            self.bigram_probs[letter1] = {}
            for letter in self.alphabet:
                count = self.bigram_counts[letter1][letter] + alpha
                self.bigram_probs[letter1][letter] = count / total
    
    def predict_letter_probs(self, masked_word, guessed_letters):
        """
        Predict probability distribution over letters given current game state
        
        Args:
            masked_word: str, e.g., "_PP_E"
            guessed_letters: set of already guessed letters
            
        Returns:
            dict: {letter: probability} for all unguessed letters
        """
        
        if len(masked_word) != self.word_length:
            # Fallback to uniform distribution
            remaining = [l for l in self.alphabet if l not in guessed_letters]
            uniform_prob = 1.0 / len(remaining) if remaining else 0
            return {l: uniform_prob for l in remaining}
        
        # Aggregate probabilities across all blank positions
        letter_scores = defaultdict(float)
        blank_positions = [i for i, c in enumerate(masked_word) if c == '_']
        
        if not blank_positions:
            return {}
        
        for pos in blank_positions:
            # Get emission probabilities for this position
            for letter_idx, letter in enumerate(self.alphabet):
                if letter in guessed_letters:
                    continue
                
                prob = self.emission_probs[pos][letter_idx]
                
                # Boost probability based on bigram context
                # Check left neighbor
                if pos > 0 and masked_word[pos-1] != '_':
                    left_letter = masked_word[pos-1]
                    if left_letter in self.bigram_probs:
                        bigram_prob = self.bigram_probs[left_letter].get(letter, 0)
                        prob *= (1 + bigram_prob * 2)  # Weight bigrams
                
                # Check right neighbor
                if pos < len(masked_word) - 1 and masked_word[pos+1] != '_':
                    right_letter = masked_word[pos+1]
                    # Reverse bigram
                    if letter in self.bigram_probs:
                        bigram_prob = self.bigram_probs[letter].get(right_letter, 0)
                        prob *= (1 + bigram_prob * 2)
                
                letter_scores[letter] += prob
        
        # Normalize to probability distribution
        total = sum(letter_scores.values())
        if total > 0:
            letter_probs = {l: score/total for l, score in letter_scores.items()}
        else:
            # Fallback
            remaining = [l for l in self.alphabet if l not in guessed_letters]
            uniform_prob = 1.0 / len(remaining) if remaining else 0
            letter_probs = {l: uniform_prob for l in remaining}
        
        return letter_probs


# ============================================================================
# PART 3: TRAIN HMMs FOR ALL WORD LENGTHS
# ============================================================================

def train_all_hmms(word_groups):
    """Train separate HMM for each word length"""
    
    hmm_models = {}
    
    for length, words in sorted(word_groups.items()):
        if len(words) < 10:  # Skip if too few examples
            print(f"Skipping length {length} (only {len(words)} words)")
            continue
        
        print(f"\nTraining HMM for length {length}...")
        hmm = HangmanHMM(length)
        hmm.train(words)
        hmm_models[length] = hmm
        
        # Show top letters for first position
        first_pos_probs = hmm.emission_probs[0]
        top_letters = sorted(enumerate(first_pos_probs), key=lambda x: x[1], reverse=True)[:5]
        print(f"  Top letters at position 0:", end=" ")
        for idx, prob in top_letters:
            print(f"{hmm.alphabet[idx]}({prob:.3f})", end=" ")
        print()
    
    return hmm_models


# ============================================================================
# PART 4: MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("HANGMAN HMM TRAINING")
    print("="*70)
    
    # Load corpus
    words = load_corpus('corpus.txt')
    
    # Group by length
    word_groups = group_words_by_length(words)
    
    # Train HMMs
    hmm_models = train_all_hmms(word_groups)
    
    # Save models
    with open('hmm_models.pkl', 'wb') as f:
        pickle.dump(hmm_models, f)
    print("\n✓ HMM models saved to 'hmm_models.pkl'")
    
    # ========================================================================
    # PART 5: TESTING AND VISUALIZATION
    # ========================================================================
    
    print("\n" + "="*70)
    print("TESTING HMM PREDICTIONS")
    print("="*70)
    
    # Test on example words
    test_cases = [
        ("APPLE", "_PP__", set(['P'])),
        ("HELLO", "H____", set(['H'])),
        ("WORLD", "_____", set()),
        ("PYTHON", "P____N", set(['P', 'N']))
    ]
    
    for word, masked, guessed in test_cases:
        length = len(word)
        if length in hmm_models:
            hmm = hmm_models[length]
            probs = hmm.predict_letter_probs(masked, guessed)
            
            # Get top 5 predictions
            top_5 = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]
            
            print(f"\nWord: {word}, Masked: {masked}, Guessed: {guessed}")
            print(f"Top 5 predictions:", end=" ")
            for letter, prob in top_5:
                marker = "✓" if letter in word else "✗"
                print(f"{letter}({prob:.3f}){marker}", end=" ")
            print()
    
    # Visualization: Letter frequency by position for 5-letter words
    if 5 in hmm_models:
        print("\n" + "="*70)
        print("VISUALIZATION: 5-Letter Word Patterns")
        print("="*70)
        
        hmm = hmm_models[5]
        
        fig, axes = plt.subplots(1, 5, figsize=(15, 3))
        for pos in range(5):
            probs = hmm.emission_probs[pos]
            top_10_idx = np.argsort(probs)[-10:]
            top_10_letters = [hmm.alphabet[i] for i in top_10_idx]
            top_10_probs = probs[top_10_idx]
            
            axes[pos].barh(top_10_letters, top_10_probs, color='steelblue')
            axes[pos].set_title(f'Position {pos}')
            axes[pos].set_xlabel('Probability')
        
        plt.tight_layout()
        plt.savefig('hmm_letter_distributions.png', dpi=150, bbox_inches='tight')
        print("✓ Saved visualization to 'hmm_letter_distributions.png'")
        plt.show()
    
    print("\n" + "="*70)
    print("HMM TRAINING COMPLETE!")
    print("="*70)
    print(f"Trained {len(hmm_models)} HMM models")
    print("Ready for RL training in next notebook")