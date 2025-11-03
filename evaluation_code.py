"""
3_Evaluation.ipynb - Final Evaluation and Scoring
This notebook evaluates the trained agent on 2000 test games
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns

# Import from previous notebooks (assuming classes are available)
# If running as standalone, copy HangmanEnvironment and HangmanQLearningAgent classes here



class HangmanEnvironment:
    """
    Hangman game environment for RL training
    """
    
    def __init__(self, words, hmm_models, max_lives=6):
        self.words = words
        self.hmm_models = hmm_models
        self.max_lives = max_lives
        self.alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        
        # Current game state
        self.target_word = None
        self.masked_word = None
        self.guessed_letters = None
        self.lives_left = None
        self.game_over = None
        self.won = None
        
    def reset(self):
        """Start a new game with random word"""
        self.target_word = random.choice(self.words).upper()
        self.masked_word = '_' * len(self.target_word)
        self.guessed_letters = set()
        self.lives_left = self.max_lives
        self.game_over = False
        self.won = False
        
        return self._get_state()
    
    def _get_state(self):
        """Get current state representation"""
        # Get HMM predictions
        hmm_probs = self._get_hmm_probs()
        
        state = {
            'masked_word': self.masked_word,
            'guessed_letters': self.guessed_letters.copy(),
            'lives_left': self.lives_left,
            'hmm_probs': hmm_probs,
            'word_length': len(self.target_word),
            'blanks_remaining': self.masked_word.count('_'),
            'num_guessed': len(self.guessed_letters)
        }
        return state
    
    def _get_hmm_probs(self):
        """Get letter probability distribution from HMM"""
        word_length = len(self.target_word)
        
        if word_length in self.hmm_models:
            hmm = self.hmm_models[word_length]
            probs_dict = hmm.predict_letter_probs(self.masked_word, self.guessed_letters)
            
            # Convert to 26-dimensional vector
            probs_vector = np.zeros(26)
            for letter, prob in probs_dict.items():
                idx = ord(letter) - ord('A')
                probs_vector[idx] = prob
        else:
            # Fallback: uniform distribution over unguessed letters
            probs_vector = np.ones(26)
            for letter in self.guessed_letters:
                idx = ord(letter) - ord('A')
                probs_vector[idx] = 0
            
            total = probs_vector.sum()
            if total > 0:
                probs_vector /= total
        
        return probs_vector
    
    def step(self, action):
        """
        Take an action (guess a letter)
        
        Args:
            action: str, letter to guess (A-Z)
            
        Returns:
            next_state, reward, done, info
        """
        
        letter = action.upper()
        
        # Check for repeated guess
        if letter in self.guessed_letters:
            reward = -20  # Heavy penalty for repeated guess
            return self._get_state(), reward, self.game_over, {'repeated': True}
        
        self.guessed_letters.add(letter)
        
        # Check if letter is in word
        if letter in self.target_word:
            # Correct guess - reveal letters
            new_masked = ""
            for i, char in enumerate(self.target_word):
                if char == letter or self.masked_word[i] != '_':
                    new_masked += char
                else:
                    new_masked += '_'
            
            self.masked_word = new_masked
            
            # Check if won
            if '_' not in self.masked_word:
                self.game_over = True
                self.won = True
                reward = 100  # Big reward for winning
            else:
                reward = 10  # Small reward for correct guess
                
        else:
            # Wrong guess
            self.lives_left -= 1
            reward = -15  # Penalty for wrong guess
            
            # Check if lost
            if self.lives_left == 0:
                self.game_over = True
                self.won = False
                reward = -100  # Big penalty for losing
        
        next_state = self._get_state()
        info = {
            'repeated': False,
            'correct': letter in self.target_word,
            'won': self.won
        }
        
        return next_state, reward, self.game_over, info
    
    def get_valid_actions(self):
        """Get list of valid actions (unguessed letters)"""
        return [l for l in self.alphabet if l not in self.guessed_letters]


# ============================================================================
# PART 2: RL AGENT (Q-LEARNING WITH FUNCTION APPROXIMATION)
# ============================================================================

class HangmanQLearningAgent:
    """
    Q-Learning agent with state feature extraction
    Uses approximate Q-learning with feature-based representation
    """
    
    def __init__(self, alpha=0.1, gamma=0.95, epsilon=1.0, epsilon_decay=0.9995, epsilon_min=0.01):
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q-table: dict mapping (state_key, action) -> Q-value
        self.q_table = defaultdict(float)
        
        self.alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    
    def _state_to_key(self, state):
        """Convert state dict to hashable key"""
        # Simplified state representation for Q-table
        masked = state['masked_word']
        guessed = ''.join(sorted(state['guessed_letters']))
        lives = state['lives_left']
        
        return f"{masked}|{guessed}|{lives}"
    
    def get_q_value(self, state, action):
        """Get Q-value for state-action pair"""
        key = self._state_to_key(state)
        return self.q_table[(key, action)]
    
    def get_action(self, state, valid_actions, train=True):
        """
        Choose action using epsilon-greedy policy
        
        Args:
            state: current state dict
            valid_actions: list of valid actions
            train: bool, whether in training mode
        """
        
        if not valid_actions:
            return None
        
        # Epsilon-greedy exploration
        if train and random.random() < self.epsilon:
            # Explore: choose randomly, but weighted by HMM probabilities
            hmm_probs = state['hmm_probs']
            action_probs = []
            for action in valid_actions:
                idx = ord(action) - ord('A')
                action_probs.append(hmm_probs[idx])
            
            # Normalize
            total = sum(action_probs)
            if total > 0:
                action_probs = [p/total for p in action_probs]
                action = np.random.choice(valid_actions, p=action_probs)
            else:
                action = random.choice(valid_actions)
        else:
            # Exploit: choose best action
            # Combine Q-values with HMM probabilities
            best_score = float('-inf')
            best_action = valid_actions[0]
            
            hmm_probs = state['hmm_probs']
            
            for action in valid_actions:
                q_value = self.get_q_value(state, action)
                idx = ord(action) - ord('A')
                hmm_prob = hmm_probs[idx]
                
                # Combined score: weighted sum of Q-value and HMM probability
                score = q_value + 5 * hmm_prob  # Weight HMM predictions
                
                if score > best_score:
                    best_score = score
                    best_action = action
            
            action = best_action
        
        return action
    
    def update(self, state, action, reward, next_state, done, valid_next_actions):
        """Update Q-value using Q-learning update rule"""
        
        current_q = self.get_q_value(state, action)
        
        if done:
            # No future rewards if episode is done
            target_q = reward
        else:
            # Get max Q-value for next state
            if valid_next_actions:
                max_next_q = max([self.get_q_value(next_state, a) for a in valid_next_actions])
            else:
                max_next_q = 0
            
            target_q = reward + self.gamma * max_next_q
        
        # Q-learning update
        new_q = current_q + self.alpha * (target_q - current_q)
        
        key = self._state_to_key(state)
        self.q_table[(key, action)] = new_q
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)




# ============================================================================
# PART 1: LOAD MODELS AND DATA
# ============================================================================

def load_models_and_data():
    """Load all required models and data"""
    
    print("Loading HMM models...")
    with open('hmm_models.pkl', 'rb') as f:
        hmm_models = pickle.load(f)
    print(f"✓ Loaded {len(hmm_models)} HMM models")
    
    print("\nLoading trained RL agent...")
    with open('trained_agent.pkl', 'rb') as f:
        agent = pickle.load(f)
    print(f"✓ Loaded agent with {len(agent.q_table)} Q-table entries")
    
    print("\nLoading test words...")
    with open('corpus.txt', 'r') as f:
        words = [line.strip().upper() for line in f if line.strip()]
    print(f"✓ Loaded {len(words)} words")
    
    return hmm_models, agent, words


# ============================================================================
# PART 2: EVALUATION FUNCTION
# ============================================================================

def evaluate_agent(env, agent, num_games=2000):
    """
    Evaluate agent on test games
    
    Returns:
        dict with detailed metrics
    """
    
    print("="*70)
    print(f"EVALUATING AGENT ON {num_games} GAMES")
    print("="*70)
    
    results = {
        'wins': 0,
        'losses': 0,
        'total_wrong_guesses': 0,
        'total_repeated_guesses': 0,
        'game_details': []
    }
    
    for game_num in range(num_games):
        state = env.reset()
        
        game_info = {
            'word': env.target_word,
            'word_length': len(env.target_word),
            'wrong_guesses': 0,
            'repeated_guesses': 0,
            'total_guesses': 0,
            'won': False,
            'guessed_letters': []
        }
        
        while not env.game_over:
            valid_actions = env.get_valid_actions()
            
            if not valid_actions:
                break
            
            # Choose action (no exploration during evaluation)
            action = agent.get_action(state, valid_actions, train=False)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            game_info['guessed_letters'].append(action)
            game_info['total_guesses'] += 1
            
            if info.get('repeated', False):
                game_info['repeated_guesses'] += 1
            elif not info.get('correct', False):
                game_info['wrong_guesses'] += 1
            
            state = next_state
        
        # Record game outcome
        game_info['won'] = env.won
        
        if env.won:
            results['wins'] += 1
        else:
            results['losses'] += 1
        
        results['total_wrong_guesses'] += game_info['wrong_guesses']
        results['total_repeated_guesses'] += game_info['repeated_guesses']
        results['game_details'].append(game_info)
        
        # Print progress
        if (game_num + 1) % 200 == 0:
            current_success_rate = results['wins'] / (game_num + 1)
            avg_wrong = results['total_wrong_guesses'] / (game_num + 1)
            avg_repeated = results['total_repeated_guesses'] / (game_num + 1)
            
            print(f"\nProgress: {game_num + 1}/{num_games}")
            print(f"  Success Rate: {current_success_rate:.2%}")
            print(f"  Avg Wrong Guesses: {avg_wrong:.2f}")
            print(f"  Avg Repeated Guesses: {avg_repeated:.2f}")
    
    return results


# ============================================================================
# PART 3: SCORING CALCULATION
# ============================================================================

def calculate_final_score(results, num_games):
    """
    Calculate final score using the given formula:
    Final Score = (Success Rate * 2000) - (Total Wrong Guesses * 5) - (Total Repeated Guesses * 2)
    """
    
    success_rate = results['wins'] / num_games
    total_wrong = results['total_wrong_guesses']
    total_repeated = results['total_repeated_guesses']
    
    final_score = (success_rate * 2000) - (total_wrong * 5) - (total_repeated * 2)
    
    print("\n" + "="*70)
    print("FINAL SCORING")
    print("="*70)
    print(f"Total Games: {num_games}")
    print(f"Wins: {results['wins']}")
    print(f"Losses: {results['losses']}")
    print(f"Success Rate: {success_rate:.4f} ({success_rate*100:.2f}%)")
    print(f"\nTotal Wrong Guesses: {total_wrong}")
    print(f"Avg Wrong Guesses per Game: {total_wrong/num_games:.2f}")
    print(f"\nTotal Repeated Guesses: {total_repeated}")
    print(f"Avg Repeated Guesses per Game: {total_repeated/num_games:.2f}")
    print("\n" + "-"*70)
    print("SCORE CALCULATION:")
    print(f"  (Success Rate × 2000) = {success_rate:.4f} × 2000 = {success_rate * 2000:.2f}")
    print(f"  (Wrong Guesses × 5) = {total_wrong} × 5 = {total_wrong * 5:.2f}")
    print(f"  (Repeated Guesses × 2) = {total_repeated} × 2 = {total_repeated * 2:.2f}")
    print("-"*70)
    print(f"FINAL SCORE: {final_score:.2f}")
    print("="*70)
    
    return {
        'final_score': final_score,
        'success_rate': success_rate,
        'total_wrong_guesses': total_wrong,
        'avg_wrong_guesses': total_wrong / num_games,
        'total_repeated_guesses': total_repeated,
        'avg_repeated_guesses': total_repeated / num_games
    }


# ============================================================================
# PART 4: DETAILED ANALYSIS
# ============================================================================

def analyze_results(results):
    """Perform detailed analysis of results"""
    
    print("\n" + "="*70)
    print("DETAILED ANALYSIS")
    print("="*70)
    
    # Analysis by word length
    length_stats = defaultdict(lambda: {'wins': 0, 'games': 0, 'wrong': 0})
    
    for game in results['game_details']:
        length = game['word_length']
        length_stats[length]['games'] += 1
        length_stats[length]['wrong'] += game['wrong_guesses']
        if game['won']:
            length_stats[length]['wins'] += 1
    
    print("\nPerformance by Word Length:")
    print("-" * 70)
    print(f"{'Length':<8} {'Games':<8} {'Wins':<8} {'Success Rate':<15} {'Avg Wrong':<12}")
    print("-" * 70)
    
    for length in sorted(length_stats.keys()):
        stats = length_stats[length]
        success_rate = stats['wins'] / stats['games'] if stats['games'] > 0 else 0
        avg_wrong = stats['wrong'] / stats['games'] if stats['games'] > 0 else 0
        print(f"{length:<8} {stats['games']:<8} {stats['wins']:<8} {success_rate*100:>6.2f}%        {avg_wrong:>6.2f}")
    
    # Find hardest words
    print("\n" + "="*70)
    print("MOST CHALLENGING WORDS (Lost Games):")
    print("="*70)
    
    lost_games = [g for g in results['game_details'] if not g['won']]
    if lost_games:
        # Sort by wrong guesses
        hardest = sorted(lost_games, key=lambda x: x['wrong_guesses'], reverse=True)[:10]
        
        for i, game in enumerate(hardest, 1):
            print(f"{i}. {game['word']} - {game['wrong_guesses']} wrong guesses")
            print(f"   Guessed: {', '.join(game['guessed_letters'][:10])}")
    
    # Most efficient wins
    print("\n" + "="*70)
    print("MOST EFFICIENT WINS:")
    print("="*70)
    
    won_games = [g for g in results['game_details'] if g['won']]
    if won_games:
        best = sorted(won_games, key=lambda x: x['wrong_guesses'])[:10]
        
        for i, game in enumerate(best, 1):
            print(f"{i}. {game['word']} - {game['wrong_guesses']} wrong guesses, {game['total_guesses']} total")
    
    return length_stats


# ============================================================================
# PART 5: VISUALIZATIONS
# ============================================================================

def create_visualizations(results, length_stats):
    """Create comprehensive visualizations"""
    
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    fig = plt.figure(figsize=(16, 12))
    
    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Plot 1: Success Rate by Word Length
    ax1 = fig.add_subplot(gs[0, 0])
    lengths = sorted(length_stats.keys())
    success_rates = [length_stats[l]['wins']/length_stats[l]['games'] for l in lengths]
    ax1.bar(lengths, success_rates, color='steelblue', alpha=0.7)
    ax1.set_xlabel('Word Length')
    ax1.set_ylabel('Success Rate')
    ax1.set_title('Success Rate by Word Length')
    ax1.set_ylim([0, 1])
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Average Wrong Guesses by Length
    ax2 = fig.add_subplot(gs[0, 1])
    avg_wrong = [length_stats[l]['wrong']/length_stats[l]['games'] for l in lengths]
    ax2.bar(lengths, avg_wrong, color='coral', alpha=0.7)
    ax2.set_xlabel('Word Length')
    ax2.set_ylabel('Avg Wrong Guesses')
    ax2.set_title('Average Wrong Guesses by Word Length')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Distribution of Wrong Guesses
    ax3 = fig.add_subplot(gs[0, 2])
    wrong_counts = [g['wrong_guesses'] for g in results['game_details']]
    ax3.hist(wrong_counts, bins=range(0, max(wrong_counts)+2), color='orange', alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Wrong Guesses per Game')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Distribution of Wrong Guesses')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Win/Loss Pie Chart
    ax4 = fig.add_subplot(gs[1, 0])
    sizes = [results['wins'], results['losses']]
    labels = [f"Wins\n({results['wins']})", f"Losses\n({results['losses']})"]
    colors = ['#90EE90', '#FFB6C6']
    ax4.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax4.set_title('Overall Win/Loss Distribution')
    
    # Plot 5: Total Guesses Distribution
    ax5 = fig.add_subplot(gs[1, 1])
    total_guesses = [g['total_guesses'] for g in results['game_details']]
    ax5.hist(total_guesses, bins=30, color='purple', alpha=0.7, edgecolor='black')
    ax5.set_xlabel('Total Guesses per Game')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Distribution of Total Guesses')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Plot 6: Repeated Guesses Distribution
    ax6 = fig.add_subplot(gs[1, 2])
    repeated = [g['repeated_guesses'] for g in results['game_details']]
    unique_repeated = sorted(set(repeated))
    repeated_counts = [repeated.count(r) for r in unique_repeated]
    ax6.bar(unique_repeated, repeated_counts, color='red', alpha=0.7)
    ax6.set_xlabel('Repeated Guesses per Game')
    ax6.set_ylabel('Frequency')
    ax6.set_title('Distribution of Repeated Guesses')
    ax6.grid(True, alpha=0.3, axis='y')
    
    # Plot 7: Heatmap of Performance (Length vs Outcome)
    ax7 = fig.add_subplot(gs[2, :])
    
    # Create matrix for heatmap
    length_range = range(min(lengths), max(lengths)+1)
    outcome_matrix = np.zeros((2, len(length_range)))  # 2 rows: wins, losses
    
    for i, length in enumerate(length_range):
        if length in length_stats:
            outcome_matrix[0, i] = length_stats[length]['wins']
            outcome_matrix[1, i] = length_stats[length]['games'] - length_stats[length]['wins']
    
    sns.heatmap(outcome_matrix, annot=True, fmt='.0f', cmap='RdYlGn', 
                xticklabels=length_range, yticklabels=['Wins', 'Losses'],
                cbar_kws={'label': 'Count'}, ax=ax7)
    ax7.set_xlabel('Word Length')
    ax7.set_title('Performance Heatmap: Wins vs Losses by Word Length')
    
    plt.savefig('evaluation_results.png', dpi=150, bbox_inches='tight')
    print("✓ Saved evaluation visualizations to 'evaluation_results.png'")
    plt.show()


# ============================================================================
# PART 6: SAVE RESULTS
# ============================================================================

def save_results_to_file(results, score_info, length_stats):
    """Save detailed results to text file"""
    
    with open('evaluation_results.txt', 'w') as f:
        f.write("="*70 + "\n")
        f.write("HANGMAN AGENT EVALUATION RESULTS\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Total Games: {len(results['game_details'])}\n")
        f.write(f"Wins: {results['wins']}\n")
        f.write(f"Losses: {results['losses']}\n")
        f.write(f"Success Rate: {score_info['success_rate']:.4f} ({score_info['success_rate']*100:.2f}%)\n\n")
        
        f.write(f"Total Wrong Guesses: {score_info['total_wrong_guesses']}\n")
        f.write(f"Average Wrong Guesses: {score_info['avg_wrong_guesses']:.2f}\n\n")
        
        f.write(f"Total Repeated Guesses: {score_info['total_repeated_guesses']}\n")
        f.write(f"Average Repeated Guesses: {score_info['avg_repeated_guesses']:.2f}\n\n")
        
        f.write("="*70 + "\n")
        f.write(f"FINAL SCORE: {score_info['final_score']:.2f}\n")
        f.write("="*70 + "\n\n")
        
        f.write("Performance by Word Length:\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Length':<8} {'Games':<8} {'Wins':<8} {'Success Rate':<15} {'Avg Wrong':<12}\n")
        f.write("-" * 70 + "\n")
        
        for length in sorted(length_stats.keys()):
            stats = length_stats[length]
            success_rate = stats['wins'] / stats['games']
            avg_wrong = stats['wrong'] / stats['games']
            f.write(f"{length:<8} {stats['games']:<8} {stats['wins']:<8} {success_rate*100:>6.2f}%        {avg_wrong:>6.2f}\n")
    
    print("✓ Saved detailed results to 'evaluation_results.txt'")


# ============================================================================
# PART 7: MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    # Load models and data
    hmm_models, agent, words = load_models_and_data()
    
    # Create environment
    from_training = HangmanEnvironment  # Assumes class from training notebook
    env = HangmanEnvironment(words, hmm_models, max_lives=6)
    
    # Run evaluation
    NUM_TEST_GAMES = 2000
    results = evaluate_agent(env, agent, num_games=NUM_TEST_GAMES)
    
    # Calculate final score
    score_info = calculate_final_score(results, NUM_TEST_GAMES)
    
    # Detailed analysis
    length_stats = analyze_results(results)
    
    # Create visualizations
    create_visualizations(results, length_stats)
    
    # Save results
    save_results_to_file(results, score_info, length_stats)
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE!")
    print("="*70)
    print(f"\n🎯 FINAL SCORE: {score_info['final_score']:.2f}")
    print(f"✓ Success Rate: {score_info['success_rate']*100:.2f}%")
    print(f"✓ Avg Wrong Guesses: {score_info['avg_wrong_guesses']:.2f}")
    print(f"✓ Avg Repeated Guesses: {score_info['avg_repeated_guesses']:.2f}")
    print("\nAll results saved to:")
    print("  - evaluation_results.png")
    print("  - evaluation_results.txt")
    print("\n" + "="*70)
