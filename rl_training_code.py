"""
2_RL_Training.ipynb - Reinforcement Learning Agent for Hangman
This notebook implements and trains the RL agent using Q-learning/DQN
"""

import numpy as np
import random
import pickle
from collections import defaultdict, deque
import matplotlib.pyplot as plt

# ============================================================================
# PART 1: HANGMAN ENVIRONMENT
# ============================================================================

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
# PART 3: TRAINING LOOP
# ============================================================================

def train_agent(env, agent, num_episodes=10000, print_every=500):
    """
    Train the RL agent
    
    Args:
        env: HangmanEnvironment
        agent: HangmanQLearningAgent
        num_episodes: number of training episodes
        print_every: print progress every N episodes
    """
    
    # Metrics tracking
    episode_rewards = []
    episode_lengths = []
    success_rates = []
    wrong_guesses_history = []
    
    print("="*70)
    print("TRAINING RL AGENT")
    print("="*70)
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        steps = 0
        wrong_guesses = 0
        
        while not env.game_over:
            # Get valid actions
            valid_actions = env.get_valid_actions()
            
            if not valid_actions:
                break
            
            # Choose action
            action = agent.get_action(state, valid_actions, train=True)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            if not info.get('correct', False) and not info.get('repeated', False):
                wrong_guesses += 1
            
            # Get next valid actions
            next_valid_actions = env.get_valid_actions()
            
            # Update Q-values
            agent.update(state, action, reward, next_state, done, next_valid_actions)
            
            state = next_state
            episode_reward += reward
            steps += 1
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Track metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        wrong_guesses_history.append(wrong_guesses)
        
        # Calculate recent success rate
        if episode >= 99:
            recent_wins = sum([1 for i in range(episode-99, episode+1) 
                              if episode_rewards[i] > 0])
            success_rate = recent_wins / 100
            success_rates.append(success_rate)
        
        # Print progress
        if (episode + 1) % print_every == 0:
            avg_reward = np.mean(episode_rewards[-print_every:])
            avg_wrong = np.mean(wrong_guesses_history[-print_every:])
            recent_success = success_rates[-1] if success_rates else 0
            
            print(f"Episode {episode+1}/{num_episodes}")
            print(f"  Avg Reward: {avg_reward:.2f}")
            print(f"  Avg Wrong Guesses: {avg_wrong:.2f}")
            print(f"  Success Rate: {recent_success:.2%}")
            print(f"  Epsilon: {agent.epsilon:.4f}")
            print(f"  Q-table size: {len(agent.q_table)}")
            print()
    
    return {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'success_rates': success_rates,
        'wrong_guesses_history': wrong_guesses_history
    }


# ============================================================================
# PART 4: MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    # Load HMM models
    print("Loading HMM models...")
    with open('hmm_models.pkl', 'rb') as f:
        hmm_models = pickle.load(f)
    print(f"✓ Loaded {len(hmm_models)} HMM models")
    
    # Load training words
    print("\nLoading corpus...")
    with open('corpus.txt', 'r') as f:
        words = [line.strip().upper() for line in f if line.strip()]
    print(f"✓ Loaded {len(words)} words")
    
    # Create environment
    print("\nCreating Hangman environment...")
    env = HangmanEnvironment(words, hmm_models, max_lives=6)
    print("✓ Environment ready")
    
    # Create agent
    print("\nInitializing RL agent...")
    agent = HangmanQLearningAgent(
        alpha=0.1,
        gamma=0.95,
        epsilon=1.0,
        epsilon_decay=0.9995,
        epsilon_min=0.01
    )
    print("✓ Agent initialized")
    
    # Train agent
    print("\n" + "="*70)
    training_history = train_agent(env, agent, num_episodes=10000, print_every=1000)
    
    # Save trained agent
    print("\n" + "="*70)
    print("Saving trained agent...")
    with open('trained_agent.pkl', 'wb') as f:
        pickle.dump(agent, f)
    print("✓ Agent saved to 'trained_agent.pkl'")
    
    # ========================================================================
    # PART 5: VISUALIZATION
    # ========================================================================
    
    print("\n" + "="*70)
    print("GENERATING TRAINING VISUALIZATIONS")
    print("="*70)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Episode Rewards
    axes[0, 0].plot(training_history['episode_rewards'], alpha=0.3, color='blue')
    # Smooth with moving average
    window = 100
    if len(training_history['episode_rewards']) >= window:
        smoothed = np.convolve(training_history['episode_rewards'], 
                              np.ones(window)/window, mode='valid')
        axes[0, 0].plot(range(window-1, len(training_history['episode_rewards'])), 
                       smoothed, color='red', linewidth=2, label='Moving Avg (100)')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].set_title('Training Rewards Over Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Success Rate
    if training_history['success_rates']:
        axes[0, 1].plot(training_history['success_rates'], color='green', linewidth=2)
        axes[0, 1].set_xlabel('Episode (after 100)')
        axes[0, 1].set_ylabel('Success Rate')
        axes[0, 1].set_title('Success Rate (100-episode window)')
        axes[0, 1].set_ylim([0, 1])
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Wrong Guesses
    axes[1, 0].plot(training_history['wrong_guesses_history'], alpha=0.3, color='orange')
    if len(training_history['wrong_guesses_history']) >= window:
        smoothed = np.convolve(training_history['wrong_guesses_history'], 
                              np.ones(window)/window, mode='valid')
        axes[1, 0].plot(range(window-1, len(training_history['wrong_guesses_history'])), 
                       smoothed, color='red', linewidth=2, label='Moving Avg (100)')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Wrong Guesses')
    axes[1, 0].set_title('Wrong Guesses Per Episode')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Episode Lengths
    axes[1, 1].hist(training_history['episode_lengths'], bins=50, color='purple', alpha=0.7)
    axes[1, 1].set_xlabel('Episode Length (steps)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Distribution of Episode Lengths')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_progress.png', dpi=150, bbox_inches='tight')
    print("✓ Saved training visualization to 'training_progress.png'")
    plt.show()
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Final epsilon: {agent.epsilon:.4f}")
    print(f"Q-table entries: {len(agent.q_table)}")
    print(f"Final success rate: {training_history['success_rates'][-1]:.2%}" if training_history['success_rates'] else "N/A")
    print("\nReady for evaluation!")