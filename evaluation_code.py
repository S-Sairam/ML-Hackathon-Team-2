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