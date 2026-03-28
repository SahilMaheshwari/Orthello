"""
train_ga.py — Genetic Algorithm training for an Othello neural network.

Architecture overview
---------------------
  OthelloNet   : fully-connected policy network
                 input  (65,)  — board (64 cells, each in {-1, 0, 1}) + current_player
                 hidden (128,) → ReLU → (64,) → ReLU → (32,) → ReLU
                 output (64,)  — logit score for each board square

Genetic algorithm
-----------------
  1. Initialise a population of POPULATION_SIZE networks with random weights.
  2. Each generation:
       a. Evaluate fitness: every individual plays EVAL_GAMES games as Black
          *against the best individual from the previous generation* (random for gen 0).
          Fitness = wins + 0.5 × draws.
       b. Elitism: keep the top ELITE_FRAC of the population unchanged.
       c. Crossover: sample two parents proportionally to fitness; produce a child
          by uniform weight crossover.
       d. Mutation: add Gaussian noise (std = MUTATION_STD) to each weight with
          probability MUTATION_RATE.
       e. The new population = elites + children, total POPULATION_SIZE individuals.
  3. After every generation, save the best individual's weights and report stats.

Usage
-----
    python train_ga.py                         # train with defaults
    python train_ga.py --generations 50 --pop 100
    python train_ga.py --eval --model best_gen_020.pt   # evaluate saved model

Requirements
------------
    pip install torch numpy
    (othello.py must be in the same directory)
"""

import argparse
import copy
import os
import random
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn

# Import our Othello engine
from orthello import OthelloGame, play_game, random_agent, greedy_agent

# ──────────────────────────────────────────────────────────────────────
# Hyper-parameters (all overridable via CLI)
# ──────────────────────────────────────────────────────────────────────

POPULATION_SIZE = 60          # individuals per generation
ELITE_FRAC      = 0.15        # fraction kept unchanged each generation
MUTATION_RATE   = 0.12        # probability of perturbing each weight
MUTATION_STD    = 0.08        # std of Gaussian noise added on mutation
CROSSOVER_RATE  = 0.70        # probability child inherits from parent A vs B per weight
EVAL_GAMES      = 6           # games played (as Black) vs champion to score fitness
GENERATIONS     = 30          # total generations to run
SAVE_DIR        = "ga_models_winmargin_betterscoring" # directory for checkpoints
DEVICE          = torch.device("cpu")   # switch to "cuda" if available


# ──────────────────────────────────────────────────────────────────────
# Neural network
# ──────────────────────────────────────────────────────────────────────

class OthelloNet(nn.Module):
    """
    Simple fully-connected policy network for Othello.

    Input  : 65-dim vector
               • cells 0-63: board flattened, from the perspective of the
                 current player (own pieces = +1, opponent = -1, empty = 0)
               • cell 64: current player indicator (+1 Black, -1 White)
    Output : 64-dim logit vector (one per board cell)
             Softmax + masking with valid-move set gives the policy.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(65, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def board_to_tensor(game: OthelloGame) -> torch.Tensor:
    """Convert game state to a 65-dim tensor from the current player's perspective."""
    player = game.current_player
    # Flip board so own pieces are always +1
    board_view = game.board * player
    flat = board_view.flatten().astype(np.float32)
    features = np.append(flat, float(player))
    return torch.tensor(features, dtype=torch.float32, device=DEVICE)


def net_agent(net: OthelloNet):
    """Return an agent function that uses `net` to pick moves."""
    net.eval()

    def agent(game: OthelloGame):
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            return (-1, -1)

        with torch.no_grad():
            x = board_to_tensor(game).unsqueeze(0)   # (1, 65)
            logits = net(x).squeeze(0)               # (64,)

        # Build a mask: -inf for illegal squares
        mask = torch.full((64,), float("-inf"), device=DEVICE)
        for r, c in valid_moves:
            mask[r * 8 + c] = 0.0

        scores = logits + mask
        best_idx = int(scores.argmax())
        return (best_idx // 8, best_idx % 8)

    return agent


# ──────────────────────────────────────────────────────────────────────
# Weight-level GA operations
# ──────────────────────────────────────────────────────────────────────

def get_flat_weights(net: OthelloNet) -> np.ndarray:
    return np.concatenate([p.data.cpu().numpy().ravel() for p in net.parameters()])


def set_flat_weights(net: OthelloNet, flat: np.ndarray):
    offset = 0
    for p in net.parameters():
        numel = p.numel()
        p.data.copy_(
            torch.tensor(flat[offset:offset + numel], dtype=torch.float32)
              .reshape(p.shape)
        )
        offset += numel


def crossover(parent_a: np.ndarray, parent_b: np.ndarray) -> np.ndarray:
    """Uniform crossover: each weight comes from parent A with prob CROSSOVER_RATE."""
    mask = np.random.rand(len(parent_a)) < CROSSOVER_RATE
    return np.where(mask, parent_a, parent_b)


def mutate(weights: np.ndarray) -> np.ndarray:
    """Add Gaussian noise to each weight with probability MUTATION_RATE."""
    noise_mask = np.random.rand(len(weights)) < MUTATION_RATE
    noise = np.random.randn(len(weights)) * MUTATION_STD
    return weights + noise * noise_mask


def make_child(parents_flat: List[np.ndarray], fitnesses: np.ndarray) -> np.ndarray:
    """Tournament-select two parents, crossover, then mutate."""
    # Fitness-proportionate (roulette wheel) selection
    total = fitnesses.sum()
    if total == 0:
        probs = np.ones(len(fitnesses)) / len(fitnesses)
    else:
        probs = fitnesses / total

    idx_a, idx_b = np.random.choice(len(parents_flat), size=2, replace=False, p=probs)
    child = crossover(parents_flat[idx_a], parents_flat[idx_b])
    return mutate(child)


# ──────────────────────────────────────────────────────────────────────
# Fitness evaluation
# ──────────────────────────────────────────────────────────────────────

def evaluate_individual(net: OthelloNet, opponent_agent, n_games: int = EVAL_GAMES) -> float:
    """
    Score one network against `opponent_agent`.
    Plays n_games as Black and n_games as White → 2*n_games total.
    Returns wins + 0.5*draws (max = 2*n_games).
    """
    agent = net_agent(net)
    score = 0.0
    for _ in range(n_games):
        w, s = play_game(agent, opponent_agent)
        if w == 1:
            score += 1.0
        elif w == 0:
            score += 0.5
        score += (s[1] - 32)/64  # margin bonus (normalized score difference)

    for _ in range(n_games):
        w, s = play_game(opponent_agent, agent)
        if w == -1:
            score += 1.0
        elif w == 0:
            score += 0.5
        score += (s[-1] - 32)/64  # margin bonus (normalized score difference)

    return max(0, score)


def evaluate_population(
    population: List[OthelloNet],
    champion_agent,
    n_games: int = EVAL_GAMES,
) -> np.ndarray:
    fitnesses = np.zeros(len(population))
    for i, net in enumerate(population):
        fitnesses[i] = evaluate_individual(net, champion_agent, n_games)
    return fitnesses


# ──────────────────────────────────────────────────────────────────────
# Genetic algorithm main loop
# ──────────────────────────────────────────────────────────────────────

def run_ga(
    generations: int,
    population_size: int,
    eval_games: int,
    save_dir: str,
):
    os.makedirs(save_dir, exist_ok=True)
    log_lines = []

    def log(msg: str):
        print(msg)
        log_lines.append(msg)

    # ── Initialise population ──────────────────────────────────────────
    log(f"\n{'═'*60}")
    log(f"  Othello Genetic Algorithm Training")
    log(f"  Population : {population_size}")
    log(f"  Generations: {generations}")
    log(f"  Eval games : {eval_games}  (× 2 sides  = {eval_games*2} per individual)")
    log(f"  Elite frac : {ELITE_FRAC}")
    log(f"  Mutation   : rate={MUTATION_RATE}  std={MUTATION_STD}")
    log(f"{'═'*60}\n")

    population: List[OthelloNet] = [OthelloNet().to(DEVICE) for _ in range(population_size)]

    champion_net = None      # best individual from previous generation
    champion_agent = random_agent   # gen 0 opponent is the random baseline

    n_elite = max(1, int(population_size * ELITE_FRAC))
    best_ever_fitness = -1.0
    best_ever_net = None

    for gen in range(1, generations + 1):
        t_start = time.time()
        log(f"── Generation {gen:03d} ──────────────────────────────────────")

        # ── Evaluate ───────────────────────────────────────────────────
        fitnesses = evaluate_population(population, champion_agent, eval_games)

        ranked_idx = np.argsort(fitnesses)[::-1]    # descending
        best_fitness = fitnesses[ranked_idx[0]]
        mean_fitness = fitnesses.mean()
        worst_fitness = fitnesses[ranked_idx[-1]]

        log(f"  Fitness  best={best_fitness:.2f}  mean={mean_fitness:.2f}  worst={worst_fitness:.2f}")

        # ── Save champion ──────────────────────────────────────────────
        champion_net = copy.deepcopy(population[ranked_idx[0]])
        champion_agent = net_agent(champion_net)

        ckpt_path = os.path.join(save_dir, f"best_gen_{gen:03d}.pt")
        torch.save({
            "generation": gen,
            "fitness":    best_fitness,
            "state_dict": champion_net.state_dict(),
        }, ckpt_path)

        if best_fitness > best_ever_fitness:
            best_ever_fitness = best_fitness
            best_ever_net = copy.deepcopy(champion_net)
            torch.save({
                "generation": gen,
                "fitness":    best_ever_fitness,
                "state_dict": best_ever_net.state_dict(),
            }, os.path.join(save_dir, "best_ever.pt"))
            log(f"  ★ New best-ever: {best_ever_fitness:.2f}  (saved best_ever.pt)")

        # ── Build next generation ──────────────────────────────────────
        flat_weights = [get_flat_weights(population[i]) for i in ranked_idx]

        # Elites survive unchanged
        new_population: List[OthelloNet] = []
        for i in range(n_elite):
            elite = copy.deepcopy(population[ranked_idx[i]])
            new_population.append(elite)

        # Children fill the rest
        elite_fitnesses = fitnesses[ranked_idx]   # sorted descending
        while len(new_population) < population_size:
            child_flat = make_child(flat_weights, elite_fitnesses)
            child_net = OthelloNet().to(DEVICE)
            set_flat_weights(child_net, child_flat)
            new_population.append(child_net)

        population = new_population
        elapsed = time.time() - t_start
        log(f"  Time: {elapsed:.1f}s\n")

    # ── Final report ───────────────────────────────────────────────────
    log(f"\n{'═'*60}")
    log(f"  Training complete!")
    log(f"  Best-ever fitness: {best_ever_fitness:.2f}")
    log(f"  Model saved to: {os.path.join(save_dir, 'best_ever.pt')}")
    log(f"{'═'*60}")

    # Save log
    log_path = os.path.join(save_dir, "training_log.txt")
    with open(log_path, "w") as f:
        f.write("\n".join(log_lines))
    print(f"\nLog written to {log_path}")

    return best_ever_net


# ──────────────────────────────────────────────────────────────────────
# Evaluation / benchmark mode
# ──────────────────────────────────────────────────────────────────────

def benchmark_model(model_path: str, n_games: int = 200):
    """Load a saved model and benchmark it against random and greedy agents."""
    ckpt = torch.load(model_path, map_location=DEVICE)
    net = OthelloNet().to(DEVICE)
    net.load_state_dict(ckpt["state_dict"])
    agent = net_agent(net)

    gen = ckpt.get("generation", "?")
    fitness = ckpt.get("fitness", "?")
    print(f"\nLoaded model from {model_path}  (gen={gen}, fitness={fitness})")

    for opp_name, opp_agent in [("random", random_agent), ("greedy", greedy_agent)]:
        results = {1: 0, -1: 0, 0: 0}
        for _ in range(n_games // 2):
            w, s = play_game(agent, opp_agent)       # our model plays Black
            results[w] += 1
        for _ in range(n_games // 2):
            w, s = play_game(opp_agent, agent)       # our model plays White
            # flip perspective
            results[-w] += 1

        win_rate = (results[1] + 0.5 * results[0]) / n_games * 100
        print(f"\n  vs {opp_name} ({n_games} games):")
        print(f"    Wins  : {results[1]}")
        print(f"    Draws : {results[0]}")
        print(f"    Losses: {results[-1]}")
        print(f"    Win%  : {win_rate:.1f}%")


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Genetic-algorithm training for an Othello neural network"
    )
    p.add_argument("--generations", type=int, default=GENERATIONS)
    p.add_argument("--pop",         type=int, default=POPULATION_SIZE,
                   help="Population size")
    p.add_argument("--eval-games",  type=int, default=EVAL_GAMES,
                   help="Games vs champion per fitness evaluation (per side)")
    p.add_argument("--save-dir",    type=str, default=SAVE_DIR)
    p.add_argument("--eval",        action="store_true",
                   help="Benchmark a saved model instead of training")
    p.add_argument("--model",       type=str, default=None,
                   help="Path to .pt model file (used with --eval)")
    p.add_argument("--bench-games", type=int, default=200,
                   help="Number of benchmark games (used with --eval)")
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.eval:
        model_path = args.model
        if model_path is None:
            # Try to find best_ever.pt automatically
            default = os.path.join(args.save_dir, "best_ever.pt")
            if os.path.exists(default):
                model_path = default
            else:
                print("No model specified. Use --model <path.pt>")
                exit(1)
        benchmark_model(model_path, n_games=args.bench_games)
    else:
        best_net = run_ga(
            generations=args.generations,
            population_size=args.pop,
            eval_games=args.eval_games,
            save_dir=args.save_dir,
        )