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

from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn

# Import our Othello engine
from orthello import OthelloGame, play_game, random_agent, greedy_agent, search_agent
from cli_utils import get_formatter
# ──────────────────────────────────────────────────────────────────────
# Hyper-parameters (all overridable via CLI)
# ──────────────────────────────────────────────────────────────────────

POPULATION_SIZE = 60                                  # individuals per generation
ELITE_FRAC      = 0.15                                # fraction kept unchanged each generation
MUTATION_RATE   = 0.12                                # probability of perturbing each weight
MUTATION_STD    = 0.08                                # std of Gaussian noise added on mutation
CROSSOVER_RATE  = 0.70                                # probability child inherits from parent A vs B per weight
EVAL_GAMES      = 6                                   # games played (as Black) vs champion to score fitness
GENERATIONS     = 30                                  # total generations to run
SAVE_DIR        = "ga_models_winmargin_betterscoring" # directory for checkpoints
DEVICE          = torch.device("cpu")                 # switch to "cuda" if available
ARCHIVE_MAX       = 12                                # max number of old champions kept
POOL_ARCHIVE_SAMP = 3                                 # how many archive champs to sample per generation
POOL_PEER_SAMP    = 3                                 # how many random peers each individual faces
MARGIN_WEIGHT     = 0.15                              # how much score margin matters vs win/loss

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


def clone_net(net: OthelloNet) -> OthelloNet:
    """Deep-copy a model safely onto DEVICE."""
    return copy.deepcopy(net).to(DEVICE)


def evaluate_individual(
    net: OthelloNet,
    opponent_agents: List,
    games_per_opponent: int = 2,
) -> float:
    """
    Evaluate one network against a pool of opponents.

    For each opponent:
      - plays games_per_opponent as Black
      - plays games_per_opponent as White

    Returns average fitness across all games:
      result_score + small margin bonus
    """
    agent = net_agent(net)

    total_score = 0.0
    total_games = 0

    for opponent_agent in opponent_agents:
        # Play as Black
        for _ in range(games_per_opponent):
            w, s = play_game(agent, opponent_agent)

            result_score = 1.0 if w == 1 else 0.5 if w == 0 else 0.0
            margin_score = (s[1] - s[-1]) / 64.0   # our score - opp score
            total_score += result_score + MARGIN_WEIGHT * margin_score
            total_games += 1

        # Play as White
        for _ in range(games_per_opponent):
            w, s = play_game(opponent_agent, agent)

            result_score = 1.0 if w == -1 else 0.5 if w == 0 else 0.0
            margin_score = (s[-1] - s[1]) / 64.0   # our score - opp score
            total_score += result_score + MARGIN_WEIGHT * margin_score
            total_games += 1

    return max(0.0, total_score / max(1, total_games))


def evaluate_population(
    population: List[OthelloNet],
    fixed_opponents: List,
    peer_sample_size: int = POOL_PEER_SAMP,
    games_per_opponent: int = 2,
) -> np.ndarray:
    """
    Evaluate each individual against:
      - fixed opponents (champions / baselines / archive)
      - a few random peers from the same generation

    This makes fitness much more robust than single-champion evaluation.
    """
    fitnesses = np.zeros(len(population))

    with tqdm(total=len(population), desc="Evaluating population", unit="individual",
              bar_format='{desc}: {percentage:3.0f}%|{bar:20}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
              colour='cyan') as pbar:
        for i, net in enumerate(population):
            opponents = list(fixed_opponents)

            # Sample a few peers from the same generation (excluding self)
            peer_indices = [j for j in range(len(population)) if j != i]
            if peer_indices:
                sampled = random.sample(peer_indices, min(peer_sample_size, len(peer_indices)))
                for j in sampled:
                    opponents.append(net_agent(population[j]))

            fitnesses[i] = evaluate_individual(
                net,
                opponents,
                games_per_opponent=games_per_opponent,
            )
            pbar.set_postfix({"fitness": f"{fitnesses[i]:.3f}"})
            pbar.update(1)

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
    fmt = get_formatter()

    def log(msg: str):
        print(msg)
        log_lines.append(msg)

    # ── Initialise population ──────────────────────────────────────────
    fmt.header("🧬 Othello Genetic Algorithm Training", width=65)
    
    config_lines = [
        f"Population  : {population_size}",
        f"Generations : {generations}",
        f"Eval games  : {eval_games} per opponent (× 2 sides = {eval_games*2} per individual)",
        f"Elite frac  : {ELITE_FRAC} ({max(1, int(population_size * ELITE_FRAC))} individuals)",
        f"Mutation    : rate={MUTATION_RATE}  std={MUTATION_STD}",
    ]
    
    for line in config_lines:
        log(f"  {line}")
    log("")

    population: List[OthelloNet] = [OthelloNet().to(DEVICE) for _ in range(population_size)]
    champion_archive: List[OthelloNet] = []
    best_ever_agent = random_agent

    champion_net = None     
    #TODO CHECK
    #champion_agent = lambda g: search_agent(g, depth=3)   # gen 0 opponent is the search agent (depth=3) baseline
    champion_agent = random_agent  # gen 0 opponent is the random agent baseline (easier)

    n_elite = max(1, int(population_size * ELITE_FRAC))
    best_ever_fitness = -1.0
    best_ever_net = None

    for gen in range(1, generations + 1):
        t_start = time.time()
        fmt.subheader(f"Generation {gen:03d}/{generations:03d}")

        # ── Build opponent pool ────────────────────────────────────────
        fixed_opponents = [champion_agent, best_ever_agent, greedy_agent]

        if champion_archive:
            sampled_archive = random.sample(
                champion_archive,
                min(POOL_ARCHIVE_SAMP, len(champion_archive))
            )
            fixed_opponents.extend([net_agent(n) for n in sampled_archive])

        # ── Evaluate ───────────────────────────────────────────────────
        fitnesses = evaluate_population(
            population,
            fixed_opponents=fixed_opponents,
            peer_sample_size=POOL_PEER_SAMP,
            games_per_opponent=eval_games,
        )

        ranked_idx = np.argsort(fitnesses)[::-1]    # descending
        best_fitness = fitnesses[ranked_idx[0]]
        mean_fitness = fitnesses.mean()
        worst_fitness = fitnesses[ranked_idx[-1]]

        fmt.stats_line(best=f"{best_fitness:.3f}", mean=f"{mean_fitness:.3f}", worst=f"{worst_fitness:.3f}")

        # ── Save champion ──────────────────────────────────────────────
        champion_net = copy.deepcopy(population[ranked_idx[0]])
        champion_agent = net_agent(champion_net)
        champion_archive.append(clone_net(champion_net))
        if len(champion_archive) > ARCHIVE_MAX:
            champion_archive.pop(0)

        ckpt_path = os.path.join(save_dir, f"best_gen_{gen:03d}.pt")
        torch.save({
            "generation": gen,
            "fitness":    best_fitness,
            "state_dict": champion_net.state_dict(),
        }, ckpt_path)

        if best_fitness > best_ever_fitness:
            best_ever_fitness = best_fitness
            best_ever_net = clone_net(champion_net)
            best_ever_agent = net_agent(best_ever_net)

            torch.save({
                "generation": gen,
                "fitness":    best_ever_fitness,
                "state_dict": best_ever_net.state_dict(),
            }, os.path.join(save_dir, "best_ever.pt"))

            fmt.success(f"New best-ever: {best_ever_fitness:.3f}  (saved best_ever.pt)")
        
        fmt.info(f"Time: {time.time() - t_start:.1f}s")

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

    # ── Final report ───────────────────────────────────────────────────
    fmt.section("🎯 Training Complete")
    fmt.highlight("Best-ever fitness:", f"{best_ever_fitness:.2f}", color="green")
    fmt.highlight("Model saved to:", os.path.join(save_dir, 'best_ever.pt'), color="yellow")

    # Save log
    log_path = os.path.join(save_dir, "training_log.txt")
    with open(log_path, "w") as f:
        f.write("\n".join(log_lines))
    fmt.success(f"Log written to {log_path}")
    print()

    return best_ever_net


# ──────────────────────────────────────────────────────────────────────
# Evaluation 1 model vs random and greedy baselines
# ──────────────────────────────────────────────────────────────────────

def benchmark_model(model_path: str, n_games: int = 200):
    """Load a saved model and benchmark it against random and greedy agents."""
    fmt = get_formatter()
    
    ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)
    net = OthelloNet().to(DEVICE)
    net.load_state_dict(ckpt["state_dict"])
    agent = net_agent(net)

    gen = ckpt.get("generation", "?")
    fitness = ckpt.get("fitness", "?")
    fmt.header(f"📊 Model Benchmark", width=60)
    fmt.highlight("Loaded model:", model_path, color="cyan")
    fmt.highlight("Generation:", f"{gen}", color="yellow")
    fmt.highlight("Fitness:", f"{fitness}", color="green")
    print()

    with tqdm(total=n_games * 3, desc="Playing games", unit="game",
              bar_format='{desc}: {percentage:3.0f}%|{bar:20}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
              colour='green') as pbar:
        for opp_name, opp_agent in [
            ("random", random_agent),
            ("greedy", greedy_agent),
            ("search_depth3", lambda g: search_agent(g, depth=2)),
        ]:  
            results = {1: 0, -1: 0, 0: 0}
            for _ in range(n_games // 2):
                w, s = play_game(agent, opp_agent)       # our model plays Black
                results[w] += 1
                pbar.update(1)
    
            for _ in range(n_games // 2):
                w, s = play_game(opp_agent, agent)       # our model plays White
                # flip perspective
                results[-w] += 1
                pbar.update(1)

            win_rate = (results[1] + 0.5 * results[0]) / n_games * 100
            fmt.subheader(f"vs {opp_name} ({n_games} games)")
            fmt.stats_line(Wins=results[1], Draws=results[0], Losses=results[-1], Win_pct=f"{win_rate:.1f}%")
    
    print()


# ──────────────────────────────────────────────────────────────────────
# Evaluation 2 models
# ──────────────────────────────────────────────────────────────────────

def load_model_agent(model_path: str):
    """Load a saved .pt model and return (net, agent, metadata)."""
    ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)
    net = OthelloNet().to(DEVICE)
    net.load_state_dict(ckpt["state_dict"])
    agent = net_agent(net)

    meta = {
        "generation": ckpt.get("generation", "?"),
        "fitness": ckpt.get("fitness", "?"),
        "path": model_path,
    }
    return net, agent, meta


def evaluate_models(model_a_path: str, model_b_path: str, n_games: int = EVAL_GAMES):
    """
    Evaluate two saved models against each other.
    Plays n_games with A as Black, then n_games with B as Black.
    Total games = 2 * n_games.
    """
    fmt = get_formatter()
    
    _, agent_a, meta_a = load_model_agent(model_a_path)
    _, agent_b, meta_b = load_model_agent(model_b_path)

    fmt.header("⚔️  Model vs Model", width=60)
    fmt.highlight("Model A:", meta_a['path'], color="cyan")
    fmt.highlight("  Gen / Fitness:", f"{meta_a['generation']} / {meta_a['fitness']}", color="yellow")
    fmt.highlight("Model B:", meta_b['path'], color="magenta")
    fmt.highlight("  Gen / Fitness:", f"{meta_b['generation']} / {meta_b['fitness']}", color="yellow")
    print()

    # A as Black
    a_black = {
        "A_wins": 0,
        "B_wins": 0,
        "draws": 0,
        "A_points": 0,
        "B_points": 0,
        "margin_sum": 0,   # A score - B score
    }

    # B as Black
    b_black = {
        "A_wins": 0,
        "B_wins": 0,
        "draws": 0,
        "A_points": 0,
        "B_points": 0,
        "margin_sum": 0,   # A score - B score
    }

    total_games = 2 * n_games

    with tqdm(total=total_games, desc="Playing games", unit="game",
              bar_format='{desc}: {percentage:3.0f}%|{bar:20}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
              colour='blue') as pbar:
        # ── A plays Black ─────────────────────────────────────────────
        for _ in range(n_games):
            w, s = play_game(agent_a, agent_b)

            a_score = s[1]     # Black = A
            b_score = s[-1]    # White = B

            a_black["A_points"] += a_score
            a_black["B_points"] += b_score
            a_black["margin_sum"] += (a_score - b_score)

            if w == 1:
                a_black["A_wins"] += 1
            elif w == -1:
                a_black["B_wins"] += 1
            else:
                a_black["draws"] += 1

            pbar.update(1)

        # ── B plays Black ─────────────────────────────────────────────
        for _ in range(n_games):
            w, s = play_game(agent_b, agent_a)

            b_score = s[1]     # Black = B
            a_score = s[-1]    # White = A

            b_black["A_points"] += a_score
            b_black["B_points"] += b_score
            b_black["margin_sum"] += (a_score - b_score)

            if w == 1:
                b_black["B_wins"] += 1
            elif w == -1:
                b_black["A_wins"] += 1
            else:
                b_black["draws"] += 1

            pbar.update(1)

    # ── Totals ────────────────────────────────────────────────────────
    total_a_wins = a_black["A_wins"] + b_black["A_wins"]
    total_b_wins = a_black["B_wins"] + b_black["B_wins"]
    total_draws  = a_black["draws"]  + b_black["draws"]

    total_a_points = a_black["A_points"] + b_black["A_points"]
    total_b_points = a_black["B_points"] + b_black["B_points"]
    total_margin = a_black["margin_sum"] + b_black["margin_sum"]

    a_score_total = total_a_wins + 0.5 * total_draws
    b_score_total = total_b_wins + 0.5 * total_draws

    # ── Print breakdown ───────────────────────────────────────────────
    fmt = get_formatter()
    fmt.header("🏆 Model vs Model Results", width=62)

    fmt.subheader(f"Model A: {meta_a['path']}")
    fmt.subheader(f"Model B: {meta_b['path']}")
    print()

    fmt.section("When A plays BLACK")
    fmt.stats_line(A_wins=a_black['A_wins'], Draws=a_black['draws'], B_wins=a_black['B_wins'])
    fmt.stats_line(A_avg_points=f"{a_black['A_points'] / n_games:.2f}", 
                   B_avg_points=f"{a_black['B_points'] / n_games:.2f}",
                   A_margin=f"{a_black['margin_sum'] / n_games:+.2f}")

    fmt.section("When B plays BLACK")
    fmt.stats_line(B_wins=b_black['B_wins'], Draws=b_black['draws'], A_wins=b_black['A_wins'])
    fmt.stats_line(B_avg_points=f"{b_black['B_points'] / n_games:.2f}",
                   A_avg_points=f"{b_black['A_points'] / n_games:.2f}",
                   A_margin=f"{b_black['margin_sum'] / n_games:+.2f}")

    fmt.section("By Model Color")
    print(f"  A as Black (W/D/L): {a_black['A_wins']}/{a_black['draws']}/{a_black['B_wins']}")
    print(f"  A as White (W/D/L): {b_black['A_wins']}/{b_black['draws']}/{b_black['B_wins']}")
    print(f"  B as Black (W/D/L): {b_black['B_wins']}/{b_black['draws']}/{b_black['A_wins']}")
    print(f"  B as White (W/D/L): {a_black['B_wins']}/{a_black['draws']}/{a_black['A_wins']}")

    fmt.section("Overall Results")
    fmt.stats_line(Total_games=total_games, A_wins=total_a_wins, Draws=total_draws, B_wins=total_b_wins)
    fmt.stats_line(A_score=f"{a_score_total:.1f}/{total_games} ({100 * a_score_total / total_games:.1f}%)",
                   B_score=f"{b_score_total:.1f}/{total_games} ({100 * b_score_total / total_games:.1f}%)")
    fmt.stats_line(A_avg_points=f"{total_a_points / total_games:.2f}",
                   B_avg_points=f"{total_b_points / total_games:.2f}",
                   A_margin=f"{total_margin / total_games:+.2f}")

    if a_score_total > b_score_total:
        fmt.success("Winner: Model A 🏆")
    elif b_score_total > a_score_total:
        fmt.success("Winner: Model B 🏆")
    else:
        fmt.info("Result: Tie 🤝")
    
    print()

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
                   help="Games per side for fitness evaluation / model-vs-model eval")
    p.add_argument("--save-dir",    type=str, default=SAVE_DIR)

    p.add_argument("--eval",        action="store_true",
                   help="Benchmark a saved model instead of training")
    p.add_argument("--model",       type=str, default=None,
                   help="Path to .pt model file (used with --eval)")
    p.add_argument("--bench-games", type=int, default=200,
                   help="Number of benchmark games (used with --eval)")

    p.add_argument("--eval-models", nargs=2, metavar=("MODEL_A", "MODEL_B"),
                   help="Evaluate two saved models against each other")

    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.eval_models:
        model_a, model_b = args.eval_models
        evaluate_models(model_a, model_b, n_games=args.eval_games)

    elif args.eval:
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