"""
othello.py — A complete Othello (Reversi) engine.

Board representation:
    0  = empty
    1  = Black (the player who moves first)
   -1  = White

Public API
----------
OthelloGame
    .board          np.ndarray (8, 8)
    .current_player int  (1 or -1)
    .get_valid_moves() -> list[(r, c)]
    .make_move(r, c) -> bool
    .is_game_over()  -> bool
    .get_winner()    -> int  (1, -1, or 0 for draw)
    .get_score()     -> dict {1: int, -1: int}
    .copy()          -> OthelloGame
    .render()        (pretty-prints the board)

Standalone play
---------------
    python othello.py          # human vs human
    python othello.py --demo   # auto-plays random moves
"""

import numpy as np
import argparse
import random
import sys


DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1),
              ( 0, -1),           ( 0, 1),
              ( 1, -1), ( 1, 0), ( 1, 1)]

BOARD_SIZE = 8


class OthelloGame:
    """Full Othello game state and logic."""

    def __init__(self):
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        # Standard starting position
        mid = BOARD_SIZE // 2
        self.board[mid - 1][mid - 1] =  1   # Black
        self.board[mid - 1][mid]     = -1   # White
        self.board[mid][mid - 1]     = -1   # White
        self.board[mid][mid]         =  1   # Black
        self.current_player = 1             # Black moves first
        self._pass_count = 0               # consecutive passes

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------

    def _in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE

    def _flips_in_direction(self, r: int, c: int, dr: int, dc: int, player: int):
        """Return list of (row, col) squares that would be flipped in one direction."""
        opponent = -player
        flips = []
        nr, nc = r + dr, c + dc
        while self._in_bounds(nr, nc) and self.board[nr][nc] == opponent:
            flips.append((nr, nc))
            nr += dr
            nc += dc
        # The run of opponent pieces must be terminated by one of our own pieces
        if flips and self._in_bounds(nr, nc) and self.board[nr][nc] == player:
            return flips
        return []

    def _get_flips(self, r: int, c: int, player: int):
        """All squares flipped if `player` plays at (r, c). Empty list = illegal."""
        if self.board[r][c] != 0:
            return []
        all_flips = []
        for dr, dc in DIRECTIONS:
            all_flips.extend(self._flips_in_direction(r, c, dr, dc, player))
        return all_flips

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_valid_moves(self, player: int = None):
        """Return list of (row, col) legal moves for *player* (defaults to current player)."""
        if player is None:
            player = self.current_player
        moves = []
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if self._get_flips(r, c, player):
                    moves.append((r, c))
        return moves

    def make_move(self, r: int, c: int) -> bool:
        """
        Place a disc at (r, c) for the current player.
        Returns True on success, False if the move is illegal.
        A special "pass" move is encoded as (-1, -1).
        """
        if (r, c) == (-1, -1):
            # Pass — only legal when the current player has no moves
            valid = self.get_valid_moves()
            if valid:
                return False          # cannot pass when moves exist
            self._pass_count += 1
            self.current_player = -self.current_player
            return True

        flips = self._get_flips(r, c, self.current_player)
        if not flips:
            return False

        self.board[r][c] = self.current_player
        for fr, fc in flips:
            self.board[fr][fc] = self.current_player

        self._pass_count = 0
        self.current_player = -self.current_player

        # If the new current player has no moves, pass automatically for them
        if not self.get_valid_moves():
            self._pass_count += 1
            self.current_player = -self.current_player

        return True

    def is_game_over(self) -> bool:
        """Game ends when neither player can move, or the board is full."""
        if np.count_nonzero(self.board == 0) == 0:
            return True
        if self._pass_count >= 2:
            return True
        # Both players have no moves
        if not self.get_valid_moves(1) and not self.get_valid_moves(-1):
            return True
        return False

    def get_score(self) -> dict:
        black = int(np.sum(self.board == 1))
        white = int(np.sum(self.board == -1))
        return {1: black, -1: white}

    def get_winner(self) -> int:
        """Returns 1 (Black), -1 (White), or 0 (draw). Meaningful after game over."""
        s = self.get_score()
        if s[1] > s[-1]:
            return 1
        elif s[-1] > s[1]:
            return -1
        return 0

    def copy(self) -> "OthelloGame":
        g = OthelloGame.__new__(OthelloGame)
        g.board = self.board.copy()
        g.current_player = self.current_player
        g._pass_count = self._pass_count
        return g

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    SYMBOLS = {0: "·", 1: "●", -1: "○"}

    def render(self, highlight: list = None):
        highlight = set(highlight or [])
        header = "   " + "  ".join(str(c) for c in range(BOARD_SIZE))
        print(header)
        print("  +" + "---" * BOARD_SIZE + "+")
        for r in range(BOARD_SIZE):
            row_str = f"{r} |"
            for c in range(BOARD_SIZE):
                cell = self.board[r][c]
                sym = self.SYMBOLS[cell]
                if (r, c) in highlight:
                    sym = "+"     # show valid move
                row_str += f" {sym} "
            row_str += "|"
            print(row_str)
        print("  +" + "---" * BOARD_SIZE + "+")
        sc = self.get_score()
        print(f"  ● Black: {sc[1]}   ○ White: {sc[-1]}")
        player_name = "Black" if self.current_player == 1 else "White"
        print(f"  Turn: {player_name}")


# ------------------------------------------------------------------
# Helper: play one full game between two callables
# ------------------------------------------------------------------

def play_game(black_fn, white_fn, verbose=False):
    """
    Play a full game.

    black_fn(game) -> (r, c) or (-1,-1)
    white_fn(game) -> (r, c) or (-1,-1)

    Returns winner: 1, -1, or 0.
    """
    game = OthelloGame()
    agents = {1: black_fn, -1: white_fn}

    while not game.is_game_over():
        valid = game.get_valid_moves()
        if not valid:
            game.make_move(-1, -1)
            continue

        move = agents[game.current_player](game)
        if move not in valid:
            move = random.choice(valid)    # fallback for illegal suggestions

        if verbose:
            game.render(highlight=valid)
            print(f"  → Move: {move}\n")

        game.make_move(*move)

    if verbose:
        game.render()
        w = game.get_winner()
        print("\nGame over!", {1: "Black wins!", -1: "White wins!", 0: "Draw!"}[w])

    return game.get_winner(), game.get_score(), game.board.copy()


# ------------------------------------------------------------------
# Simple agents (useful for testing / demo)
# ------------------------------------------------------------------

def random_agent(game: OthelloGame):
    moves = game.get_valid_moves()
    return random.choice(moves) if moves else (-1, -1)


def greedy_agent(game: OthelloGame):
    """Pick the move that captures the most discs immediately."""
    moves = game.get_valid_moves()
    if not moves:
        return (-1, -1)
    best, best_move = -1, moves[0]
    for r, c in moves:
        n = len(game._get_flips(r, c, game.current_player))
        if n > best:
            best, best_move = n, (r, c)
    return best_move

def evaluate_board(game: OthelloGame, player: int) -> float:
    """Simple evaluation: piece difference from player's perspective."""
    score = game.get_score()
    return score[player] - score[-player]


def negamax(game: OthelloGame, depth: int, player: int) -> float:
    if depth == 0 or game.is_game_over():
        return evaluate_board(game, player)

    moves = game.get_valid_moves()
    if not moves:
        # pass move
        g2 = game.copy()
        g2.make_move(-1, -1)
        return -negamax(g2, depth - 1, -player)

    best = -float("inf")

    for r, c in moves:
        g2 = game.copy()
        g2.make_move(r, c)
        val = -negamax(g2, depth - 1, -player)
        best = max(best, val)

    return best


def search_agent(game: OthelloGame, depth: int = 3):
    moves = game.get_valid_moves()
    if not moves:
        return (-1, -1)

    best_move = moves[0]
    best_score = -float("inf")

    for r, c in moves:
        g2 = game.copy()
        g2.make_move(r, c)
        score = -negamax(g2, depth - 1, -game.current_player)

        if score > best_score:
            best_score = score
            best_move = (r, c)

    return best_move

def human_agent(game: OthelloGame):
    valid = game.get_valid_moves()
    game.render(highlight=valid)
    print("Valid moves:", valid)
    while True:
        try:
            raw = input("Your move (row col): ").strip()
            r, c = map(int, raw.split())
            if (r, c) in valid:
                return (r, c)
            print("  ✗ Invalid move, try again.")
        except (ValueError, KeyboardInterrupt):
            print("  ✗ Enter two integers separated by a space.")


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Othello game engine")
    parser.add_argument("--demo", action="store_true", help="Auto-play random vs greedy demo")
    parser.add_argument("--games", type=int, default=1, help="Number of demo games")
    args = parser.parse_args()

    if args.demo:
        results = {1: 0, -1: 0, 0: 0}
        for i in range(args.games):
            w = play_game(random_agent, greedy_agent, verbose=(args.games == 1))
            results[w] += 1
        if args.games > 1:
            print(f"\nResults over {args.games} games:")
            print(f"  Black (random) wins: {results[1]}")
            print(f"  White (greedy) wins: {results[-1]}")
            print(f"  Draws:               {results[0]}")
    else:
        # Human vs Human
        play_game(human_agent, human_agent, verbose=False)