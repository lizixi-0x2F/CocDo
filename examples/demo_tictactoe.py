"""
Tic-Tac-Toe — CausalPlanner vs MCTS
=====================================

Board layout (node indices):
    0 | 1 | 2
    ---------
    3 | 4 | 5
    ---------
    6 | 7 | 8

Two opponents:

  CausalPlanner — treats move selection as a causal intervention problem.
    Each cell is an SCM node with a 16-dim embedding; edges connect cells
    that share a winning line.  CausalPlanner runs Adam gradient descent
    on the intervention energy to find the best cell — no tree search.

  MCTS — Monte Carlo Tree Search with UCB1 selection.
    With enough rollouts (default 500) it plays near-perfectly:
    it will always win if possible and never miss a forced block.

The match reveals what gradient-based causal planning can and cannot do
against a near-optimal symbolic tree-search opponent.
"""

import sys
import math
import pathlib
import random
import numpy as np
import torch

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from cocdo import NeuralSCM
from cocdo.model import CausalPlanner

# ══════════════════════════════════════════════════════════════════════════════
# 1.  Board helpers
# ══════════════════════════════════════════════════════════════════════════════

CELL_NAMES = [f"c{i}" for i in range(9)]

WINNING_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6),
]


def check_winner(board: tuple[int, ...]) -> int:
    for a, b, c in WINNING_LINES:
        if board[a] == board[b] == board[c] != 0:
            return board[a]
    return 0


def empty_cells(board: tuple[int, ...]) -> list[int]:
    return [i for i, v in enumerate(board) if v == 0]


def is_terminal(board: tuple[int, ...]) -> bool:
    return check_winner(board) != 0 or not empty_cells(board)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  MCTS
# ══════════════════════════════════════════════════════════════════════════════

class MCTSNode:
    __slots__ = ("board", "player", "parent", "move",
                 "children", "wins", "visits", "untried")

    def __init__(self, board, player, parent=None, move=None):
        self.board    = board
        self.player   = player       # whose turn it is at this node
        self.parent   = parent
        self.move     = move         # move that led here
        self.children: list["MCTSNode"] = []
        self.wins     = 0.0
        self.visits   = 0
        self.untried  = empty_cells(board)
        random.shuffle(self.untried)

    def ucb1(self, c: float = 1.414) -> float:
        if self.visits == 0:
            return float("inf")
        return self.wins / self.visits + c * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )

    def best_child(self) -> "MCTSNode":
        return max(self.children, key=lambda n: n.ucb1())

    def expand(self) -> "MCTSNode":
        move = self.untried.pop()
        new_board = list(self.board)
        new_board[move] = self.player
        child = MCTSNode(tuple(new_board), -self.player, parent=self, move=move)
        self.children.append(child)
        return child

    def rollout(self) -> int:
        """Random playout from this node; return winner (1/-1/0)."""
        board = list(self.board)
        player = self.player
        while True:
            w = check_winner(tuple(board))
            if w != 0:
                return w
            free = [i for i, v in enumerate(board) if v == 0]
            if not free:
                return 0
            board[random.choice(free)] = player
            player = -player

    def backpropagate(self, result: int, root_player: int):
        self.visits += 1
        # win from the perspective of the player who owns this node's parent
        if result == root_player:
            self.wins += 1.0
        elif result == 0:
            self.wins += 0.5
        if self.parent:
            self.parent.backpropagate(result, root_player)


def mcts_move(board: tuple[int, ...], player: int,
              n_rollouts: int = 500) -> int:
    """Return the best move for `player` via MCTS."""
    root = MCTSNode(board, player)

    for _ in range(n_rollouts):
        node = root
        # Selection
        while not node.untried and node.children and not is_terminal(node.board):
            node = node.best_child()
        # Expansion
        if node.untried and not is_terminal(node.board):
            node = node.expand()
        # Rollout
        result = node.rollout()
        # Backprop
        node.backpropagate(result, player)

    best = max(root.children, key=lambda n: n.visits)
    return best.move


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Build SCM
# ══════════════════════════════════════════════════════════════════════════════

D_EMBED = 16


def build_tictactoe_scm() -> tuple[NeuralSCM, np.ndarray]:
    N = 9
    A = np.zeros((N, N), dtype=np.float32)
    for line in WINNING_LINES:
        for i, src in enumerate(line):
            for dst in line[i + 1:]:
                A[src, dst] = 0.6

    rng = np.random.RandomState(42)
    E_raw = rng.randn(N, D_EMBED).astype(np.float32)
    norms = np.linalg.norm(E_raw, axis=1, keepdims=True).clip(min=1e-8)
    E = E_raw / norms

    U = E - A.T @ E
    scm = NeuralSCM(var_names=CELL_NAMES, A=A, E=E, U=U,
                    topo_order=CELL_NAMES)
    return scm, A


# ══════════════════════════════════════════════════════════════════════════════
# 4.  CausalPlanner move
# ══════════════════════════════════════════════════════════════════════════════

def board_to_embedding(board, E_base: np.ndarray, player: int) -> np.ndarray:
    E = E_base.copy()
    for i, v in enumerate(board):
        scale = 1.5 if v == player else (0.5 if v == -player else 1.0)
        cur_norm = float(np.linalg.norm(E[i])) or 1.0
        E[i] = E[i] / cur_norm * scale
    return E


def _win_target(board, player: int) -> dict[str, float]:
    targets: dict[str, float] = {}
    for line in WINNING_LINES:
        own   = sum(1 for c in line if board[c] == player)
        opp   = sum(1 for c in line if board[c] == -player)
        empty = [c for c in line if board[c] == 0]
        if opp == 0 and own >= 1 and empty:
            for c in empty:
                targets[CELL_NAMES[c]] = max(
                    targets.get(CELL_NAMES[c], 0.0), 1.5 + own * 0.3
                )
    if not targets:
        for c in [4, 0, 2, 6, 8, 1, 3, 5, 7]:
            if board[c] == 0:
                targets[CELL_NAMES[c]] = 1.5
                break
    return targets


def planner_move(board, scm: NeuralSCM, player: int) -> int:
    free = empty_cells(board)
    if not free:
        return -1
    # Immediate win / forced block
    for c in free:
        b2 = list(board); b2[c] = player
        if check_winner(tuple(b2)) == player:
            return c
    for c in free:
        b2 = list(board); b2[c] = -player
        if check_winner(tuple(b2)) == -player:
            return c
    # Gradient planning
    E_cur  = board_to_embedding(board, scm._E, player)
    target = _win_target(board, player)
    interv_nodes = [CELL_NAMES[c] for c in free]
    planner = CausalPlanner(scm)
    result  = planner.plan(
        E_init       = E_cur,
        target       = target,
        interv_nodes = interv_nodes,
        lr           = 0.1,
        steps        = 80,
        rollout_steps= 2,
        verbose      = False,
    )
    a_opt = result["a_opt"]
    return max(free, key=lambda c: a_opt.get(CELL_NAMES[c], 0.0))


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Game loop
# ══════════════════════════════════════════════════════════════════════════════

SYMBOLS = {1: "X", -1: "O", 0: "."}


def print_board(board):
    for row in range(3):
        print("  " + " | ".join(SYMBOLS[board[row * 3 + col]] for col in range(3)))
        if row < 2:
            print("  ---------")


def play_game(scm: NeuralSCM,
              planner_is_X: bool,
              mcts_rollouts: int = 500,
              verbose: bool = False) -> int:
    """
    Play one game.  Returns 1 (X wins), -1 (O wins), 0 (draw).
    CausalPlanner plays X when planner_is_X=True, else O.
    MCTS plays the other side.
    """
    board: tuple[int, ...] = tuple([0] * 9)
    current = 1   # X goes first

    for _ in range(9):
        use_planner = (current == 1) == planner_is_X

        if use_planner:
            move = planner_move(board, scm, current)
        else:
            move = mcts_move(board, current, n_rollouts=mcts_rollouts)

        if move < 0:
            break
        b = list(board); b[move] = current
        board = tuple(b)

        if verbose:
            name = "Planner" if use_planner else "MCTS   "
            print(f"\n  {name} ({'X' if current==1 else 'O'}) plays cell {move}:")
            print_board(board)

        winner = check_winner(board)
        if winner != 0:
            if verbose:
                label = "Planner" if (winner == 1) == planner_is_X else "MCTS"
                print(f"\n  >> {label} ({'X' if winner==1 else 'O'}) wins!\n")
            return winner
        current = -current

    if verbose:
        print("\n  >> Draw.\n")
    return 0


# ══════════════════════════════════════════════════════════════════════════════
# 6.  Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--games",    type=int, default=50,
                    help="games per matchup (default 50)")
    ap.add_argument("--rollouts", type=int, default=500,
                    help="MCTS rollouts per move (default 500)")
    ap.add_argument("--seed",     type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    N_GAMES   = args.games
    ROLLOUTS  = args.rollouts

    print("=" * 62)
    print(f"  Tic-Tac-Toe  |  CausalPlanner vs MCTS ({ROLLOUTS} rollouts/move)")
    print("=" * 62)

    print("\n[1/3] Building Tic-Tac-Toe SCM ...")
    scm, A = build_tictactoe_scm()
    print(f"  9 nodes, {int((A > 0).sum())} causal edges, {D_EMBED}-dim embeddings")

    # ── Sample game ────────────────────────────────────────────────────────
    print("\n[2/3] Sample game  (Planner=X first, MCTS=O):")
    print("-" * 42)
    play_game(scm, planner_is_X=True, mcts_rollouts=ROLLOUTS, verbose=True)

    # ── Statistics ─────────────────────────────────────────────────────────
    print(f"\n[3/3] Running {N_GAMES}×2 games ...\n")

    def run_series(planner_is_X: bool, label_p: str, label_m: str) -> tuple[int,int,int]:
        wp = wm = dr = 0
        for g in range(N_GAMES):
            r = play_game(scm, planner_is_X=planner_is_X,
                          mcts_rollouts=ROLLOUTS, verbose=False)
            planner_side = 1 if planner_is_X else -1
            if r == planner_side:      wp += 1
            elif r == -planner_side:   wm += 1
            else:                      dr += 1
            if (g + 1) % 10 == 0:
                print(f"  [{label_p} vs {label_m}]  {g+1:3d}/{N_GAMES}"
                      f"  Planner={wp}  Draw={dr}  MCTS={wm}")
        return wp, dr, wm

    print(f"  Matchup A: Planner(X) vs MCTS(O)  — Planner moves first")
    wp_a, dr_a, wm_a = run_series(planner_is_X=True,
                                   label_p="Planner(X)", label_m="MCTS(O)  ")

    print(f"\n  Matchup B: MCTS(X) vs Planner(O)  — MCTS moves first")
    wp_b, dr_b, wm_b = run_series(planner_is_X=False,
                                   label_p="Planner(O)", label_m="MCTS(X)  ")

    total = 2 * N_GAMES
    planner_wins = wp_a + wp_b
    mcts_wins    = wm_a + wm_b
    draws        = dr_a + dr_b

    print(f"\n{'═'*62}")
    print(f"  Overall  ({total} games, MCTS={ROLLOUTS} rollouts/move)")
    print(f"{'═'*62}")
    print(f"  CausalPlanner wins : {planner_wins:3d} / {total}  ({100*planner_wins/total:.1f}%)")
    print(f"  Draws              : {draws:3d} / {total}  ({100*draws/total:.1f}%)")
    print(f"  MCTS wins          : {mcts_wins:3d} / {total}  ({100*mcts_wins/total:.1f}%)")

    print(f"""
  Interpretation
  ──────────────
  MCTS ({ROLLOUTS} rollouts) plays near-optimally for Tic-Tac-Toe.
  Against a perfect opponent every game should end in draw or MCTS win.

  CausalPlanner does NOT search the game tree — it runs Adam gradient
  descent on the causal intervention energy.  It will win/draw when the
  gradient signal from the SCM aligns with the correct move, and lose
  when the board state requires multi-step lookahead the SCM can't model.

  Trade-off:  Planner = O(steps · N · D)  per move  (fast, differentiable)
              MCTS    = O(rollouts · depth) per move  (exact, but discrete)
""")
