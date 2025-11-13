import heapq

class PuzzleState:
    def __init__(self, board, parent, move, depth, cost):
        self.board = board          # tuple representation for immutability
        self.parent = parent        # parent state
        self.move = move            # move taken to reach this state
        self.depth = depth          # g(n) cost from start
        self.cost = cost            # f(n) = g(n) + h(n)

    def __lt__(self, other):
        # For priority queue, compare by total cost
        return self.cost < other.cost

def manhattan_distance(board, goal):
    distance = 0
    for i in range(9):
        if board[i] != 0:
            x1, y1 = divmod(i, 3)
            x2, y2 = divmod(goal.index(board[i]), 3)
            distance += abs(x1 - x2) + abs(y1 - y2)
    return distance

# Possible moves mapped to index change
moves = {'U': -3, 'D': 3, 'L': -1, 'R': 1}

def is_valid_move(blank_pos, move):
    # Check if moving blank tile is valid by position and move direction
    if move == 'U' and blank_pos < 3:
        return False
    if move == 'D' and blank_pos > 5:
        return False
    if move == 'L' and blank_pos % 3 == 0:
        return False
    if move == 'R' and blank_pos % 3 == 2:
        return False
    return True

def move_tile(board, blank_pos, move):
    new_board = list(board)
    new_blank_pos = blank_pos + moves[move]
    new_board[blank_pos], new_board[new_blank_pos] = new_board[new_blank_pos], new_board[blank_pos]
    return tuple(new_board)

def is_solvable(board):
    # Check solvability based on inversion count (even inversions means solvable)
    inversion_count = 0
    arr = [x for x in board if x != 0]
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if arr[i] > arr[j]:
                inversion_count += 1
    return inversion_count % 2 == 0

def reconstruct_path(state):
    path = []
    while state:
        path.append(state)
        state = state.parent
    return list(reversed(path))

def print_board(board):
    print("+---+---+---+")
    for i in range(0, 9, 3):
        print("|", " | ".join(' ' if x == 0 else str(x) for x in board[i:i+3]), "|")
    print("+---+---+---+")

def a_star(start, goal):
    if not is_solvable(start):
        print("This puzzle configuration is not solvable.")
        return None

    open_list = []
    closed_set = set()

    h = manhattan_distance(start, goal)
    start_state = PuzzleState(start, None, None, 0, h)
    heapq.heappush(open_list, start_state)

    while open_list:
        current = heapq.heappop(open_list)
        if current.board == goal:
            return current

        closed_set.add(current.board)
        blank_pos = current.board.index(0)

        for move in moves:
            if not is_valid_move(blank_pos, move):
                continue

            next_board = move_tile(current.board, blank_pos, move)
            if next_board in closed_set:
                continue

            g = current.depth + 1
            h = manhattan_distance(next_board, goal)
            next_state = PuzzleState(next_board, current, move, g, g + h)
            heapq.heappush(open_list, next_state)

    return None

def print_solution(solution):
    path = reconstruct_path(solution)
    print(f"Solution found in {len(path) - 1} moves:\n")
    for step in path:
        if step.move:
            print(f"Move: {step.move}")
        print_board(step.board)

# Example usage
start = (1, 2, 3,
         4, 0, 5,
         6, 7, 8)

goal = (1, 2, 3,
        4, 5, 6,
        7, 8, 0)

solution = a_star(start, goal)
if solution:
    print_solution(solution)
else:
    print("No solution exists.")
