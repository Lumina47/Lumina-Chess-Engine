import chess
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import os
import random
import traceback

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class AlphaZeroChess:
    def __init__(self, model_path='C:/Users/kking/Lumina/Lumina/best_model.keras', c_puct=1.0, mcts_simulations=100):
        self.model_path = model_path
        self.c_puct = c_puct
        self.mcts_simulations = mcts_simulations
        self.model = self.load_or_create_model()

        # Piece-square tables for evaluation
        self.piece_square_tables = self.create_piece_square_tables()

    def load_or_create_model(self):
        if os.path.exists(self.model_path):
            model = tf.keras.models.load_model(self.model_path)
        else:
            model = self.create_model()
            self.save_model(model)
        return model

    def create_model(self):
        inputs = layers.Input(shape=(8, 8, 12))
        x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(inputs)
        for _ in range(4):
            x = self.residual_block(x)
        x = layers.Flatten()(x)
        
        # Policy head
        policy = layers.Dense(4096, activation='softmax', name='policy')(x)
        
        # Value head
        value = layers.Dense(1, activation='tanh', name='value')(x)
        
        model = models.Model(inputs=inputs, outputs=[policy, value])
        model.compile(optimizer='adam', loss=['categorical_crossentropy', 'mean_squared_error'])
        return model

    def residual_block(self, x):
        res = layers.Conv2D(256, (3, 3), padding='same')(x)
        res = layers.BatchNormalization()(res)
        res = layers.ReLU()(res)
        res = layers.Conv2D(256, (3, 3), padding='same')(res)
        res = layers.BatchNormalization()(res)
        x = layers.add([x, res])
        x = layers.ReLU()(x)
        return x

    def save_model(self, model):
        model.save(self.model_path)

    def mcts(self, board, training=True, temperature=1.0):
        root_node = Node(board)
        policy, _ = self.evaluate_position(board)
        root_node.expand(policy)

        if training:
            noise = np.random.dirichlet([0.3] * len(root_node.children))
            for i, child in enumerate(root_node.children.values()):
                child.prior = 0.75 * child.prior + 0.25 * noise[i]

        for _ in range(self.mcts_simulations):
            node = root_node
            while not node.is_leaf():
                node = node.select_child(self.c_puct)
            if node.visits > 0:
                policy, value = self.evaluate_position(node.board)
                node.expand(policy)
                if len(node.children) == 0:  # Check for terminal nodes
                    break
                node = random.choice(list(node.children.values())) if training else node
            else:
                _, value = self.evaluate_position(node.board)
            node.backpropagate(value)

        if len(root_node.children) == 0:  # Check if there are no legal moves left
            return None

        if training:
            visits = np.array([child.visits for child in root_node.children.values()])
            visits = visits ** (1 / temperature)
            visits_sum = visits.sum()
            probabilities = visits / visits_sum
            move = random.choices(list(root_node.children.keys()), weights=probabilities, k=1)[0]
        else:
            move = max(root_node.children, key=lambda move: root_node.children[move].visits)

        return move

    def evaluate_position(self, board):
        board_array = self.board_to_array(board)
        board_array = np.expand_dims(board_array, axis=0)
        policy, value = self.model.predict(board_array)
        value = value[0][0]
        
        # Add additional evaluation metrics, including piece-square tables
        additional_value = self.evaluate_additional_metrics(board)
        return policy, value + additional_value

    def evaluate_additional_metrics(self, board):
        piece_values = {
            chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
            chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0
        }
        center_control = 0
        piece_count = 0
        total_value = 0
        piece_square_score = 0
        king_safety = 0
        pawn_structure = 0
        
        for square, piece in board.piece_map().items():
            piece_value = piece_values.get(piece.piece_type, 0)
            if piece.color == chess.WHITE:
                total_value += piece_value
                piece_square_score += self.piece_square_tables[piece.piece_type][square]
            else:
                total_value -= piece_value
                piece_square_score -= self.piece_square_tables[piece.piece_type][chess.square_mirror(square)]
            
            # Center control
            if square in [chess.E4, chess.E5, chess.D4, chess.D5]:
                center_control += 1 if piece.color == chess.WHITE else -1
            
            # King safety: count attackers near the king
            if piece.piece_type == chess.KING:
                king_safety += self.evaluate_king_safety(board, piece.color)
            
            # Pawn structure: consider doubled, isolated, and backward pawns
            if piece.piece_type == chess.PAWN:
                pawn_structure += self.evaluate_pawn_structure(board, square, piece.color)

        # Weigh these factors
        return (0.1 * total_value + 0.05 * center_control +
                0.03 * piece_square_score + 0.05 * king_safety +
                0.02 * pawn_structure)
        
    def evaluate_king_safety(self, board, color):
        # Determine the king's square and set variables based on color
        if color == chess.WHITE:
            king_square = board.king(chess.WHITE)
            enemy_pieces = chess.BLACK
        else:
            king_square = board.king(chess.BLACK)
            enemy_pieces = chess.WHITE
        
        if king_square is None:
            print(f"King square for color {color} not found.")
            return 0

        king_rank = chess.square_rank(king_square)
        king_file = chess.square_file(king_square)
        
        safety_score = 0

        # 1. Pawn Shield
        pawn_shield_score = 0
        shield_offsets = [-1, 0, 1]
        if king_file >= 3:  # King is on the kingside
            for offset in shield_offsets:
                if 0 <= king_file + offset < 8:
                    pawn_rank = king_rank - 1 if color == chess.WHITE else king_rank + 1
                    if 0 <= pawn_rank < 8:  # Ensure pawn rank is valid
                        pawn_square = chess.square(king_file + offset, pawn_rank)

                        if not (0 <= pawn_square < 64):
                            print(f"Invalid pawn_square index: {pawn_square}")
                            continue
                        
                        pawn = board.piece_at(pawn_square)

                        if pawn and pawn.piece_type == chess.PAWN and pawn.color == color:
                            pawn_shield_score += 20  # Add points for each pawn in the shield
                        else:
                            pawn_shield_score -= 10  # Penalty for missing pawns
        safety_score += pawn_shield_score

        # 2. Open Files and Semi-Open Files Near the King
        open_file_penalty = 0
        for offset in shield_offsets:
            file_to_check = king_file + offset
            if 0 <= file_to_check < 8:
                if self.is_open_file(board, file_to_check):
                    open_file_penalty -= 20  # Strong penalty for open file
                elif self.is_semi_open_file(board, file_to_check, color):
                    open_file_penalty -= 10  # Lesser penalty for semi-open file
        safety_score += open_file_penalty

        # 3. Enemy Pieces Near King
        nearby_enemy_penalty = 0
        for offset in range(-2, 3):  # Checking squares near the king (within two ranks and files)
            for file_offset in range(-2, 3):
                if 0 <= king_file + file_offset < 8 and 0 <= king_rank + offset < 8:
                    square_to_check = chess.square(king_file + file_offset, king_rank + offset)
                    #print(f"Checking square near king: {square_to_check}")
                    if not (0 <= square_to_check < 64):
                        #print(f"Invalid square_to_check index: {square_to_check}")
                        continue
                    piece = board.piece_at(square_to_check)
                    if piece and piece.color == enemy_pieces:
                        piece_value = self.get_piece_value(piece.piece_type)
                        nearby_enemy_penalty -= piece_value / 2  # Penalty based on the type of enemy piece near the king
        safety_score += nearby_enemy_penalty

        # 4. King Mobility
        king_mobility = len(list(board.attacks(king_square)))  # How many squares the king can safely move to
        mobility_score = king_mobility * 5  # Reward for king mobility
        safety_score += mobility_score

        # 5. Direct Attacks on King or Adjacent Squares
        direct_attack_penalty = 0
        attackers = list(board.attackers(enemy_pieces, king_square))
        for square in attackers:
            direct_attack_penalty -= 10
        for square in chess.SQUARES:  # Check adjacent squares for attacks
            if chess.square_distance(king_square, square) == 1:
                if board.is_attacked_by(enemy_pieces, square):
                    direct_attack_penalty -= 5
        safety_score += direct_attack_penalty
        
        return safety_score

    def is_open_file(self, board, file):
        for rank in range(8):
            piece = board.piece_at(chess.square(file, rank))
            if piece and piece.piece_type == chess.PAWN:
                return False
        return True

    def is_semi_open_file(self, board, file, color):
        found_pawn = False
        for rank in range(8):
            piece = board.piece_at(chess.square(file, rank))
            if piece and piece.piece_type == chess.PAWN:
                if piece.color == color:
                    return False
                found_pawn = True
        return found_pawn

    def get_piece_value(self, piece_type):
        if piece_type == chess.PAWN:
            return 100
        elif piece_type == chess.KNIGHT or piece_type == chess.BISHOP:
            return 300
        elif piece_type == chess.ROOK:
            return 500
        elif piece_type == chess.QUEEN:
            return 900
        return 0


    def evaluate_pawn_structure(self, board, pawn_square, pawn_color):
        pawn_file = chess.square_file(pawn_square)
        pawn_rank = chess.square_rank(pawn_square)
        
        doubled_pawn = any(
            board.piece_at(chess.square(pawn_file, rank)) == chess.Piece(chess.PAWN, pawn_color)
            for rank in range(8) if rank != pawn_rank
        )
        
        isolated_pawn = not any(
            board.piece_at(chess.square(pawn_file + i, pawn_rank)) == chess.Piece(chess.PAWN, pawn_color)
            for i in [-1, 1] if 0 <= pawn_file + i < 8
        )

        weak_square_penalty = 0
        for offset in [-1, 1]:
            neighbor_file = pawn_file + offset
            if 0 <= neighbor_file < 8:
                for rank in range(8):
                    square = chess.square(neighbor_file, rank)
                    if board.piece_at(square) is None and not board.attackers(pawn_color, square):
                        weak_square_penalty += 0.5

        outpost_bonus = 0
        if isolated_pawn and 3 <= pawn_rank <= 6:
            for offset in [-1, 1]:
                neighbor_file = pawn_file + offset
                if 0 <= neighbor_file < 8:
                    square = chess.square(neighbor_file, pawn_rank + 1)
                    if board.piece_at(square) is None and board.attackers(not pawn_color, square):
                        outpost_bonus += 0.5

        # Base value for a pawn, adjust based on structure
        value = 1
        if doubled_pawn:
            value -= 0.5
        if isolated_pawn:
            value -= 0.3

        return value - weak_square_penalty + outpost_bonus

    def create_piece_square_tables(self):
        return {
            chess.PAWN: np.array([
                0, 0, 0, 0, 0, 0, 0, 0,
                5, 5, 5, 5, 5, 5, 5, 5,
                1, 1, 2, 3, 3, 2, 1, 1,
                0.5, 0.5, 1, 2.5, 2.5, 1, 0.5, 0.5,
                0, 0, 0, 2, 2, 0, 0, 0,
                0.5, -0.5, -1, 0, 0, -1, -0.5, 0.5,
                0.5, 1, 1, -2, -2, 1, 1, 0.5,
                0, 0, 0, 0, 0, 0, 0, 0
            ]),
            chess.KNIGHT: np.array([
                -5, -4, -3, -3, -3, -3, -4, -5,
                -4, -2, 0, 0, 0, 0, -2, -4,
                -3, 0, 1, 1.5, 1.5, 1, 0, -3,
                -3, 0.5, 1.5, 2, 2, 1.5, 0.5, -3,
                -3, 0, 1.5, 2, 2, 1.5, 0, -3,
                -3, 0.5, 1, 1.5, 1.5, 1, 0.5, -3,
                -4, -2, 0, 0.5, 0.5, 0, -2, -4,
                -5, -4, -3, -3, -3, -3, -4, -5
            ]),
            chess.BISHOP: np.array([
                -2, -1, -1, -1, -1, -1, -1, -2,
                -1, 0, 0, 0, 0, 0, 0, -1,
                -1, 0, 0.5, 1, 1, 0.5, 0, -1,
                -1, 0.5, 0.5, 1, 1, 0.5, 0.5, -1,
                -1, 0, 1, 1, 1, 1, 0, -1,
                -1, 1, 1, 1, 1, 1, 1, -1,
                -1, 0.5, 0, 0, 0, 0, 0.5, -1,
                -2, -1, -1, -1, -1, -1, -1, -2
            ]),
            chess.ROOK: np.array([
                0, 0, 0, 0, 0, 0, 0, 0,
                0.5, 1, 1, 1, 1, 1, 1, 0.5,
                -0.5, 0, 0, 0, 0, 0, 0, -0.5,
                -0.5, 0, 0, 0, 0, 0, 0, -0.5,
                -0.5, 0, 0, 0, 0, 0, 0, -0.5,
                -0.5, 0, 0, 0, 0, 0, 0, -0.5,
                -0.5, 0, 0, 0, 0, 0, 0, -0.5,
                0, 0, 0, 0.5, 0.5, 0, 0, 0
            ]),
            chess.QUEEN: np.array([
                -2, -1, -1, -0.5, -0.5, -1, -1, -2,
                -1, 0, 0, 0, 0, 0, 0, -1,
                -1, 0, 0.5, 0.5, 0.5, 0.5, 0, -1,
                -0.5, 0, 0.5, 0.5, 0.5, 0.5, 0, -0.5,
                0, 0, 0.5, 0.5, 0.5, 0.5, 0, -0.5,
                -1, 0.5, 0.5, 0.5, 0.5, 0.5, 0, -1,
                -1, 0, 0.5, 0, 0, 0, 0, -1,
                -2, -1, -1, -0.5, -0.5, -1, -1, -2
            ]),
            chess.KING: np.array([
                -3, -4, -4, -5, -5, -4, -4, -3,
                -3, -4, -4, -5, -5, -4, -4, -3,
                -3, -4, -4, -5, -5, -4, -4, -3,
                -3, -4, -4, -5, -5, -4, -4, -3,
                -2, -3, -3, -4, -4, -3, -3, -2,
                -1, -2, -2, -2, -2, -2, -2, -1,
                2, 2, -1, -1, -1, -1, 2, 2,
                2, 3, 1, 0, 0, 1, 3, 2
            ])
        }

    def board_to_array(self, board):
        board_array = np.zeros((8, 8, 12), dtype=np.int8)
        for square, piece in board.piece_map().items():
            piece_type = piece.piece_type - 1
            color = int(piece.color)
            board_array[chess.square_rank(square), chess.square_file(square), piece_type + 6 * color] = 1
        return board_array
    
    def self_play(self, num_games):
        for game_index in range(num_games):
            try:
                print(f"Starting game {game_index + 1}")
                board = chess.Board()
                game_moves = []
                move_number = 1

                while not board.is_game_over():
                    move = self.mcts(board, training=True, temperature=1.0)
                    if move is None:
                        print("No move returned by MCTS.")
                        break
                    
                    game_moves.append(move)
                    board.push(move)

                    if len(game_moves) % 2 == 0:  # Print after both moves have been played
                        move_pair = game_moves[-2:]  # Last two moves
                        move_str = f"{move_number}. {board.san(move_pair[0])} {board.san(move_pair[1])}"
                        eval_value = self.evaluate_position(board)
                        print(f"{move_str} Eval: {eval_value:.2f}")
                        move_number += 1

                result = board.result()
                print(f"Game {game_index + 1} finished. Result: {result}")
                if result == "1/2-1/2":
                    draw_reason = self.get_draw_reason(board)
                    print(f"Draw reason: {draw_reason}")
                print(f"Total moves: {move_number - 1}\n")

                self.train_on_game(game_moves, result)

            except Exception as e:
                print(f"An error occurred during game {game_index + 1}: {str(e)}")
                # Print more details if available
                import traceback
                traceback.print_exc()
                break

    def get_draw_reason(self, board):
        if board.is_stalemate():
            return "Stalemate"
        elif board.is_insufficient_material():
            return "Insufficient material"
        elif board.is_fivefold_repetition():
            return "Fivefold repetition"
        elif board.can_claim_threefold_repetition():
            return "Threefold repetition"
        elif board.is_seventyfive_moves():
            return "Seventy-five moves rule"
        else:
            return "Unknown reason"

    def train_on_game(self, game_moves, result):
        try:
            X, y_policy, y_value = [], [], []
            outcome = 0
            if result == "1-0":
                outcome = 1
            elif result == "0-1":
                outcome = -1

            for i, (board, move) in enumerate(game_moves):
                board_array = self.board_to_array(board)
                X.append(board_array)
                policy_target = np.zeros(4096)
                move_index = self.move_to_index(move)
                if move_index < 4096:
                    policy_target[move_index] = 1
                y_policy.append(policy_target)

                if i > 0:
                    prev_board, _ = game_moves[i - 1]
                    prev_value = self.evaluate_additional_metrics(prev_board)
                    current_value = self.evaluate_additional_metrics(board)
                    value_change = current_value - prev_value
                else:
                    value_change = 0

                value_target = outcome + value_change
                y_value.append(value_target)

            X = np.array(X)
            y_policy = np.array(y_policy)
            y_value = np.array(y_value)

            print(f"Training on {len(X)} positions from this game.")
            self.model.fit(X, [y_policy, y_value], epochs=1)
        except IndexError as e:
            print(f"IndexError during game training: {str(e)}")
            print(f"Game moves count: {len(game_moves)}")
            print(f"X shape: {np.array(X).shape}")
            print(f"y_policy shape: {np.array(y_policy).shape}")
            print(f"y_value shape: {np.array(y_value).shape}")
        except ValueError as e:
            print(f"ValueError during game training: {str(e)}")
        except Exception as e:
            import traceback
            print(f"An unexpected error occurred during game training: {str(e)}")
            traceback.print_exc()

    def move_to_index(self, move):
        index = move.from_square * 64 + move.to_square
        if index < 0 or index >= 4096:
            raise ValueError(f"Index out of range: {index}")
        return index

    def train_model(self, num_games):
        print(f"Starting training for {num_games} games...")
        self.self_play(num_games)
        print("Training complete. Saving model...")
        self.save_model(self.model)
        print("Model saved successfully.")

    def play_game(self, color):
        board = chess.Board()
        while not board.is_game_over():
            if board.turn == (color == 'white'):
                move = input("Enter your move (in SAN format): ")
                try:
                    board.push_san(move)
                except ValueError:
                    print("Invalid move. Try again.")
                    continue
            else:
                move = self.mcts(board, training=False, temperature=0.0)
                if move is None:
                    print("No legal moves left. The game is over.")
                    break
                print(f"Engine plays: {board.san(move)}")
                board.push(move)
        result = board.result()
        print(f"Game over. Result: {result}")
        if result == "1/2-1/2":
            draw_reason = self.get_draw_reason(board)
            print(f"Draw reason: {draw_reason}")
            
    def print_moves_in_pairs(moves, board):
        move_strs = [move.uci() for move in moves]
        pairs = [move_strs[i:i + 2] for i in range(0, len(move_strs), 2)]
        for i, pair in enumerate(pairs, start=1):
            # Get the evaluation after this pair of moves
            eval = evaluate_position(board)
            eval_str = f"Eval: {eval:.2f}"
            print(f"{i}. {' '.join(pair)} {eval_str}")
class Node:
    def __init__(self, board):
        self.board = board
        self.visits = 0
        self.value_sum = 0
        self.children = {}
        self.prior = 0
        self.parent = None

    def is_leaf(self):
        return len(self.children) == 0

    def expand(self, policy):
        legal_moves = list(self.board.legal_moves)
        policy = np.array(policy).reshape(8, 8, 8, 8)
        for move in legal_moves:
            from_square = move.from_square
            to_square = move.to_square
            prior = policy[chess.square_rank(from_square)][chess.square_file(from_square)][chess.square_rank(to_square)][chess.square_file(to_square)]
            new_board = self.board.copy()
            new_board.push(move)
            child_node = Node(new_board)
            child_node.parent = self
            child_node.prior = prior
            self.children[move] = child_node

    def select_child(self, c_puct):
        total_visits = sum(child.visits for child in self.children.values())
        best_move, best_score = None, -float('inf')
        for move, child in self.children.items():
            uct_score = (child.value_sum / child.visits if child.visits > 0 else 0) + \
                        c_puct * child.prior * (total_visits ** 0.5 / (1 + child.visits))
            if uct_score > best_score:
                best_move, best_score = move, uct_score
        return self.children[best_move]
    
    def backpropagate(self, value):
        self.visits += 1
        self.value_sum += value
        if self.parent:
            self.parent.backpropagate(value)

def main():
    model_path = 'C:/Users/kking/Lumina/Lumina/best_model.keras'
    az_chess = AlphaZeroChess(model_path=model_path)
    
    while True:
        command = input("Enter command (train [number], play [white/black]): ").strip().lower()
        if command.startswith("train"):
            try:
                num_games = int(command.split()[1])
                az_chess.train_model(num_games)
            except (IndexError, ValueError):
                print("Usage: train [number]")
        elif command.startswith("play"):
            try:
                _, color = command.split()
                if color in ["white", "black"]:
                    az_chess.play_game(color)
                else:
                    print("Invalid color. Choose 'white' or 'black'.")
            except IndexError:
                print("Usage: play [white/black]")
        else:
            print("Unknown command. Please use 'train [number]' or 'play [white/black]'.")

if __name__ == "__main__":
    main()
