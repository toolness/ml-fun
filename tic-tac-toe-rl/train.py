from pathlib import Path

import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.utils import to_categorical

from tictactoe import Board


MODEL_PATH = Path('model.h5')


def unflatten_move(square_num):
    row = square_num // Board.SIZE
    col = square_num % Board.SIZE
    return (row, col)


def flatten_move(row, col):
    return row * Board.SIZE + col


def make_move(model, board):
    move_probs = model.predict(board.array.reshape(1, Board.SQUARES))[0]
    square_num = np.random.choice(range(Board.SQUARES), p=move_probs)
    return unflatten_move(square_num)


def player_name(turn):
    return f"Player {Board.CHARS[turn]}"


def play_game(model, verbose=False):
    board = Board()
    actions = {
        Board.X: [],
        Board.O: [],
    }
    turn = Board.X
    winner = None
    total_turns = 0
    while True:
        name = player_name(turn)
        player_board = board if turn == Board.X else board.flipped_players
        move = make_move(model, player_board)
        if board.is_occupied(*move):
            if verbose:
                print(f"ACK, {name} chose {move} but that square is "
                      f"occupied.")
            winner = turn * Board.FLIP_PLAYER
            break
        actions[turn].append((
            player_board.array.reshape(Board.SQUARES),
            to_categorical([flatten_move(*move)],
                           num_classes=Board.SQUARES)[0],
        ))
        board = board.with_square(*move, turn)
        if verbose:
            print(f"{name}'s turn:\n{board}\n\n")
        if board.winner != Board.NEITHER:
            assert board.winner == turn
            winner = board.winner
            break
        elif board.is_draw:
            break
        turn = turn * Board.FLIP_PLAYER
        total_turns += 1
    if verbose:
        if winner:
            print(f"{player_name(winner)} wins this game.")
        else:
            print("It's a draw.")

    return total_turns, actions[winner] if winner else None, board


def train_through_play(model, num_games=1000, epochs=10):
    winning_actions = []
    total_turns = 0
    games_forfeited = 0
    games_tied = 0
    for i in range(num_games):
        num_turns, actions, final_board = play_game(model)
        total_turns += num_turns
        if actions is None:
            games_tied += 1
        else:
            winning_actions.extend(actions)
            if final_board.winner == Board.NEITHER:
                games_forfeited += 1

    avg_turns_per_game = total_turns / num_games

    if winning_actions:
        # Apparently zip() is its own inverse if you use *, which is odd.
        boards, moves = zip(*winning_actions)
        boards = np.array(list(boards))
        moves = np.array(list(moves))

        model.fit(boards, moves, epochs=epochs)

    print(f"Avg turns/game: {avg_turns_per_game}  "
          f"games forfeited: {games_forfeited}/{num_games}  "
          f"games tied: {games_tied}/{num_games}")


def get_model():
    if not MODEL_PATH.exists():
        model = Sequential([
            Dense(Board.SQUARES, input_shape=(Board.SQUARES,)),
            Activation('relu'),
            Dense(32),
            Activation('relu'),
            Dense(96),
            Activation('relu'),
            Dense(Board.SQUARES),
            Activation('softmax'),
        ])

        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )

        model.save(str(MODEL_PATH))

    return load_model(str(MODEL_PATH))


def main():
    model = get_model()

    i = 0

    while True:
        i += 1
        print(f"Playing/training batch {i}. Press CTRL-C to stop.")
        train_through_play(model)
        print("Saving model...")
        model.save(str(MODEL_PATH))


if __name__ == '__main__':
    main()
