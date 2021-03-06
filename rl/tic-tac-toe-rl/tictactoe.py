import numpy as np


class Board:
    X = -1
    O = 1
    NEITHER = 0

    FLIP_PLAYER = -1

    SIZE = 3
    SQUARES = SIZE * SIZE

    CHARS = {
        X: 'X',
        O: 'O',
        NEITHER: '-',
    }

    REVERSE_CHARS = dict([
        (val, key) for (key, val) in CHARS.items()
    ])

    def __init__(self, board=None):
        if board is None:
            board = np.zeros((self.SIZE, self.SIZE))
        self.board = board
        self._winner = None

    def with_square(self, row, col, val):
        board = np.copy(self.board)
        board[row][col] = val
        return self.__class__(board)

    def is_occupied(self, row, col):
        return self.board[row][col] != self.NEITHER

    def _calc_winner(self):
        row_sums = np.sum(self.board, axis=0)
        col_sums = np.sum(self.board, axis=1)
        diag_sums = np.array([
            np.sum(np.diagonal(self.board)),
            np.sum(np.diagonal(np.fliplr(self.board))),
        ])

        x_win = self.X * self.SIZE
        o_win = self.O * self.SIZE

        if (np.any(row_sums == x_win) or np.any(col_sums == x_win) or
                np.any(diag_sums == x_win)):
            return self.X

        if (np.any(row_sums == o_win) or np.any(col_sums == o_win) or
                np.any(diag_sums == o_win)):
            return self.O

        return self.NEITHER

    @property
    def is_draw(self):
        return (not np.any(self.board == self.NEITHER) and
                self.winner == self.NEITHER)

    @property
    def winner(self):
        if self._winner is None:
            self._winner = self._calc_winner()
        return self._winner

    @property
    def array(self):
        return self.board

    @property
    def flipped_players(self):
        return self.__class__(self.board * self.FLIP_PLAYER)

    def __eq__(self, other):
        return np.all(self.board == other.board)

    def __ne__(self, other):
        return np.any(self.board != other.board)

    def __str__(self):
        lines = []
        for row in range(self.SIZE):
            lines.append(''.join([
                self.CHARS[self.board[row][col]]
                for col in range(self.SIZE)
            ]))
        return '\n'.join(lines)

    @classmethod
    def from_string(cls, string):
        string = string.replace('\n', '').replace(' ', '')
        if len(string) != cls.SQUARES:
            raise ValueError(f"invalid length: {len(string)}")
        return cls(np.array([
            cls.REVERSE_CHARS[i] for i in string
        ]).reshape(cls.SIZE, cls.SIZE))
