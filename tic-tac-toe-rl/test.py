from unittest import TestCase, main

import numpy as np

from tictactoe import Board
from util import invert_categorical


class UtilTests(TestCase):
    def test_invert_categorical_works(self):
        self.assertEqual(list(invert_categorical(np.array([
            0,
            0,
            1,
        ]))), [0.5, 0.5, 0])


class BoardTests(TestCase):
    def assertBoardEquals(self, board, lines):
        self.assertEqual(str(board).split('\n'), lines)

    def test_flipped_players_works(self):
        self.assertBoardEquals(Board.from_string('''
            XX-
            -O-
            ---
        ''').flipped_players, [
            'OO-',
            '-X-',
            '---',
        ])

    def test_is_draw_works(self):
        self.assertFalse(Board().is_draw)

        self.assertTrue(Board.from_string('''
            XOX
            OOX
            OXO
        ''').is_draw)

        self.assertFalse(Board.from_string('''
            XOX
            OOX
            OXX
        ''').is_draw)

    def test_is_occupied_works(self):
        self.assertFalse(Board().is_occupied(1, 2))
        self.assertTrue(Board().with_square(1, 2, Board.X).is_occupied(1, 2))

    def test_eq_works(self):
        self.assertEquals(Board(), Board())

    def test_neq_works(self):
        self.assertNotEqual(Board().with_square(0, 1, Board.X), Board())

    def test_array_works(self):
        b = Board()
        self.assertTrue(b.board is b.array)

    def test_empty_board_works(self):
        self.assertBoardEquals(Board(), [
            '---',
            '---',
            '---'
        ])

    def test_with_square_works(self):
        self.assertBoardEquals(Board().with_square(0, 1, Board.X), [
            '-X-',
            '---',
            '---'
        ])

    def test_winner_works_with_no_win(self):
        self.assertEqual(Board().winner, Board.NEITHER)

        self.assertEqual(Board.from_string('''
            XX-
            ---
            ---
        ''').winner, Board.NEITHER)

        self.assertEqual(Board.from_string('''
            -OX
            ---
            -O-
        ''').winner, Board.NEITHER)

        self.assertEqual(Board.from_string('''
            O--
            -O-
            --X
        ''').winner, Board.NEITHER)

    def test_winner_works_with_row_win(self):
        self.assertEqual(Board.from_string('''
            XXX
            ---
            ---
        ''').winner, Board.X)

    def test_winner_works_with_col_win(self):
        self.assertEqual(Board.from_string('''
            -O-
            -O-
            -O-
        ''').winner, Board.O)

    def test_winner_works_with_diag_win(self):
        self.assertEqual(Board.from_string('''
            O--
            -O-
            --O
        ''').winner, Board.O)

        self.assertEqual(Board.from_string('''
            --X
            -X-
            X--
        ''').winner, Board.X)

    def test_from_string_works(self):
        lines = [
            'OX-',
            '--X',
            '-X-',
        ]
        self.assertBoardEquals(Board.from_string('\n'.join(lines)), lines)


if __name__ == '__main__':
    main()
