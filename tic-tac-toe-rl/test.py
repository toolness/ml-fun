from unittest import TestCase, main

from tictactoe import Board


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

    def test_eq_works(self):
        self.assertEquals(Board(), Board())

    def test_neq_works(self):
        self.assertNotEqual(Board().set(0, 1, Board.X), Board())

    def test_array_works(self):
        b = Board()
        self.assertTrue(b.board is b.array)

    def test_empty_board_works(self):
        self.assertBoardEquals(Board(), [
            '---',
            '---',
            '---'
        ])

    def test_set_works(self):
        self.assertBoardEquals(Board().set(0, 1, Board.X), [
            '-X-',
            '---',
            '---'
        ])

    def test_get_winner_works_with_no_win(self):
        self.assertEqual(Board().get_winner(), Board.NEITHER)

        self.assertEqual(Board.from_string('''
            XX-
            ---
            ---
        ''').get_winner(), Board.NEITHER)

        self.assertEqual(Board.from_string('''
            -OX
            ---
            -O-
        ''').get_winner(), Board.NEITHER)

        self.assertEqual(Board.from_string('''
            O--
            -O-
            --X
        ''').get_winner(), Board.NEITHER)

    def test_get_winner_works_with_row_win(self):
        self.assertEqual(Board.from_string('''
            XXX
            ---
            ---
        ''').get_winner(), Board.X)

    def test_get_winner_works_with_col_win(self):
        self.assertEqual(Board.from_string('''
            -O-
            -O-
            -O-
        ''').get_winner(), Board.O)

    def test_get_winner_works_with_diag_win(self):
        self.assertEqual(Board.from_string('''
            O--
            -O-
            --O
        ''').get_winner(), Board.O)

        self.assertEqual(Board.from_string('''
            --X
            -X-
            X--
        ''').get_winner(), Board.X)

    def test_from_string_works(self):
        lines = [
            'OX-',
            '--X',
            '-X-',
        ]
        self.assertBoardEquals(Board.from_string('\n'.join(lines)), lines)


if __name__ == '__main__':
    main()
