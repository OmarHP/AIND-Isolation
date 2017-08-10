"""This file is provided as a starting template for writing your own unit
tests to run and debug your minimax and alphabeta agents locally.  The test
cases used by the project assistant are not public.
"""

import unittest

import isolation
import game_agent

from importlib import reload


class IsolationTest(unittest.TestCase):
    """Unit tests for isolation agents"""

    def setUp(self):
        reload(game_agent)
        self.player1 = "Player1"
        self.player2 = "Player2"
        self.game = isolation.Board(self.player1, self.player2)

    def test_getDistances(self):
        reload(game_agent)
        self.player1 = game_agent.AlphaBetaPlayer()
        self.player2 = game_agent.AlphaBetaPlayer()
        self.game = isolation.Board(self.player1, self.player2)
        self.game.apply_move((3,3))
        self.game.apply_move((3,1))
        score = game_agent.custom_score(self.game, self.player1)
        print("Score: {0}".format(score))
        print(self.game.to_string())

    def test_getMaxDepth(self):
        reload(game_agent)
        self.player1 = game_agent.AlphaBetaPlayer()
        self.player2 = game_agent.AlphaBetaPlayer()
        self.game = isolation.Board(self.player1, self.player2)
        self.game.apply_move((3,3))
        self.game.apply_move((2,3))
        self.game.apply_move((1,4))
        self.game.apply_move((3,5))
        self.game.apply_move((2,6))
        self.game.apply_move((4,3))
        self.game.apply_move((0,5))
        self.game.apply_move((3,1))
        self.game.apply_move((2,3))
        self.game.apply_move((5,2))
        self.game.apply_move((4,4))
        self.game.apply_move((6,4))
        self.game.apply_move((3,6))
        self.game.apply_move((4,5))
        self.game.apply_move((5,5))
        self.game.apply_move((5,3))
        self.game.apply_move((3,4))
        self.game.apply_move((4,1))
        self.game.apply_move((1,5))
        self.game.apply_move((2,2))
        self.game.apply_move((0,3))
        self.game.apply_move((1,0))
        self.game.apply_move((2,4))
        self.game.apply_move((0,2))
        self.game.apply_move((1,6))
        self.game.apply_move((2,1))
        self.game.apply_move((0,4))
        self.game.apply_move((4,2))
        self.game.apply_move((2,5))
        self.game.apply_move((5,4))
        score = game_agent.get_max_depth(self.game, self.player1)
        print("Score: {0}".format(score))
        print(self.game.to_string())


if __name__ == '__main__':
    unittest.main()
