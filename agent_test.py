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

    def test_get_distances_center(self):

        """
        
        |  |  |  |  |  |  |  |              | 2| 3| 2| 3| 2| 3| 2|
        |  |  |  |  |  |  |  |              | 3| 4| 1| 2| 1| 4| 3|
        |  |  |  |  |  |  |  |              | 2| 1| 2| 3| 2| 1| 2|
        |  |  |  | 1|  |  |  |    =====>    | 3| 2| 3| 0| 3| 2| 3|
        |  |  |  |  |  |  |  |              | 2| 1| 2| 3| 2| 1| 2|
        |  |  |  |  |  |  |  |              | 3| 4| 1| 2| 1| 4| 3|
        |  |  |  |  |  |  |  |              | 2| 3| 2| 3| 2| 3| 2|

              Game State                    Distances to Player 1
        """

        reload(game_agent)
        self.player1 = game_agent.AlphaBetaPlayer()
        self.player2 = game_agent.AlphaBetaPlayer()
        self.game = isolation.Board(self.player1, self.player2)
        self.game.apply_move((3, 3))
        distances = game_agent.get_distances(self.game, self.game.get_player_location(self.player1))
        assert distances == [2, 3, 2, 3, 2, 3, 2,
                             3, 4, 1, 2, 1, 4, 3,
                             2, 1, 2, 3, 2, 1, 2,
                             3, 2, 3, 0, 3, 2, 3,
                             2, 1, 2, 3, 2, 1, 2,
                             3, 4, 1, 2, 1, 4, 3, 
                             2, 3, 2, 3, 2, 3, 2]

    def test_get_distances_case1(self):
        """
        | -| -|  | -| -| -|  |              | -| -| 2| -| -| -| 4|
        |  | -| -| -|  | -| -|              | 1| -| -| -| 3| -| -|
        | -| -| -| -| -|  | -|              | -| -| -| -| -| 5| -|
        | -| 1| -| -| -| -| -|    =====>    | -| 0| -| -| -| -| -|
        |  | -| -| -| -| -|  |              |10| -| -| -| -| -| 6|
        | -| -| -|  | -| -| -|              | -| -| -| 8| -| -| -|
        | -|  | -| -| -|  | -|              | -| 9| -| -| -| 7| -|

             Game State                      Distances to Player 1
        """

        self.player1 = game_agent.AlphaBetaPlayer()
        self.player2 = game_agent.AlphaBetaPlayer()
        self.game = isolation.Board(self.player1, self.player2)
        self.game._board_state[0: self.game.height * self.game.width] = [float("inf") for _ in range(self.game.height * self.game.width)]
        self.game._board_state[1] = 0
        self.game._board_state[4] = 0
        self.game._board_state[10] = 0
        self.game._board_state[13] = 0
        self.game._board_state[14] = 0
        self.game._board_state[26] = 0
        self.game._board_state[29] = 0
        self.game._board_state[37] = 0
        self.game._board_state[41] = 0
        self.game._board_state[42] = 0
        self.game._board_state[46] = 0

        self.game.apply_move((3,1))

        distances = game_agent.get_distances(self.game, self.game.get_player_location(self.player1))
        assert distances[1] == 1
        assert distances[4] == 10
        assert distances[10] == 0
        assert distances[13] == 9
        assert distances[14] == 2
        assert distances[26] == 8
        assert distances[29] == 3
        assert distances[37] == 5
        assert distances[41] == 7
        assert distances[42] == 4
        assert distances[46] == 6

    def test_get_max_depth_case1(self):
        """
        | -| -|  | -| -| -|  |              | -| -| 2| -| -| -| 4|
        |  | -| -| -|  | -| -|              | 1| -| -| -| 3| -| -|
        | -| -| -| -| -|  | -|              | -| -| -| -| -| 5| -|
        | -| 1| -| -| -| -| -|    =====>    | -| 0| -| -| -| -| -|   ===> Max height = 6
        | -| -| -| -| -| -|  |              | -| -| -| -| -| -| 6|
        | -| -| -| -| -| -| -|              | -| -| -| -| -| -| -|
        | -| -| -| -| -|  | -|              | -| -| -| -| -| 7| -|

             Game State                      Max height from Player 1, cut off at 6 
        """

        self.player1 = game_agent.AlphaBetaPlayer()
        self.player2 = game_agent.AlphaBetaPlayer()
        self.game = isolation.Board(self.player1, self.player2)
        self.game._board_state[0: self.game.height * self.game.width] = [float("inf") for _ in range(self.game.height * self.game.width)]
        self.game._board_state[1] = 0 #1
        self.game._board_state[10] = 0 #0
        self.game._board_state[14] = 0 #2
        self.game._board_state[29] = 0 #3
        self.game._board_state[37] = 0 #5
        self.game._board_state[41] = 0 #7
        self.game._board_state[42] = 0 #4
        self.game._board_state[46] = 0 #6

        self.game.apply_move((3,1))

        max_depth = game_agent.get_max_depth(self.game, self.game.get_player_location(self.player1))
        assert max_depth == 6
        

if __name__ == '__main__':
    unittest.main()
