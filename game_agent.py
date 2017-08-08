"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random

directions = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                      (1, -2), (1, 2), (2, -1), (2, 1)]

less_comparator = lambda x, y: True if x < y else False
less_equal_comparator = lambda x, y: True if x <= y else False
INFINITY = float("inf")

class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass

def get_distances(game, player):
    blanks = game.get_blank_spaces()
    distances = [float("inf") for i in range(game.height * game.width)]
    row, col = game.get_player_location(player)
    queue = [(row, col)]
    distances[row + col * game.height] = 0
    while len(queue) > 0:
        row, col = queue.pop(0)
        dist = distances[row + col * game.height]
        for dr, dc in directions:
            next_r = row + dr
            next_c = col + dc
            if 0 <= next_r < game.height and 0 <= next_c < game.width:
                index = next_r + next_c * game.height
                if (next_r, next_c) in blanks:
                    if dist + 1 < distances[index]:
                        distances[index] =  dist + 1
                        queue.append((next_r, next_c))

    return distances

def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_distances = get_distances(game, player)
    opp_distances = get_distances(game, game.get_opponent(player))

    compare = less_comparator
    if game.active_player == player:
        compare = less_equal_comparator

    score = 0
    for i, own_dist in enumerate(own_distances):
        opp_dist = opp_distances[i]
        if compare(own_dist, opp_dist) and own_dist != INFINITY:
            score += 1
    return score


def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_distances = get_distances(game, player)
    opp_distances = get_distances(game, game.get_opponent(player))

    compare = less_comparator
    if game.active_player == player:
        compare = less_equal_comparator

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))

    max_length = 0
    for i, own_dist in enumerate(own_distances):
        opp_dist = opp_distances[i]
        if compare(own_dist, opp_dist) and own_dist > max_length:
            max_length = own_dist
    return own_moves + max_length


def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_distances = get_distances(game, player)
    opp_distances = get_distances(game, game.get_opponent(player))

    compare = less_comparator
    if game.active_player == player:
        compare = less_equal_comparator

    score = 0
    for i, own_dist in enumerate(own_distances):
        opp_dist = opp_distances[i]
        if compare(own_dist, opp_dist):
            score += 1
        else:
            score -= 1
    return score


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        legal_moves = game.get_legal_moves()
        best_move =  random.choice(legal_moves) if legal_moves  else (-1, -1) # Best move before search

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # TODO: finish this function!
        legal_moves = game.get_legal_moves()
        best_move =  random.choice(legal_moves) if legal_moves  else (-1, -1) # Best move before search
        best_value = float("-inf") # Best score before search
        # evaluate each move in legal moves
        for move in legal_moves:
            # Get the maximum posible score playing the current move
            value = self._min_value(game.forecast_move(move), depth - 1)
            # Check if the current move's score is better than best value
            if (value > best_value):
                # Save best move and score until now
                best_move = move
                best_value = value
        return best_move
    
    def _cutoff_test(self, game, depth):
        """ Check if it is a terminal state or if depth has been reached. """
        if not game.get_legal_moves() or depth <= 0:
            return True
        return False

    def _max_value(self, game, depth):
        """ Returns the maximum posible value from all legal moves """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        # If it's a terminal state or depth has been reached return state' score
        if self._cutoff_test(game, depth):
            return self.score(game, self)
        value = float("-inf")
        # Evaluate each legal move in order to find the maximum score
        for move in game.get_legal_moves():
            value = max(value, self._min_value(game.forecast_move(move), depth - 1))
        return value

    def _min_value(self, game, depth):
        """ Returns the minimum posible value from all legal moves """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        # If it's a terminal state or depth has been reached return state' score
        if self._cutoff_test(game, depth):
            return self.score(game, self)
        value = float("inf")
        # Evaluate each legal move in order to find the minimum score
        for move in game.get_legal_moves():
            value = min(value, self._max_value(game.forecast_move(move), depth - 1))
        return value
        

class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # TODO: finish this function!
        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        legal_moves = game.get_legal_moves()
        best_move =  random.choice(legal_moves) if legal_moves  else (-1, -1) # Best move before search

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            depth = 1
            while(True): # Iterative deepening until timeout rises
                best_move = self.alphabeta(game, depth)
                depth += 1 # One level deeper

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # TODO: finish this function!
        legal_moves = game.get_legal_moves()
        best_move =  random.choice(legal_moves) if legal_moves  else (-1, -1) # Best move before search
        best_value = float("-inf") # Best score before search
        # evaluate each move in legal moves
        for move in legal_moves:
            # Get the maximum posible score playing the current move
            value = self._min_value(game.forecast_move(move), depth -1, alpha, beta)
            # Check if the current move's score is better than best value
            if (value > best_value):
                # Save best move and score until now
                best_move = move
                best_value = value
            # Check if it is possible to prune
            if value >= beta:
                return move
            # Update alpha (lower bound)
            alpha = max(alpha, value)
        return best_move

    def _cutoff_test(self, game, depth):
        """ Check if it is a terminal state or if depth has been reached. """
        if not game.get_legal_moves() or depth <= 0:
            return True
        return False

    def _max_value(self, game, depth, alpha, beta):
        """ Returns the maximum posible value from all legal moves """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        # If it's a terminal state or depth has been reached return state' score
        if self._cutoff_test(game, depth):
            return self.score(game, self)
        value = float("-inf")
        # Evaluate each legal move in order to find the maximum score
        for move in game.get_legal_moves():
            value = max(value, self._min_value(game.forecast_move(move), depth - 1, alpha, beta))
            # Check if it's possible to prune
            if value >= beta:
                return value
            # Update alpha (lower bound)
            alpha = max(alpha, value)
        return value

    def _min_value(self, game, depth, alpha, beta):
        """ Returns the minimum posible value from all legal moves """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        # If it's a terminal state or depth has been reached return state' score
        if self._cutoff_test(game, depth):
            return self.score(game, self)
        value = float("inf")
        # Evaluate each legal move in order to find the minimum score
        for move in game.get_legal_moves():
            value = min(value, self._max_value(game.forecast_move(move), depth - 1, alpha, beta))
            # Check if it's possible to prune
            if value <= alpha:
                return value
            # Update beta (upper bound)
            beta = min(beta, value)
        return value