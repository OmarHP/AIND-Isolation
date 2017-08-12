"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random

directions = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                      (1, -2), (1, 2), (2, -1), (2, 1)]

INFINITY = float("inf")

class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def get_max_depth(game, loc):
    """Get the max depth can be reached from location (loc).
    
    Stop the search at max depth of 6 on grounds of efficiency.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    loc : (int, int)
            A coordinate pair (row, column) indicating where to start the search from

    Returns
    -------
    int
        The max depth found 
    """
    # Save the locations that are not reachable or were visited before
    visited = game._board_state[0:game.height * game.width]
    # The search is performed by a depth-first search recursive algorithm
    # 1 is subtracted from result since current location is depth 0
    return _get_max_depth_recursive(game, loc, visited, 0) - 1 

def _get_max_depth_recursive(game, loc, visited, depth):
    """This algorithm is based on a depth-first search algorithm to find the height of a tree, 
       and was modified to be stopped at max depth of 6 (a 7 height)" 
    """
    # Check if max depth has been reached
    if depth == 6:
        return 1
    row, col = loc
    max_depth = 0
    
    # Iterate over each possible move in every direction
    for dr, dc in directions:
        next_r = row + dr
        next_c = col + dc
        # Check if next location is in of bounds
        if 0 <= next_r < game.height and 0 <= next_c < game.width:
            index = next_r + next_c * game.height
            # Check if next location is reachable and has not been visited before
            if visited[index] == 0:
                # Mark next location as visited
                visited[index] = 1
                next_loc = (next_r, next_c)
                # Continue the search one level deeper from current location
                value = _get_max_depth_recursive(game, next_loc, visited, depth + 1)
                # Pick the max depth found so far
                max_depth = max(max_depth, value)
                # Mark next location as not visited
                visited[index] = 0
                # Stop search if max depth has been found
                if max_depth + depth == 6:
                    break

    return 1 + max_depth

def get_distances(game, loc):
    """Get distances from location (loc) to every position in board.

    The function is implemented using breadth-first search.
    
    Parameters
    ---------- 
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    loc : (int, int)
            A coordinate pair (row, column) indicating where to start the search from

    Returns
    -------
    list<float>
        The distances from location to every position in board 
    """
    blanks = game.get_blank_spaces()
    # Initialize all distances with max posible distance 
    distances = [float("inf") for i in range(game.height * game.width)]
    row, col = loc
    queue = [(row, col)]
    # Initial location is at 0 distance
    distances[row + col * game.height] = 0
    while len(queue) > 0:
        row, col = queue.pop(0)
        dist = distances[row + col * game.height]
        # Iterate over each possible move in every direction 
        for dr, dc in directions:
            next_r = row + dr
            next_c = col + dc
            # Check if next location is not out of bounds
            if 0 <= next_r < game.height and 0 <= next_c < game.width:
                index = next_r + next_c * game.height
                # Check if next location is available
                if (next_r, next_c) in blanks:
                    #Check if next location has not been found before
                    if dist + 1 < distances[index]:
                        distances[index] =  dist + 1
                        #Continue searching from next location
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

    # Get players location
    own_location = game.get_player_location(player)
    opp_location = game.get_player_location(game.get_opponent(player))

    # Initialize distances to 1, it applies if players has not done 
    # any move so all availables positions are at 1 move distance
    own_distances = [1 if game._board_state[idx] == 0 else float("inf") for idx in range(game.height * game.width)] 
    opp_distances = [1 if game._board_state[idx] == 0 else float("inf") for idx in range(game.height * game.width)]

    # If player has done at least one move, get distances to every position from current player location
    if own_location is not None:
        own_distances = get_distances(game, own_location)

    # If opponent has done at least one move, get distances to every position from current opponent location
    if opp_location is not None:
        opp_distances = get_distances(game, opp_location) 

    score = 0
    # Count how many positions are closer to each player
    for i, own_dist in enumerate(own_distances):
        opp_dist = opp_distances[i]
        if own_dist < opp_dist:
            score += 1
        elif own_dist > opp_dist:
            score -= 1
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
    
    # Get player's legal moves
    own_moves = game.get_legal_moves(player)
    # Get opponent's legal moves
    opp_moves = game.get_legal_moves(game.get_opponent(player))
    
    accum = 0
    # Get max reachable depth from each player's legal move and sum up
    for move in own_moves:
        accum += (get_max_depth(game, move)) + 1

    # Get max reachable depth from each opponents's 
    # legal move and substrac from player' sum
    for move in opp_moves:
        accum -= (get_max_depth(game, move)) + 1
    
    return accum


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

    # Get player's legal moves
    own_moves = game.get_legal_moves(player)
    # Get opponent's legal moves
    opp_moves = game.get_legal_moves(game.get_opponent(player))

    # Sum up max reachable depth from each player's legal move
    own_accum = 0
    for move in own_moves:
        own_accum += get_max_depth(game, move) + 1

    # Sum up max reachable depth from each opponent's legal move
    opp_accum = 0
    for move in opp_moves:
        opp_accum += get_max_depth(game, move) + 1

    # Calculate the max reachable depth mean for each player
    own_mean = own_accum / len (own_moves) if len(own_moves) else 0
    opp_mean = opp_accum / len (opp_moves) if len(opp_moves) else 0

    return own_mean - opp_mean


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
        self.count = 0
        self.mean = 0


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
        depth = 1

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            while True: # Iterative deepening until timeout rises
                best_move = self.alphabeta(game, depth)
                depth += 1 # One level deeper
        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        if self.mean == 0:
            self.mean = depth

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