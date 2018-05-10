"""
Homework 3 - CS640
Dakota Hawkins

This program intelligently plays the "Atropos" game using a static evaluator
combined with the minimax algorithm.

The static evaluator takes a linear combination of the local entropy of colored
nodes and the proportion of empty nodes nearby. 

The run the script simply execute the following command in the terminal

python dyh0110Player.py "<board_state>LastPlay:<last_play>", where the variables
<board_state> and <last_play> fit the guidelines explained in the homework.
"""

import re
import itertools
import numpy as np
from scipy import stats
import sys


# starting_string =  "[13][302][1003][30002][100003][3000002][121212]LastPlay:null"
# starting_string = '[32][103][3322][13233][332222][1322223][30212122][1212121]LastPlay:(2,2,3,3)'
# move=(1, 3, 1, 3)

class AtroposGame(object):
    """
    A class to play Atropos.

    Instance Variables:
        board (dict): Atropos board as dictionary of dictionaries:
                Key: (x, y, z): redundant coordinate for node position on board.
                    where x = height, y = left distance, and z = right distance.
                Value: dictionary
                    (key, string) 'color': (value, string) color of node.
                    (key, string) 'neighbors': (value, list) tuple coordinates
                        of neighboring nodes.
        rows (list): list of lists representing current board state.
        layers (int): height of board.
        size (int): size of the board game.
        last_move (tuple): last move made on the board. Tuple should be follow
            the (c, x, y, z) format of Atropos moves.
        lambda1: weight to apply to local entropy during static evaluation.
            Default is 1.25.
        lambda2: weight to apply to proportion of empty nodes during static
            evaluation. Default is 0.25.
        lambda3: weight to apply to the ratio between local and global entropy.
            Default is 0.75
    """

    def __init__(self, input_string):
        """
        A class to play Atropos

        Arguments:
            input_string (string): a single string following the format outlined
                in the homework description. For example,  
                 "[13][302][1003][30002][100003][3000002][121212]LastPlay:null"
                 where brackets represent rows, integers represent colors, and
                 each position in a bracket represents a game board position.
                 A 4-tuple (c, x, y, z) should follow "LastPlay:" if a move had
                 been made on the board state. 'null' is passed if no move has
                 been made yet.
        """
        self.color_dict = {0: None, 1: 'red', 2: 'blue', 3: 'green',
                           None: 0, 'red': 1, 'blue': 2, 'green': 3}
        self.last_move = None
        self.lambda1 = 1.25
        self.lambda2 = 0.25
        self.lambda3 = 0.75
        self.__parse_input(input_string)
        

    
    def __parse_input(self, input_string):
        """
        Parses input passed into the program.
        
        Arguments:
            input_string (string): a single string following the format outlined
                in the homework description. For example,  
                 "[13][302][1003][30002][100003][3000002][121212]LastPlay:null"
                 where brackets represent rows, integers represent colors, and
                 each position in a bracket represents a game board position.
                 A 4-tuple (c, x, y, z) should follow "LastPlay:" if a move had
                 been made on the board state. 'null' is passed if no move has
                 been made yet.

        Returns:
            None
        """
        if "LastPlay:" not in input_string:
            raise IOError("Incorrect input format. `LastPlay:` not found.")

        split_input = input_string.split("LastPlay:")
        if len(split_input) != 2:
            raise IOError("Incorrect input format. Multiple `LastPlay:` instances found.")

        board_state, last_move = split_input[0], split_input[1]
        self.__init_board(board_state)
        if last_move != 'null':
            last_move = re.findall("\((.*?)\)", last_move)[0].split(',')
            self.last_move = tuple([int(x) for x in last_move])

    def __check_position(self, position):
        """Check whether coordinate is pointing to a real position."""

        return sum(position) == self.size + 2
    
    def __check_boundaries(self, position):
        """Check whether position coordinates follow appropriate bounds"""
        arr = np.array(position)
        return all(arr >= 0) and all(arr < self.layers)

    def __get_right_coord(self, x, y):
        """Get right distance given an x and y coordinate."""
        current_row = self.rows[-(x + 1)]
        return len(current_row) - (y + 1)

    def __init_board(self, board_string):
        """
        Initialize board into a dictionary of dictionaries.

        Initialize the playing board as a dictionary of dictionaries. The
        dictionary stores all points on the board as keys, and returns the color
        of the node and all adjacent nodes as values.

        Arguments:
            board_string (string): string denoting current state of the board.
            (e.g. "[13][302][1003][31002][100003][3000002][121212]")

        Returns:
            (dict): instantiate board as dictionary of dictionaries:
                Key: (x, y, z): redundant coordinate for node position on board.
                    where x = height, y = left distance, and z = right distance.
                Value: dictionary
                    (key, string) 'color': (value, string) color of node.
                    (key, string) 'neighbors': (value, list) tuple coordinates
                        of neighboring nodes.
        """
        # extract strings in between square brackets
        rows = re.findall("\[(.*?)\]", board_string)

        # set size of board by accessing first non-boundary row and ignoring
        # flanking boundary nodes.
        self.size = len(rows[-2]) - 2
        self.layers = len(rows)  # number of layers in the node

        # board formatted as a list of lists to emulate input/output format
        self.rows = [[int(s[i]) for i in range(len(s))] for s in rows]

        # dictionary to contain nodes
        self.board = {}
        for x, row in enumerate(rows):
            for y, color in enumerate(row):
                z = len(row) - (y + 1)
                coord = (self.layers - (x + 1), y, z)
                # find neighboring nodes given current coordinates
                self.board[coord] = {'color': int(color),
                                     'neighbors': self.find_adjacent(coord)}

    def __check_move(self, move):
        """
        Determine whether a move contains valid input.

        Arguments:
            move (tuple): 4-tuple used to denote a move following the format
                (c, x, y, z).
        """
        if len(move) != 4:
            raise ValueError("Expected 4-tuple for `move`.")

        color, pos = move[0], move[1:]
        if color not in self.color_dict:
            raise ValueError("Unexpected color value: {}".format(color))

        if not self.__check_position(pos):
            raise ValueError("Non-sense coordinate: {}".format(pos))

        if self.board[pos]['color'] != 0:
            raise ValueError("Illegal move: {} is already colored.".format(pos))



    def make_move(self, move):
        """
        Color a node a new color.

        Arguments:
            move (tuple): 4-tuple used to denote a move following the format
                (c, x, y, z).
        """
        self.__check_move(move) 

        color, pos = move[0], move[1:]
        if self.board[pos]['color'] == 0:
            self.board[pos]['color'] = color
            self.rows[-(pos[0] +1)][pos[1]] = color
            self.last_move = move
        else:
            print("Illegal move: node already colored.")

        
    def game2str(self):
        """Print the current board state."""
        board_s = ['[{}]'.format(''.join(str(e) for e in x)) for x in self.rows]
        move_s = 'null'
        if self.last_move is not None:
            move_s = "({})".format(','.join(str(e) for e in self.last_move))
        return ''.join(board_s) + "LastPlay:" + move_s

    
    def find_adjacent(self, position):
        """
        Find all possible adjacent positions to a given node.

        Arguments:
            position (tuple, int): coordinates following the (x, y, z) format
                for the node of interest.
        
        Returns:
            (list, tuple, int): list of coordinates for all adjacent nodes.
        """
        neighbors = []
        # possible allowable x and y  movements to adjacent nodes.
        # shifts ordered clockwise starting from immediate left for
        # easy triangle checks
        shifts = [(0, -1), (1, -1), (1, 0), (0, 1), (-1, 1), (-1, 0)]

        for dx, dy in shifts:
            new_x = position[0] + dx
            new_y = position[1] + dy
            if new_x >= 0 and new_x < self.layers:
                new_z = self.__get_right_coord(new_x, new_y)
                new_pos = (new_x, new_y, new_z)
                if self.__check_boundaries(new_pos):
                    neighbors.append(new_pos)

        return neighbors

    def find_possible_moves(self):
        """
        Find all possible moves.

        Finds all possible moves on the current board state. If `last_move` is
        none all positions are availabe. Else only nodes adjacent to `last_move`
        are provided.

        Returns:
            (list, tuple): list of tuples representing moves following the
                (c, x, y, z) format.
        """
        allowed_positions = []
        possible_moves = []

        # If no move has been made yet, moving to any empty node is possible
        if self.last_move is None:
             for pos in self.board:
                 if self.board[pos]['color'] == 0:
                     allowed_positions.append(pos)

        # If a move has been made, subsequent move must be to adjacent empty
        # node
        else:
            adj = self.board[self.last_move[1:]]['neighbors']
            allowed_positions = [n for n in adj if self.board[n]['color'] == 0]

            # if no uncolored, adjacent circles, find any uncolored circles
            if len(allowed_positions) == 0:
                for pos in self.board:
                    if self.board[pos]['color'] == 0:
                        allowed_positions.append(pos)

        for e in itertools.product(*[allowed_positions, [1, 2, 3]]):
            possible_moves.append((e[1], ) + e[0])
        
        return possible_moves

    def triangle_check(self, move):
        """
        Check whether a move will result in a tri-colored triangle.

        Check whether a move will result in a tri-colored triangle, and thus
        losing the game.

        Arguments:
            move (tuple): 4-tuple used to denote a move following the format
                (c, x, y, z).

        Returns:
            (boolean): whether a tri-colored triangle will be produced by the
                suggested move.
        """
        self.__check_move(move)
        color, pos = move[0], move[1:]
        neighbors = self.board[pos]['neighbors']
        # append first element to back of the list for "looping"
        neighbors.append(neighbors[0])
        for i, n1 in enumerate(neighbors[:-1]):
            n2 = neighbors[i + 1]
            if n1 in self.board[n2]['neighbors']:
                color_set = set([color, self.board[n1]['color'],
                                self.board[n2]['color']])
                if color_set == set([1, 2, 3]):
                    return True

        return False


    def evaluate_move(self, move):
        """
        Perform a static evaluation to determine the appropriateness of a move.

        Arguments:
            move (tuple): 4-tuple used to denote a move following the format
                (c, x, y, z).

        Returns:
            (float): value representing how good the proposed move is. Higher
                is generally better.
        """
        self.__check_move(move)

        color, pos = move[0], move[1:]

        # 0 indicates game loss
        if self.triangle_check(move):
            return -np.inf

        local_colors = [self.board[x]['color'] for x in self.board[pos]['neighbors']]
        local_colors.append(color)
        local_colors = np.array(local_colors)
        local_colors = local_colors[local_colors != 0]

        # Calculate Shannon equitability index over local neighborhood
        local_diversity = self.calculate_diversity(local_colors)

        # Calculate global diversity
        global_colors = [game.board[x]['color'] for x in game.board if game.board[x]['color'] != 0]
        global_diversity = self.calculate_diversity(global_colors)
        # Calculate porportion of non-zero labels
        non_empty = np.count_nonzero(local_colors) / len(local_colors)

        # if move would result in fully colored neighborhood, move is
        # completely safe and forces difficult move from opponent, otherwise
        # multiply by negative to force bad plays
        if non_empty == 1:
            non_empty *= 2
        # else:
        #     non_empty *= -1

        out = np.array([self.lambda1 * local_diversity,
                       self.lambda2 * non_empty,
                       self.lambda3 * (local_diversity / global_diversity)])
        return np.sum(out)

    def calculate_diversity(self, colors):
        """
        Calculate the Shannon Diversity Index / Entropy over a set of colors.

        Shannon's equitability is returned to ensure values between 0 and 1

        Arguments:
            colors (array-like): array-like container of color labels.

        Returns:
            (float): Shannon's equitability index.
        """
        colors = np.array(colors)
        entropy = stats.entropy(colors) #/ np.log(3)
        assert entropy >= 0
        return entropy

    def copy(self):
        return AtroposGame(self.game2str())


class AtroposAI(object):
    """
    An artifical intelligence class to play the 'Atropos' game using the
    minimax algorithm.

    Instance Variables:
        game (AtroposGame): an AtroposGame instance.
    """

    def __init__(self, game):
        """
        An artifical intelligence class to play the 'Atropos' game using the
        minimax algorithm with alpha-beta pruning.

        Arguments:
            game (AtroposGame): an AtroposGame instance.
            read_depth (int): number of moves to look ahead. Default is 4. 
        """

        self.game = game
        self.read_depth = 3
        self.__init_minimax_tree()

    def __init_minimax_tree(self):
        """
        Instantiate minimax tree.
        """
        self.tree = {'root': {'game': self.game, 'score': None, 'parent': None,
                             'children': None, 'layer': 0, 'best_move': None}}
        self.nodes_at_layer = {0: ['root']}
        past_nodes = []
        for i in range(self.read_depth - 1):
            # expand from initial game state
            if i == 0:
                children_keys = self.expand_node(self.game, i + 1, 'root')
                self.tree['root']['children'] = children_keys
                if len(children_keys) == 0:
                    children_keys = None
                past_nodes = children_keys
            else:
                # expand from all possible game states create from previous
                # moves
                searched_nodes = []
                for key in past_nodes:
                    if self.tree[key]['score'] >= 0:
                        children_keys = self.expand_node(self.tree[key]['game'],
                                                         i + 1, key)
                        if len(children_keys) == 0:
                            children_keys = None
                        self.tree[key]['children'] = children_keys
                        
                        if children_keys is not None:
                            searched_nodes += children_keys
                past_nodes = searched_nodes

            self.nodes_at_layer[i + 1] = past_nodes
                    

    def expand_node(self, game, layer, parent):
        moves = game.find_possible_moves()
        keys = []
        for i, mv in enumerate(moves):
            new_game = game.copy()
            score = new_game.evaluate_move(mv)
            new_game.make_move(mv)
            key = '{}-{}'.format(parent, i)
            keys.append(key)
            self.tree[key] = {'game': new_game,
                              'score': score, 'parent': parent,
                              'children': None, 'layer': layer,
                              'best_move': None}
        return keys

    def minimax(self):
        """
        Perform minimax over the movement tree to find the best move.
        """
        # extract nodes in the second to last layer
        for layer in range(self.read_depth - 2, -1, -1):
            nodes = self.nodes_at_layer[layer]
            for node in nodes:
                if self.tree[node]['children'] is not None:
                    leaves = self.tree[node]['children']
                    scores = np.array([self.tree[n]['score'] for n in leaves])
                    # parent branch is even --> maximize over children
                    if (layer % 2) == 0:
                        move = np.argmax(scores)
                    # parent branch is odd --> minimize
                    else:
                        # adjust weights to losing moves commited by player2
                        scores[scores == -np.inf] = np.inf
                        move = np.argmin(scores)
                    key, score = leaves[move], scores[move]
                    self.tree[node]['score'] = score
                    self.tree[node]['best_move'] = key   


if __name__ == "__main__":
    # print to stderr for debugging purposes
    # remove all debugging statements before submitting your code
    msg = "Given board " + sys.argv[1] + "\n";
    sys.stderr.write(msg);

    #parse the input string, i.e., argv[1]
    game = AtroposGame(sys.argv[1])
    
    #perform intelligent search to determine the next move
    ai = AtroposAI(game)
    ai.minimax()
    best_move = ai.tree['root']['best_move']
    game.make_move(ai.tree[best_move]['game'].last_move)
    out = '({})'.format(','.join((str(s) for s in game.last_move)))

    #print to stdout for AtroposGame
    sys.stdout.write(out);
    # As you can see Zook's algorithm is not very intelligent. He 
    # will be disqualified.

