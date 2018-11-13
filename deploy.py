import json
import os
import time
from sys import maxsize as infinity

import chess
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from keras.models import load_model

app = Flask(__name__)
CORS(app)


@app.route('/', methods=['GET'])
def process():
    if request.method == 'GET':
        input_data = dict(request.args)
        if input_data == {}:
            return app.send_static_file('chess.html')
        else:
            print('------------------------')
            print('Input Data: ', input_data)
            print('------------------------')
            move = input_data['move_sent_to_py'][0]
            position = input_data['position'][0]
            computer_color = input_data['ComputerColor'][0]

            engine = Engine(move, position, computer_color)
            jsonified = jsonify(engine.build_output_data())
            return jsonified


class Engine():
    def __init__(self, move_recieved, position_recieved, computer_color):
        ''' Chess engine.
        Recieves
        move recieved in uci form (i.e. a1b1)
        position recieved in FEN notation
        computer colour as 'w' or 'b'

        Returns a dict
        ['move from'] = int between 0 and 63
        ['move to'] = int between 0 and 63
        ['bits'] = position in binary form (explained in build_binary_move)
        ['position'] = position in FEN notation
        '''
        print('Last user move made: ', move_recieved)
        print('Last position recorded: ', position_recieved)

        self.pieces = {
            'P': 1,
            'N': 2,
            'B': 3,
            'R': 4,
            'Q': 5,
            'K': 6,
            'p': 7,
            'n': 8,
            'b': 9,
            'r': 10,
            'q': 11,
            'k': 12
        }
        self.side = computer_color
        self.first_move = False
        self.move_recieved = move_recieved

        if position_recieved == 'None':
            self.board = chess.Board()
            self.first_move = True
        else:
            self.board = chess.Board(position_recieved)
        self.last_turn = self.board.fen().split()[1]
        self.turns = int(self.board.fen().split()[5])
        print(self.last_turn, ' played their move')
        if self.move_recieved != 'None':
            self.check_for_promotion(*squares_to_numbers(move_recieved,
                                                         mirror=False))
            self.move_recieved = chess.Move.from_uci(move_recieved)
            self.board.push(self.move_recieved)

    def check_for_promotion(self, mv_frm, mv_to):
        'Checks for client-side promotion'
        white_queen = chess.Piece(piece_type=chess.QUEEN, color=chess.WHITE)
        black_queen = chess.Piece(piece_type=chess.QUEEN, color=chess.BLACK)

        if self.board.piece_at(mv_frm).piece_type == chess.PAWN:
            if chess.square_rank(mv_to) == 7:
                if self.board.piece_at(mv_frm).color == chess.WHITE:
                    promoted_binary = bin(self.pieces['Q'])[2:].zfill(4)
                    self.board.set_piece_at(square=mv_frm,
                                            piece=white_queen)
            elif chess.square_rank(mv_to) == 0:
                if self.board.piece_at(mv_frm).color == chess.BLACK:
                    promoted_binary = bin(self.pieces['q'])[2:].zfill(4)
                    self.board.set_piece_at(square=mv_frm,
                                            piece=black_queen)

    def build_binary_move(self):
        '''Javascript chess gui uses binary encoding to represent moves:
        0000 0000 0000 0000 0000 0111 1111 -> From 0x7F
        0000 0000 0000 0011 1111 1000 0000 -> To >> 7, 0x7F
        0000 0000 0011 1100 0000 0000 0000 -> Captured >> 14, 0xF
        0000 0000 0100 0000 0000 0000 0000 -> EP 0x40000
        0000 0000 1000 0000 0000 0000 0000 -> Pawn Start 0x80000
        0000 1111 0000 0000 0000 0000 0000 -> Promoted Piece >> 20, 0xF
        0001 0000 0000 0000 0000 0000 0000 -> Castle 0x1000000
        This function takes movements made and converts them to binary form.
        '''
        captured_piece_binary = '0000'
        en_passant_binary = '0'
        pawn_start_binary = '0'
        promoted_binary = '0000'
        castling_binary = '0'
        white_queen = chess.Piece(piece_type=chess.QUEEN, color=chess.WHITE)
        black_queen = chess.Piece(piece_type=chess.QUEEN, color=chess.BLACK)

        if self.board.is_capture(self.uci_move):
            if self.board.is_en_passant(self.uci_move):
                captured_piece = self.board.piece_at(self.board.ep_square)
            else:
                captured_piece = self.board.piece_at(self.move_to_square)
            captured_piece_binary = bin(self.pieces[str(captured_piece)])[2:].zfill(4)

        if self.board.is_en_passant(self.uci_move):
            en_passant_binary = '1'

        if self.board.piece_at(self.move_from_square).piece_type == chess.PAWN:
            if chess.square_distance(self.move_from_square,
                                     self.move_to_square) == 2:
                pawn_start_binary = '1'

        if len(str(self.uci_move)) > 4:
            promoted_piece = str(self.uci_move)[4]
            if self.side == 'w':
                promoted_piece = promoted_piece.upper()
            promoted_binary = bin(self.pieces[promoted_piece])[2:].zfill(4)

        if self.board.is_castling(self.uci_move):
            castling_binary = '1'

        result = castling_binary + promoted_binary + pawn_start_binary + \
            en_passant_binary + captured_piece_binary

        return result

    def minimax(self, node, depth, player, alpha, beta):
        if player == 'w':
            player = 1
        elif player == 'b':
            player = -1

        if depth == 0 or node.children == []:
            return [player*node.value]
        if node.children[0] is not None:
            predicted_child = node.children[0][0]

        favourite_child = None
        best_advantage = -1*player*infinity
        for child, current_value in node.children:
            node.board.push(child)
            result = self.minimax(Node(node.board), depth-1,
                                  -1*player, alpha, beta)

            opposition_value = result[0]
            advantage_score = player*current_value + opposition_value
            if player == 1:
                if advantage_score > best_advantage:
                    best_advantage = advantage_score
                    favourite_child = child
                    alpha = max(alpha, best_advantage)
                    if beta <= alpha:
                        node.board.pop()
                        break
            elif player == -1:
                if advantage_score < best_advantage:
                    best_advantage = advantage_score
                    favourite_child = child
                    beta = min(beta, best_advantage)
                    if beta <= alpha:
                        node.board.pop()
                        break
            node.board.pop()

        return [best_advantage, favourite_child, predicted_child]

    def build_output_data(self):
        'Takes the result from minimax and returns a dict'
        output_data = {}
        if self.turns > 15:
            sdepth = 3
        else:
            sdepth = 1

        if (self.last_turn != self.side or
                self.first_move is True or self.move_recieved == 'None'):
            result = self.minimax(Node(board=self.board),
                                  depth=sdepth, player=self.side,
                                  alpha=-1*infinity, beta=infinity)
            print('result:', result)
            self.uci_move = result[1]
            pick_from = str(self.uci_move)[0:2].upper()
            pick_to = str(self.uci_move)[2:4].upper()
            self.move_from_square = int(getattr(chess, pick_from))
            self.move_to_square = int(getattr(chess, pick_to))

            output_data['move_from'] = self.move_from_square
            output_data['move_to'] = self.move_to_square
            output_data['bits'] = self.build_binary_move()
            self.board.push(self.uci_move)

        output_data['position'] = self.board.fen()
        print('-----------------------------')
        print('Output Data: ', output_data)
        print('Best score: ', result[0])
        print('NN prediction: ', result[2])
        print('-----------------------------')
        return output_data


class Node():
    'Node class used for minimax'
    def __init__(self, board):
        self.material_values = {
            chess.KING: 50000,
            chess.QUEEN: 5000,
            chess.ROOK: 900,
            chess.KNIGHT: 500,
            chess.BISHOP: 500,
            chess.PAWN: 10
        }
        self.board = board
        self.turns = int(self.board.fen().split()[5])
        self.children = []

        if self.turns > 15:
            self.create_children(15)
        else:
            self.create_children(100)

        if self.children == []:
            self.value = 0
        else:
            self.value = self.children[0][1]

    def create_children(self, n):
        self.best_moves = self.get_best_moves(*self.predict_moves())
        self.children.extend(self.best_moves[:n])

    def predict_moves(self):
        '''Uses the saved deep learning models to return a 2 lists of
        moved_from probabilities and moved_to probabilities'''
        t1 = time.time()
        nn_input = self.board.position_list_one_hot()
        nn_input = np.array(nn_input).reshape(1, 8, 8, 12)

        with graph.as_default():
            predictions = list(move_from_model.predict(nn_input))
            probabilities = list(predictions[0])
            move_from_squares = sorted(range(len(probabilities)),
                                       key=lambda k: probabilities[k])
            move_from_squares = (list(reversed(move_from_squares)))

        with graph.as_default():
            predictions = list(move_to_model.predict(nn_input))
            probabilities = list(predictions[0])
            move_to_squares = sorted(range(len(probabilities)),
                                     key=lambda k: probabilities[k])
            moved_to_squares = (list(reversed(move_from_squares)))

        return move_from_squares, move_to_squares

    def get_material_scores(self, moves):
        '''Generates material scores found by projecting the outcome
        if a move is made. If a capture is made, the material value goes
        up by the piece captured. If checkmate, the material value goes up
        by the value of the king. If stalemate, the material value goes down
        by 100000.'''
        material_scores = []
        for move in moves:
            material_score = 0
            if self.board.is_capture(move):
                if self.board.is_en_passant(move):
                    captured_piece = chess.PAWN
                else:
                    moved_to = getattr(chess, str(move)[2:4].upper())
                    captured_piece = self.board.piece_at(moved_to).piece_type
                material_score += self.material_values[captured_piece]

            self.board.push(move)
            if self.board.is_checkmate():
                material_score += self.material_values[chess.KING]
            elif self.board.is_stalemate():
                material_score -= 100000
            else:
                material_score += 0
            self.board.pop()
            material_scores.append(material_score)
        return material_scores

    def get_best_moves(self, from_sqs_list, to_sqs_list):
        '''Matches the probabilities found in predict_moves and minimizes
        the distance between the probabilities and a legal move. Returns an
        a list of legal moves, ordered by score (found by combining prediction
        score and material score).'''
        legal_moves = [str(legal) for legal in list(self.board.legal_moves)]
        legal_moves_numbered = [squares_to_numbers(move)
                                for move in list(self.board.legal_moves)]
        to_sqs_list = list(reversed(to_sqs_list))
        total_uncertainties = []

        for fro, to in legal_moves_numbered:
            uncertainty_from = from_sqs_list.index(fro)
            uncertainty_to = to_sqs_list.index(to)
            total_uncertainties.append(uncertainty_from + uncertainty_to)

        moves_ordered = [chess.Move.from_uci(move) for _, move in
                         sorted(zip(total_uncertainties, legal_moves),
                         key=lambda x: x[0])]

        prediction_scores = [(400-uncertainty) for uncertainty in
                             sorted(total_uncertainties)]
        material_scores = self.get_material_scores(moves_ordered)
        total_scores = [p_score + m_score for p_score, m_score in
                        zip(prediction_scores, material_scores)]

        return ([[move, score] for score, move in
                sorted(zip(total_scores, moves_ordered),
                       key=lambda x: x[0], reverse=True)])


def squares_to_numbers(move, mirror=True):
    '''converts a move in uci form (i.e. a1b1) to its squares
    returns two ints
    in the above case 0, 1
    can be mirrored, which will return 56, 57'''
    first_square = str(move)[0:2].upper()
    second_square = str(move)[2:4].upper()
    first_square_num = getattr(chess, first_square)
    second_square_num = getattr(chess, second_square)
    if mirror is True:
        first_square_num = chess.square_mirror(first_square_num)
        second_square_num = chess.square_mirror(second_square_num)

    return (first_square_num, second_square_num)


def position_list_one_hot(self):
    '''method added to the python-chess library for faster
    conversion of board to one hot encoding. Resulted in 100%
    increase in conversion speed by bypassing conversion to fen() first.
    '''
    builder = []
    builder_append = builder.append
    for square in chess.SQUARES_180:
        mask = chess.BB_SQUARES[square]

        if not self.occupied & mask:
            builder.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        elif bool(self.occupied_co[chess.WHITE] & mask):
            if self.pawns & mask:
                builder.extend([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
            elif self.knights & mask:
                builder.extend([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
            elif self.bishops & mask:
                builder.extend([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
            elif self.rooks & mask:
                builder.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
            elif self.queens & mask:
                builder.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
            elif self.kings & mask:
                builder.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        elif self.pawns & mask:
            builder.extend([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        elif self.knights & mask:
            builder.extend([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        elif self.bishops & mask:
            builder.extend([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        elif self.rooks & mask:
            builder.extend([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        elif self.queens & mask:
            builder.extend([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        elif self.kings & mask:
            builder.extend([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])

    return builder


chess.BaseBoard.position_list_one_hot = position_list_one_hot

global graph
graph = tf.get_default_graph()
data_folder = r'static/chessai/model'
moved_from_file = os.path.join(data_folder, 'moved_from_model.h5')
moved_to_file = os.path.join(data_folder, 'moved_to_model.h5')

move_from_model = load_model(moved_from_file)
move_to_model = load_model(moved_to_file)


if __name__ == '__main__':
    gpu_mode = False
    if gpu_mode is True:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    app.run(debug=True)
