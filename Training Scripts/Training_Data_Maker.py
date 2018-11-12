import chess.pgn
from time import time
import sys
import math
import numpy as np
import pandas as pd
import os
from multiprocessing import Pool, Manager
from itertools import product
import h5py

'''final training data will have the format:
         Network 1
      x            y
 [position1], [moved_from],
 [position2], [moved_from],
 [position3], [moved_from],
      .            .

          Network 2
      x            y
 [position1], [moved_to],
 [position2], [moved_to],
 [position3], [moved_to],
      .            .
each position is a 768-element list of numbers(each element can be from 1-12 for each of the pieces * 64 elements) indicating what piece is in the square
blank square:  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
'p':           [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
'n':           [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
'b':           [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
'r':           [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
'q':           [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
'k':           [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
'P':           [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
'N':           [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
'B':           [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
'R':           [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
'Q':           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
'K':           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]'''


def position_list_one_hot(self):
    '''method added to the python-chess library for faster
    conversion of board to one hot encoding. Resulted in 100%
    increase in speed by bypassing conversion to fen() first.
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


def position_list(self):
    '''same as position_list_one_hot except this is converts pieces to
    numbers between 1 and 12. Used for piece_moved function'''
    builder = []
    builder_append = builder.append
    for square in chess.SQUARES_180:
        mask = chess.BB_SQUARES[square]

        if not self.occupied & mask:
            builder_append(0)
        elif bool(self.occupied_co[WHITE] & mask):
            if self.pawns & mask:
                builder_append(7)
            elif self.knights & mask:
                builder_append(8)
            elif self.bishops & mask:
                builder_append(9)
            elif self.rooks & mask:
                builder_append(10)
            elif self.queens & mask:
                builder_append(11)
            elif self.kings & mask:
                builder_append(12)
        elif self.pawns & mask:
            builder_append(1)
        elif self.knights & mask:
            builder_append(2)
        elif self.bishops & mask:
            builder_append(3)
        elif self.rooks & mask:
            builder_append(4)
        elif self.queens & mask:
            builder_append(5)
        elif self.kings & mask:
            builder_append(6)

    return builder

chess.BaseBoard.position_list_one_hot = position_list_one_hot
chess.BaseBoard.position_list = position_list


def progressBar(value, endvalue, time1, time2, bar_length=20):
    '''progress bar with ETA for convenience.
     Did not affect time taken.'''
    percent = float(value) / endvalue
    arrow = '-' * int(round(percent * bar_length) - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))

    sys.stdout.write("\rProgress: [{0}] {1}% ".format(arrow + spaces, int(round(percent * 100))))
    seconds = ((time2 - time1) / (percent)) - (time2-time1)
    sys.stdout.write("ETA: {} minutes and {:0.0f} seconds".format(math.floor(seconds / 60), seconds % 60))
    sys.stdout.flush()


def piece_moved(position1, position2):
    '''Main data conversion function.
    step 1: checks the difference between two positions and returns a list
            of the affected squares.
    step 2: checks whether it is a normal move (only two squares affected), or
            en passant (3 squares affected) or castling (4 squares affected)
            step 2a: If castling, the square moved from is where the king was
                     in the beginning of the turn. Square moved to is where
                     the king is at the end of the turn.
            step 2b: If en passant, square moved from is where the pawn was
                     at the beginning of the turn. Moved to is where the pawn
                     is at the end of the turn.
    step 3: Returns two ints with the square moved from, and square moved to
    '''
    affected_squares = []
    for i in range(64):  # Step 1
        if position1[i] != position2[i]:
            affected_squares.append(i)
    if len(affected_squares) > 2:  # Step 2
        for square in affected_squares:
            if position1[square] == 12 or position1[square] == 6:  # Step 2a
                moved_from = square
            if position2[square] == 12 or position2[square] == 6:
                moved_to = square
            if position1[square] == 0:  # Step 2b
                if position2[square] == 1:
                    moved_to = square
                    for square in affected_squares:
                        if position1[square] == 1:
                            moved_from = square
                elif position2[square] == 7:
                    moved_to = square
                    for square in affected_squares:
                        if position1[square] == 7:
                            moved_from = square
    else:
        if position2[affected_squares[0]] == 0:
            moved_from, moved_to = affected_squares[0], affected_squares[1]
        else:
            moved_from, moved_to = affected_squares[1], affected_squares[0]
    return moved_from, moved_to


# ------------ test for full file ------------
h5_folder = 'testtrainingfiles'
Bulk_data = 'test_Data.h5'


def parse_file(data_folder, text_file):
    '''file parser
    Step 1: loops through 1 PGN (portable game notation) file game by game.
    Step 2: Accounts for edge cases where play was stopped in the middle of a move
            due to disconnection during the game
    Step 3: Adds the training data to an h5 file'''
    pgn = open(os.path.join(data_folder, text_file))
    counter = 0
    startfile = time()
    filename = text_file.split('.')[0]
    train_input, moved_from, moved_to = [], [], []

    while True:
        game = chess.pgn.read_game(pgn)
        if game is None:
            break
        counter += 1
        board = game.board()  # set the game board
        first = True

        for move in game.main_line():
            if first:
                position1 = board.position_list()
                one_hot_position = board.position_list_one_hot()
                train_input.append(one_hot_position)
                first = False
            board.push(move)

            one_hot_position = board.position_list_one_hot()
            position2 = board.position_list()
            train_input.append(one_hot_position)
            piece_from, piece_to = piece_moved(position1, position2)
            moved_from.append(piece_from)
            moved_to.append(piece_to)
            position1 = position2

        if len(train_input) - len(moved_from) == 1:
            del train_input[-1]

    try:
        position = np.array(train_input)
        moved_from = np.array(moved_from)
        moved_from_one_hot = np.zeros((moved_from.size, 64))
        moved_from_one_hot[np.arange(moved_from.size), moved_from] = 1
        moved_to = np.array(moved_to)
        moved_to_one_hot = np.zeros((moved_to.size, 64))
        moved_to_one_hot[np.arange(moved_to.size), moved_to] = 1

        h5f = h5py.File(h5_folder + '/' + filename +'.h5', 'w')
        h5f.create_dataset('input_position', data=position)
        h5f.create_dataset('moved_from', data=moved_from_one_hot)
        h5f.create_dataset('moved_to', data=moved_to_one_hot)
        h5f.close
    except:
        print(50*'-')
        print(50*'-')
        print('ERROR IN {}, GAME {}'.format(text_file, counter))
        print(50*'-')
        print(50*'-')

    print("\n{} processed in {:0.3f} seconds".format(filename, time()-startfile))


def combine_h5s():
    '''Main file returns many h5 files, this combines all in one.
    Creates the dataset if the h5 file doesn't already exist.
    '''
    saved_h5s = [x for x in os.listdir(h5_folder)]

    first = True
    for i, file in enumerate(saved_h5s):
        if first:
            add_file = h5py.File(h5_folder + '/' + file, 'r')
            position = add_file['input_position']
            moved_from = add_file['moved_from']
            moved_to = add_file['moved_to']
            h5f = h5py.File(Bulk_data, 'w')
            h5f.create_dataset('input_position', data=position, maxshape=(None, 768))
            h5f.create_dataset('moved_from', data=moved_from, maxshape=(None, 64))
            h5f.create_dataset('moved_to', data=moved_to, maxshape=(None, 64))
            h5f.close
            first = False
        else:
            add_file = h5py.File(h5_folder + '/' + file, 'r')
            X = add_file['input_position']
            moved_from = add_file['moved_from']
            moved_to = add_file['moved_to']

            hf = h5py.File(Bulk_data, 'a')
            hf["input_position"].resize((hf["input_position"].shape[0] + X.shape[0]), axis = 0)
            hf["input_position"][-X.shape[0]:] = X

            hf["moved_from"].resize((hf["moved_from"].shape[0] + moved_from.shape[0]), axis = 0)
            hf["moved_from"][-moved_from.shape[0]:] = moved_from

            hf["moved_to"].resize((hf["moved_to"].shape[0] + moved_to.shape[0]), axis = 0)
            hf["moved_to"][-moved_to.shape[0]:] = moved_to

            hf.close
        print('added file {} out of {} to main hdf5 datafile'.format(i+1, len(saved_h5s)))


def main(data_folder):
    ''' Goes through a series of text files and makes training data.
    Textfiles are created by breaking up the original PGN file downloaded
    from fics.org for lower RAM usage.
    datasplitter.py is responsible for breaking up the original PGN.
    Uses multiprocessing.
    '''
    for i, file in os.listdir(data_folder):
        df = pd.DataFrame(parse_file(file))
        df.to_hdf('trainingdata.h5', key='df', mode='a')
        print(i+1)

    pool = Pool(8)
    for i, _ in enumerate(pool.imap(parse_file, os.listdir(data_folder))):
        print('file {} converted to hdf5'.format(i+1))

    combine_h5s()


if __name__ == '__main__':
    starttime = time()
    main(data_folder='trainingdata')
    print('total time taken: ', time() - starttime)
