import numpy as np
'File is used in Keras Conv.py for testing'


def next_pos():
    next_pos = ['r', 'n', 'b', 'q', 'k', 'b', 'n', 'r',
                'p', 'p', 'p', 'p', 'p', 'p', 'p', 'p',
                '.', '.', '.', '.', '.', '.', '.', '.',
                '.', '.', '.', '.', '.', '.', '.', '.',
                '.', '.', '.', '.', '.', '.', '.', '.',
                '.', '.', '.', '.', '.', '.', '.', '.',
                'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P',
                'R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R']

    return next_pos


def make_clean_board():
    next_position = next_pos()
    input_board = []
    for input_square in next_position:
        if input_square == '.':
            input_board.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        elif input_square == 'p':
            input_board.extend([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        elif input_square == 'n':
            input_board.extend([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        elif input_square == 'b':
            input_board.extend([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        elif input_square == 'r':
            input_board.extend([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        elif input_square == 'q':
            input_board.extend([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        elif input_square == 'k':
            input_board.extend([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
        elif input_square == 'P':
            input_board.extend([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
        elif input_square == 'N':
            input_board.extend([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
        elif input_square == 'B':
            input_board.extend([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
        elif input_square == 'R':
            input_board.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
        elif input_square == 'Q':
            input_board.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
        elif input_square == 'K':
            input_board.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    return np.reshape(np.array(input_board), (1, 768))


def convert_position_prediction(output_board_one_hot):
    square_one_hot = []
    squares = []
    for i, element in enumerate(output_board_one_hot):
        square_one_hot.append(element)
        if (i+1) % 12 == 0:
            squares.append(square_one_hot)
            square_one_hot = []

    builder = []
    for i, piece_one_hot in enumerate(squares):
        if piece_one_hot == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]:
            builder.append('.')
        elif piece_one_hot == [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]:
            builder.append('p')
        elif piece_one_hot == [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]:
            builder.append('n')
        elif piece_one_hot == [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]:
            builder.append('b')
        elif piece_one_hot == [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]:
            builder.append('r')
        elif piece_one_hot == [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]:
            builder.append('q')
        elif piece_one_hot == [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]:
            builder.append('k')
        elif piece_one_hot == [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]:
            builder.append('P')
        elif piece_one_hot == [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]:
            builder.append('N')
        elif piece_one_hot == [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]:
            builder.append('B')
        elif piece_one_hot == [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]:
            builder.append('R')
        elif piece_one_hot == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]:
            builder.append('Q')
        elif piece_one_hot == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]:
            builder.append('K')
        if (i+1) % 8 == 0:
            builder.append('\n')
        else:
            builder.append(' ')

    return (''.join(builder))


def move_from(output_one_hot):
    board_position = next_pos()
    builder = []
    for i, e in enumerate(output_one_hot):
        # print(e)
        if e == 0:
            builder.append(board_position[i])
        else:
            builder.append(str(e))
        if (i+1) % 8 == 0:
            builder.append('\n')
        else:
            builder.append(' ')
    return ''.join(builder)
