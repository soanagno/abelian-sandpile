"""Implementations on the Abelian Sandpile and Conway's
Game of life on various meshes"""
import numpy as np
import scipy.signal as sps
import matplotlib.pyplot as plt


def sandpile(initial_state):
    """
    Generate final state for Abelian Sandpile.
    Parameters
    ----------
    initial_state : array_like or list of lists
        Initial 2d state of grid in an array of integers.
    Returns
    -------
    numpy.ndarray
            Final state of grid in array of integers.
    """

    # Initialisation
    in_state = np.copy(initial_state)
    irows, icols = np.size(in_state, 0), np.size(in_state, 1)
    # insert top & bottom boundaries
    in_state = np.insert(in_state, [0, irows], 0, axis=0)
    # insert L & R boundaries
    in_state = np.insert(in_state, [0, icols], 0, axis=1)

    while (in_state > 3).any():

        # find all the cells > 3
        here = np.where(in_state > 3)
        here = np.array(here)
        x_here = here[0, :]  # store x
        y_here = here[1, :]  # store y

        # Calculations
        in_state[x_here-1, y_here] += 1
        in_state[x_here+1, y_here] += 1
        in_state[x_here, y_here-1] += 1
        in_state[x_here, y_here+1] += 1
        in_state[x_here, y_here] -= 4
        in_state[[0, -1], :] = 0  # boundary condition a
        in_state[:, [0, -1]] = 0  # boundary condition b

        # plt.matshow(in_state, cmap='autumn',vmin=0,vmax=6)
    # print(in_state[1:-1,1:-1])
    # plt.matshow(in_state[1:-1,1:-1], cmap='autumn',vmin=0,vmax=4)
    # plt.show()
    # print(in_state)
    return(in_state[1:-1, 1:-1])


def life(initial_state, nsteps, periodic):

    """
    Perform iterations of Conway’s Game of Life.
    Parameters
    ----------
    initial_state : array_like or list of lists
        Initial 2d state of grid in an array of booleans.
    nsteps : int
        Number of steps of Life to perform.
    periodic : bool
        If true, then grid is assumed periodic.
    Returns
    -------
    numpy.ndarray
        Final state of grid in array of booleans
    """

    # Initialisation
    in_state = np.copy(initial_state)
    irows, icols = np.size(in_state, 0), np.size(in_state, 1)

    # Set auxiliery/temp array
    aux = np.zeros((irows)*(icols)).reshape(irows, icols)
    aux = np.copy(in_state)

    # 3x3 filter array for convolution
    filter_array = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])

    for _ in range(nsteps):

        # convolution with boundary conditions
        if periodic == 1:
            conv = sps.convolve2d(aux, filter_array,
                                  mode='same', boundary='wrap')
        else:
            conv = sps.convolve2d(aux, filter_array,
                                  mode='same', boundary='fill')

        # Calculations
        here1 = np.where(np.logical_or(np.greater(conv, 3), np.less(conv, 2)))
        here2 = np.where(conv == 3)
        here1 = np.array(here1)
        here2 = np.array(here2)
        x_here1 = here1[0, :]  # store x1
        y_here1 = here1[1, :]  # store y1
        x_here2 = here2[0, :]  # store x2
        y_here2 = here2[1, :]  # store y2

        aux[x_here1, y_here1] = 0
        aux[x_here2, y_here2] = 1

        '''
        # Calculations 2 (slow)
        for x in range(irows):
            for y in range(icols):

                if (conv[x, y] not in [2, 3]):
                    aux[x, y] = 0
                elif conv[x, y] == 3:
                    aux[x, y] = 1
        '''

        in_state = np.copy(aux)  # pass auxiliery array into main array

        # plt.matshow(in_state, cmap='binary')
        # print(in_state, aux)
    # plt.matshow(in_state, cmap='binary')
    in_state = np.array(in_state, dtype=bool)
    return(in_state)


def lifetri(initial_state, nsteps, periodic):

    """
    Perform iterations of Conway’s Game of Life on triangles
    Parameters
    ----------
    initial_state : array_like or list of lists
        Initial state of grid on triangles.
    nsteps : int
        Number of steps of Life to perform.
    Returns
    -------
    numpy.ndarray
        Final state of grid in array of booleans
    """

    # Initialisation
    in_state = np.copy(initial_state)
    irows, icols = np.size(in_state, 0), np.size(in_state, 1)
    # Set auxiliery/temp array with extra rows/columns as boundaries
    aux = np.zeros((irows+2)*(icols+4)).reshape(irows+2, icols+4)
    auxrows, auxcols = np.size(aux, 0), np.size(aux, 1)
    aux[1:auxrows-1, 2:auxcols-2] = np.copy(in_state)
    in_state = np.copy(aux)

    for _ in range(nsteps):

        if periodic is True:
            # Circular boundary conditions
            in_state[0, :] = np.copy(aux[-2, :])
            in_state[-1, :] = np.copy(aux[1, :])     # top & bottom sides
            in_state[:, 0:2] = np.copy(aux[:, -4:-2])
            in_state[:, -2:] = np.copy(aux[:, 2:4])  # L & R double sides
            in_state[0, 0:2] = np.copy(aux[-2, -4:-2])
            in_state[-1, -2:] = np.copy(aux[1, 2:4])   # diagonals a
            in_state[0, -2:] = np.copy(aux[-2, 2:4])
            in_state[-1, :2] = np.copy(aux[1, -4:-2])  # diagonals b

        # Calculations
        for x in range(1, auxrows-1):
            for y in range(2, auxcols-2):

                # Up or down indicator (given that first triangle is up).
                # The indicators are used in the neighbor cells' calculation
                # to avoid an if statement for triangle orientation
                up = (x+y+1) % 2 == 0
                down = 1-up

                # Calculate neighbor cells
                # Note that if up=1 then down=0
                # so the expression changes accordingly
                neighbors = \
                    sum(in_state[x-1, (y-1)*up+down*(y-2):(y+2)*up+down*(y+3)]) + \
                    sum(in_state[x, y-2:y+3]) + \
                    sum(in_state[x+1, (y-2)*up+down*(y-1):(y+3)*up+down*(y+2)]) - \
                    in_state[x, y]

                if (neighbors not in [4, 5, 6]):
                    aux[x, y] = 0
                elif neighbors == 4:
                    aux[x, y] = 1

        in_state = np.copy(aux)  # Pass auxiliery array into main array
        # plt.matshow(in_state[1:-1, 2:-2], cmap='binary')

    # plt.matshow(in_state, cmap='binary')
    return(in_state[1:-1, 2:-2])


def life_generic(matrix, initial_state, nsteps, environment, fertility):
    """
    Perform iterations of Conway’s Game of Life for an arbitrary
    collection of cells.
    Parameters
    ----------
    matrix : 2d array of bools
        a boolean matrix indicating neighbours for each row
    initial_state : 1d array of bools
        Initial state vector.
    nsteps : int
        Number of steps of Life to perform.
    environment : set of ints
        neighbour counts for which live cells survive.
    fertility : set of ints
        neighbour counts for which dead cells turn on.
    Returns
    -------
    array
        Final state.
    """

    # Initialisation
    in_state = np.copy(initial_state)
    irows = np.size(in_state, 0)
    # Set auxiliary/temp 1D array
    aux = np.copy(in_state)

    # Calculations
    for _ in range(nsteps):

        for x in range(irows):

            # Calculate neighbor cells using adjecency matrix
            neighbors = np.dot(matrix[x, :], in_state[:])

            if neighbors not in environment:
                aux[x] = 0
            elif neighbors == np.array(list(fertility)):
                aux[x] = 1
        in_state = np.copy(aux)  # Pass auxiliery array into main array

        # plt.matshow(in_state.reshape(5, 5))
    # print(1*in_state)
    # in_state = np.array(in_state, dtype=bool)
    return(in_state)
