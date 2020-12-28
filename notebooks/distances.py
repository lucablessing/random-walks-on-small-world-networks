def dist_1d(a: int, b: int, L: int) -> int:
    '''
    Distance between two points on a periodic
    line with length L.

    Arguments:
    - a: first point (integer 0,..,L-1)
    - b: second point (integer 0,...,L-1)
    - L: number of points on line

    Return:
    - d: distance between points
    '''
    # handle usage of bad arguments
    if a >= L or b >= L or a < 0 or b < 0:
        print('INFO: \na={} or b={} should be in range [0,L-1]=[0,{}]. Using \
        a%L={} or b%L={} instead.\n'.format(a, b, L, a % L, b % L))
        a = a % L
        b = b % L
    if not isinstance(a, int) \
        or not isinstance(b, int) \
            or not isinstance(L, int):
        print('INFO: \nAll arguments should be type int. You used type(a)=\
            {}, type(b)={}, type(L)={}. \nWill be converted using int(..).\
            \n'.format(type(a), type(b), type(L)))
        a, b, L = int(a), int(b), int(L)

    # return distance
    d = 0
    if abs(a-b) < L/2.:
        d = abs(a-b)
    else:
        d = L-abs(a-b)
    return d


def dist_2d_lattice(u: tuple, v: tuple, L: int) -> int:
    '''
    Distance between two node on a periodic
    lattice graph with side length L.

    Arguments:
    - u: first node (tuple (u_0, u_1))
    - v: second node (tuple (v_0, v_1))
    - L: side length of lattice

    Return:
    - d: distance between points (-1 if error occured)
    '''
    # handle arguments having wrong types
    if type(u) != tuple or type(v) != tuple:
        print('ERROR: \nArguments u and v should be type tuple. You used type(\
         u)={}, type(v)={}. \nBreak and return -1\n'.format(type(u), type(v)))
        return -1
    if type(L) != int:
        print('INFO: \nArgument L should be type int. You used type(L)={}. Now \
         using int(L)={}\n'.format(type(L), int(L)))
        L = int(L)

    d = dist_1d(u[0], v[0], L) + dist_1d(u[1], v[1], L)
    return d
