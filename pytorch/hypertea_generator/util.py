from functools import reduce

def list2vecstr_(l, vec_type = 'int'):
        return 'std::vector<{}> {{'.format(vec_type)  + ','.join(map(str, l)) + '}'


def bool2str_(x):
    return 'true' if x else 'false'

def bool2inplace_str_(x):
    return 'IN_PLACE' if x else 'NOT_IN_PLACE'


def prod_(l):
    return reduce(lambda x, y: x*y, l)