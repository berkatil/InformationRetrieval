import bisect

class Node:
    """
    Parameters
    ----------
    letter:
        The value of this node
    children_letters:
        An array which contains the letters of children nodes
    children_nodes:
        An array of pointers to children nodes
    is_terminal:
        Stores if this node is the end of any word(last character of any word)
    """
    def __init__(self, letter,children_letters,children_nodes):
        self.letter = letter
        self.children_letters = children_letters
        self.children_nodes = children_nodes
        self.is_terminal = False

def find(array, elem):
    i = bisect.bisect_left(array, elem)
    if i != len(array) and array[i] == elem:
        return i
    return -1