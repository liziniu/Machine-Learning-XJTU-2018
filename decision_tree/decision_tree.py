# -*- coding:utf-8 -*-
from math import log2
from PIL import Image, ImageDraw

# Please note that
# This script is written by Python 2.7 !!!

my_data = [['slashdot', 'USA', 'yes', 18, 'None'],
           ['google', 'France', 'yes', 23, 'Premium'],
           ['digg', 'USA', 'yes', 24, 'Basic'],
           ['kiwibotes', 'France', 'yes', 23, 'Basic'],
           ['google', 'UK', 'no', 21, 'Premium'],
           ['(direct)', 'New Zealand', 'no', 12, 'None'],
           ['(direct)', 'UK', 'no', 21, 'Basic'],
           ['google', 'USA', 'no', 24, 'Premium'],
           ['slashdot', 'France', 'yes', 19, 'None'],
           ['digg', 'USA', 'no', 18, 'None'],
           ['google', 'UK', 'no', 18, 'None'],
           ['kiwitobes', 'UK', 'no', 19, 'None'],
           ['digg', 'New Zealand', 'yes', 12, 'Basic'],
           ['google', 'UK', 'yes', 18, 'Basic'],
           ['kiwitobes', 'France', 'yes', 19, 'Basic']]


class DecisionNode:
    def __init__(
            self,
            col=-1,
            value=None,
            results=None,
            tb=None,
            fb=None
    ):
        self.col = col              # the criteria to be tested
        self.value = value          # true value
        self.results = results      # the reslut of classification; non-leave node is None
        self.tb = tb                # true
        self.fb = fb                # false


# divide dataset
def divide_set(rows, column, value):
    # according to value, divide the data into 2 classes
    split_function = None
    if isinstance(value, int) or isinstance(value, float):
        split_function = lambda row: row[column] >= value
    else:
        split_function = lambda row: row[column] == value

    set_1 = [row for row in rows if split_function(row)]
    set_2 = [row for row in rows if not split_function(row)]

    return set_1, set_2


# count the result
def unique_counts(rows):
    results = {}
    for row in rows:
        r = row[len(row) - 1]    # All possible results:None, Basic, Premium
        if r not in results:
            results[r] = 0
        results[r] += 1
    return results


# compute the entropy
def entropy(rows):
    results = unique_counts(rows)
    ent = 0.0
    for r in results.keys():
        p = float(results[r]) / len(rows)
        ent -= p * log2(p)
    return ent

# build tree recursively
def build_tree(rows, scoref=entropy):
    # basic condition 
    if len(rows) == 0:
        return DecisionNode()

    current_score = scoref(rows)      # the score before buidling next sub-tree
    best_gain = 0.0
    best_criteria = None
    best_sets = None

    column_count = len(rows[0]) - 1    # the number of features
    for col in range(column_count):
        column_values = {}
        for row in rows:
            column_values[row[col]] = 1
        for value in column_values.keys():
            set_1, set_2 = divide_set(rows, col, value)

            p = float(len(set_1)) / len(rows)
            gain = current_score - p * scoref(set_1) - (1 - p) * scoref(set_2)
            if gain > best_gain and len(set_1) > 0 and len(set_2) > 0:
                best_gain = gain
                best_criteria = (col, value)
                best_sets = (set_1, set_2)
    # build sub tree
    if best_gain > 0:
        true_branch = build_tree(best_sets[0])
        false_branch = build_tree(best_sets[1])
        return DecisionNode(col=best_criteria[0], value=best_criteria[1], tb=true_branch, fb=false_branch)
    else:
        return DecisionNode(results=unique_counts(rows))

# print the tree structure
def print_tree(tree, indent=''):
    if tree.results is not None:
        print(str(tree.results))
    else:
        # the criteria
        print(str(tree.col) + ':' + str(tree.value) + "?")
        # the left and right branch
        print(indent + "T->", end='')
        print_tree(tree.tb, indent+'  ')
        print(indent + "F->", end='')
        print_tree(tree.fb, indent+'  ')


def get_width(tree):
    if tree.tb is None and tree.fb is None:
        return 1
    return get_width(tree.tb) + get_width(tree.fb)


def get_depth(tree):
    if tree.tb is None and tree.fb is None:
        return 0
    return max(get_depth(tree.tb), get_width(tree.fb)) + 1


def draw_node(draw, tree, x, y):
    if tree.results is None:
        w1 = get_width(tree.fb) * 100
        w2 = get_width(tree.tb) * 100

        left = x - (w1 + w2) / 2
        right = x + (w1 + w2) / 2

        # criteria 
        draw.text((x-20, y-10), str(tree.col) + ":" + str(tree.value), (0, 0, 0))

        draw.line((x, y, left + w1/2, y + 100), fill=(255, 0, 0))
        draw.line((x, y, right - w2/2, y + 100), fill=(255, 0, 0))

        draw_node(draw, tree.fb, left+w1/2, y+100)
        draw_node(draw, tree.tb, right-w2/2, y+100)
    else:
        txt = ' \n'.join(['%s:%d' % v for v in tree.results.items()])
        draw.text((x - 20, y), txt, (0, 0, 0))


def draw_tree(tree, jpeg='tree.jpeg'):
    w = get_width(tree) * 100
    h = get_depth(tree) * 100 + 120

    img = Image.new('RGB', (w,h), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    draw_node(draw, tree, w/2, 20)
    img.save(jpeg, 'JPEG')

# predict on new data
def predict(observation, tree):
    if tree.results is not None:
        return tree.results
    else:
        v = observation[tree.col]
        branch = None
        if isinstance(v, int) or isinstance(v, float):
            if v >= tree.value:
                branch = tree.tb
            else:
                branch = tree.fb
        else:
            if v == tree.value:
                branch = tree.tb
            else:
                branch = tree.fb
        return predict(observation, branch)

# prune for tree to avoid over-fitting
def prune(tree, min_gain):
    if tree.tb.results is None:
        prune(tree.tb, min_gain)
    if tree.tb.results is None:
        predict(tree.fb, min_gain)

    if tree.tb.results is not None and tree.fb.results is not None:
        # merge the sub-tree
        tb, fb = [], []
        for v, c in tree.tb.results.items():
            tb += [[v]] * c
        for v, c in tree.tb.results.items():
            fb += [[v]] * c
                      
        delta = entropy(tb + fb) - (entropy(tb) + entropy(fb))/2
        if delta < min_gain:
            tree.tb, tree.fb = None, None
            tree.results = unique_counts(tb + fb)



