# -*- coding:utf-8 -*-
from math import log2
from PIL import Image, ImageDraw

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
        self.results = results      # 分类结果，非叶子结点均为None
        self.tb = tb                # true
        self.fb = fb                # false


# 分割数据集
def divide_set(rows, column, value):
    # 根据value对数据进行2分类，set_1中为true, set_2中为false
    split_function = None
    if isinstance(value, int) or isinstance(value, float):
        split_function = lambda row: row[column] >= value
    else:
        split_function = lambda row: row[column] == value

    set_1 = [row for row in rows if split_function(row)]
    set_2 = [row for row in rows if not split_function(row)]

    return set_1, set_2


# 对分类结果进行计数
def unique_counts(rows):
    results = {}
    for row in rows:
        r = row[len(row) - 1]    # 分类结果:None, Basic, Premium
        if r not in results:
            results[r] = 0
        results[r] += 1
    return results


# 对一组数据计算交叉熵
def entropy(rows):
    results = unique_counts(rows)
    ent = 0.0
    for r in results.keys():
        p = float(results[r]) / len(rows)
        ent -= p * log2(p)
    return ent


def build_tree(rows, scoref=entropy):
    # 基准情况
    if len(rows) == 0:
        return DecisionNode()

    current_score = scoref(rows)      # 分类前的得分
    best_gain = 0.0
    best_criteria = None
    best_sets = None

    column_count = len(rows[0]) - 1    # 特征数量
    for col in range(column_count):
        # 在当前列中生成一个由不同值构成的序列
        column_values = {}
        for row in rows:
            column_values[row[col]] = 1
        # 分类
        for value in column_values.keys():
            set_1, set_2 = divide_set(rows, col, value)

            p = float(len(set_1)) / len(rows)
            gain = current_score - p * scoref(set_1) - (1 - p) * scoref(set_2)
            if gain > best_gain and len(set_1) > 0 and len(set_2) > 0:
                best_gain = gain
                best_criteria = (col, value)
                best_sets = (set_1, set_2)
        # 创建子分支
    if best_gain > 0:
        # 不是叶子结点，继续递归分类，分类结果res=None, 判断条件(特征)为col,临界值为value
        true_branch = build_tree(best_sets[0])
        false_branch = build_tree(best_sets[1])
        return DecisionNode(col=best_criteria[0], value=best_criteria[1], tb=true_branch, fb=false_branch)
    else:
        # 不能再分类，返回分类的计数结果
        return DecisionNode(results=unique_counts(rows))


def print_tree(tree, indent=''):
    # 叶子结点，其results为分类结果；否则，其results为None
    if tree.results is not None:
        print(str(tree.results))
    else:
        # 打印判断条件
        print(str(tree.col) + ':' + str(tree.value) + "?")
        # 打印分支
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
        # 得到每个分支的宽度
        w1 = get_width(tree.fb) * 100
        w2 = get_width(tree.tb) * 100

        # 确定此节点所要占据的总空间
        left = x - (w1 + w2) / 2
        right = x + (w1 + w2) / 2

        # 绘制判断条件字符串
        draw.text((x-20, y-10), str(tree.col) + ":" + str(tree.value), (0, 0, 0))

        # 绘制到分支的连线
        draw.line((x, y, left + w1/2, y + 100), fill=(255, 0, 0))
        draw.line((x, y, right - w2/2, y + 100), fill=(255, 0, 0))

        # 绘制分支的节点
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


def prune(tree, min_gain):
    # 如果分支不是叶节点，则对其进行剪枝操作
    if tree.tb.results is None:
        prune(tree.tb, min_gain)
    if tree.tb.results is None:
        predict(tree.fb, min_gain)

    # 如果两个子分支都是叶子节点，则判断它们是否需要合并
    if tree.tb.results is not None and tree.fb.results is not None:
        # 构造合并后的数据集
        tb, fb = [], []
        for v, c in tree.tb.results.items():
            tb += [[v]] * c
        for v, c in tree.tb.results.items():
            fb += [[v]] * c

        # 检查熵的减少情况
        delta = entropy(tb + fb) - (entropy(tb) + entropy(fb))/2
        if delta < min_gain:
            # 合并分支
            tree.tb, tree.fb = None, None
            tree.results = unique_counts(tb + fb)



