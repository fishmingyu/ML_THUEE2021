import numpy as np 
import pandas as pd 


class TreeNode(object):
    def __init__(self, ids = None, children = [], entropy = 0, depth = 0):
        self.ids = ids           # index of data in this node
        self.entropy = entropy   # 记录当前节点的熵
        self.depth = depth       # 深度
        self.split_attribute = None # 具体选择的attribute
        self.children = children # list of its child nodes
        self.order = None       # 在子数据中
        self.label = None       # label of node if it is a leaf

    def set_properties(self, split_attribute, order):
        self.split_attribute = split_attribute
        self.order = order

    def set_label(self, label):
        self.label = label


def entropy(freq):
    # 计算一个array中的熵
    freq_0 = freq[np.array(freq).nonzero()[0]] # 计算出现频率
    prob_0 = freq_0/float(freq_0.sum())
    return -np.sum(prob_0*np.log(prob_0))

class DecisionTreeID3(object):
    def __init__(self, data, charac_m, min_gain = 1e-8):
        self.root = None
        self.Ntrain = 0
        self.source = data[0]
        self.target = data[1]
        self.min_gain = min_gain
        self.charac_m = charac_m
    
    def fit(self):
        self.Ntrain = self.source.count()[0] # 要训练的输入数据量
        self.attributes = np.array(list(self.source)) # 总共数据的属性
        
        self.labels = self.target.unique() # 数据的label
        ids = [i for i in range(self.Ntrain)] # 数据的标号
        self.root = TreeNode(ids = ids, entropy = self.node_entropy(ids), depth = 0)
        queue = [self.root] # 构造根节点队列
        while queue:
            node = queue.pop() # 从队列中取出节点
            node.children = self._split(node)
            if not node.children: # 
                self._set_label(node)
            queue += node.children
                
    def node_entropy(self, ids):
        # 计算node中的熵
        if len(ids) == 0: return 0 
        # exit()
        freq = np.array(self.target[ids].value_counts())
        return entropy(freq)

    def _set_label(self, node):
        node.set_label(self.target.iloc[node.ids].mode()[0]) # 设置最多同样样本出现的label
    
    def _split(self, node): # 节点分裂
        ids = node.ids # 取出当前节点的id序列
        best_gain = 0
        best_splits = []
        best_attribute = None # 最好的attribute
        order = None # 记录节点的值所对应分裂的att的所有相关值
        sub_data = self.source.iloc[ids, :] # 子数据
        # sub_data = sub_data.reset_index(drop=True)
        attr = self.attributes[np.random.choice(len(self.attributes), self.charac_m, replace=False)].tolist()
        for i, att in enumerate(attr):
            data = self.source[att]
            values = data.iloc[ids].unique().tolist() # 取出attribute对应的values值，是一个unique的常量
            if len(values) == 1: continue # entropy = 0
            splits = []
            # 这里遍历所有值，判断子数据中具体哪些index的值满足和source中的value相同的值
            # 每一个split对应一种attribute中的一种value
            for val in values: 
                # 子数据中的数据id
                sub_ids = sub_data.index[sub_data[att] == val].tolist()
                if sub_ids != []:
                    splits.append(sub_ids)   
            # if min(map(len, splits)) < 1: continue
            
            # 计算当前attribute下的信息增益
            part_entropy = 0
            for split in splits:
                part_entropy += len(split)*self.node_entropy(split)/len(ids)
            gain = node.entropy - part_entropy 
            # 比较是否是最优的gain
            if gain > best_gain:
                best_gain = gain 
                best_splits = splits
                best_attribute = att
                order = values

        node.set_properties(best_attribute, order)
        child_nodes = [TreeNode(ids = split,
                     entropy = self.node_entropy(split), depth = node.depth + 1) for split in best_splits]
        return child_nodes

    def predict(self, new_data):
        npoints = new_data.count()[0] # 数据量
        labels = [None]*npoints
        for n in range(npoints):
            x = new_data.iloc[n, :] # one point 
            node = self.root
            found = True
            while node.children: 
                # 找到node中记录attribute的index
                id = None
                v = x[node.split_attribute]
                if v in node.order:
                    id = node.order.index(v)
                    node = node.children[id]
                else:# 找不到, out of bag
                    found = False
                    break
            if found:
                labels[n] = node.label
            else: # random set a target
                labels[n] = np.random.randint(0, 1, 1)[0]
        return labels
