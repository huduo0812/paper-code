import networkx as nx
import torch
import torch.nn as nn
import scipy.sparse as sp
import numpy as np
from pytorch_util import weights_init
from gcn import GCN
import torch.nn.functional as F
import copy
from policy_nn import PolicyNN
import gnns
import torch.optim as optim
from utils import progress_bar
import matplotlib.pyplot as plt


class gnn_explain():
    def __init__(self, max_node, max_step, target_class, max_iters):
        print('Start training pipeline')
        self.graph = nx.Graph()
        self.max_node = max_node
        self.max_step = max_step
        self.max_iters = max_iters
        self.num_class = 2
        self.node_type = 7
        self.learning_rate = 0.01
        self.roll_out_alpha = 2
        self.roll_out_penalty = -0.1
        self.policyNets = PolicyNN(self.node_type, self.node_type)
        self.gnnNets = gnns.DisNets()
        self.reward_stepwise = 0.1
        self.target_class = target_class
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.policyNets.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=5e-4)
        self.dict = {0: 'C', 1: 'N', 2: 'O', 3: 'F', 4: 'I', 5: 'Cl', 6: 'Br'}
        self.color = {0: 'g', 1: 'r', 2: 'b', 3: 'c', 4: 'm', 5: 'w', 6: 'y'}
        self.max_poss_degree = {0: 4, 1: 5, 2: 2, 3: 1, 4: 7, 5: 7, 6: 5}  # 不同节点label类型最大的度

    def train(self):
        # 加载基模型
        checkpoint = torch.load('./Is_Acyclic/ckpt.pth')
        self.gnnNets.load_state_dict(checkpoint['net'])
        print('GNN model:\n', self.gnnNets)

        for i in range(self.max_iters):
            self.graph_reset()  # 重置为只有一个label=0的节点的图
            for j in range(self.max_step):
                self.optimizer.zero_grad()  # 清空梯度
                reward_pred = 0
                reward_step = 0
                n = self.graph.number_of_nodes()  # 记录当前节点数
                if (n > self.max_node):
                    break
                self.graph_old = copy.deepcopy(self.graph)
                X, A = self.read_from_graph(self.graph)
                X = torch.from_numpy(X)
                A = torch.from_numpy(A)
                start_action, start_logits_ori, tail_action, tail_logits_ori = self.policyNets(X.float(), A.float(),
                                                                                               n + self.node_type)
                # 得到start节点，当前图节点概率，end节点，除start节点外当前+候选节概率

                # 当end节点不在当前图节点中时
                if (tail_action >= n):
                    if (n == self.max_node):
                        flag = False
                    else:
                        self.add_node(self.graph, n, tail_action.item() - n)  # 添加节点
                        flag = self.add_edge(self.graph, start_action.item(), n)  # 添加边
                # 当end节点在当前节点中时
                else:
                    flag = self.add_edge(self.graph, start_action.item(), tail_action.item())

                if flag == True:
                    # 检查当前的图是否满足图规则
                    validity = self.check_validity(self.graph)

                if flag == True:
                    if validity == True:  # 当前生成图有效
                        reward_step = self.reward_stepwise
                        X_new, A_new = self.read_from_graph_raw(self.graph)  # 返回以节点label类型为特征的one-hot矩阵，返回图的加权邻接矩阵
                        X_new = torch.from_numpy(X_new)
                        A_new = torch.from_numpy(A_new)
                        logits, probs = self.gnnNets(X_new.float(), A_new.float())  # 图分类的嵌入，图分类的概率
                        _, prediction = torch.max(logits, 0)  # 返回最大值的索引
                        # Rtf第一项中间奖励
                        # 0.5 是 1/l l为GNN总的类别
                        if self.target_class == prediction:  # 判断生成图预测类别是否为原图类别
                            reward_pred = probs[prediction] - 0.5  # 正反馈 预测相同，Rtf第一项
                        else:
                            reward_pred = probs[self.target_class] - 0.5  # 负反馈

                        # 计算最终图的损失
                        reward_rollout = []
                        for roll in range(10):  # 生成10个最终图，得到奖励或者惩罚
                            reward_cur = self.roll_out(self.graph, j)
                            reward_rollout.append(reward_cur)
                        reward_avg = torch.mean(torch.stack(reward_rollout))

                        # 设置Rtf奖励
                        total_reward = reward_step + reward_pred + reward_avg * self.roll_out_alpha

                        # 回滚
                        if total_reward < 0:
                            self.graph = copy.deepcopy(self.graph_old)
                        # 计算总损失
                        loss = total_reward * (self.criterion(start_logits_ori[None, :], start_action.expand(1))
                                               + self.criterion(tail_logits_ori[None, :], tail_action.expand(1)))
                    else:
                        total_reward = -1  # 违反规则，生成图无效
                        self.graph = copy.deepcopy(self.graph_old)
                        loss = total_reward * (self.criterion(start_logits_ori[None, :], start_action.expand(1))
                                               + self.criterion(tail_logits_ori[None, :], tail_action.expand(1)))
                else:
                    # 加入节点或者边失败， 或者违反图规则
                    reward_step = -1
                    total_reward = reward_step + reward_pred
                    loss = total_reward * (self.criterion(start_logits_ori[None, :], start_action.expand(1)) +
                                           self.criterion(tail_logits_ori[None, :], tail_action.expand(1)))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policyNets.parameters(), 100)
                self.optimizer.step()
        self.graph_draw(self.graph)  # 画出生成图
        plt.show()
        X_new, A_new = self.read_from_graph_raw(self.graph)
        X_new = torch.from_numpy(X_new)
        A_new = torch.from_numpy(A_new)
        logits, probs = self.gnnNets(X_new.float(), A_new.float())
        prob = probs[self.target_class].item()
        # print(prob) # 打印生成图目标类别的概率
        print(self.target_class)
        # print(" ")
        print(prob)

    # 画图
    def graph_draw(self, graph):
        attr = nx.get_node_attributes(graph, "label")
        labels = {}
        color = []
        for n in attr:
            labels[n] = self.dict[attr[n]]
            # color = color+ self.color[attr[n]]
            color.append(self.color[attr[n]])

        nx.draw(graph, labels=labels, node_color=color)

    # 图规则
    def check_validity(self, graph):
        node_types = nx.get_node_attributes(graph, 'label')  # 得到每个节点及对应类型dict
        for i in range(graph.number_of_nodes()):
            degree = graph.degree(i)  # 得到每个节点的度
            max_allow = self.max_poss_degree[node_types[i]]  # 得到该节点类型最大的度
            if (degree > max_allow):  # 超出最大的度则说明生成度图不具有有效性
                return False
        return True

    # 回滚函数
    def roll_out(self, graph, j):
        cur_graph = copy.deepcopy(graph)
        step = 0
        while (cur_graph.number_of_nodes() <= self.max_node and step < self.max_step - j):
            graph_old = copy.deepcopy(cur_graph)
            step = step + 1
            X, A = self.read_from_graph(cur_graph)
            n = cur_graph.number_of_nodes()
            X = torch.from_numpy(X)
            A = torch.from_numpy(A)
            start_action, start_logits_ori, tail_action, tail_logits_ori = self.policyNets(X.float(), A.float(),
                                                                                           n + self.node_type)
            if (tail_action >= n):
                if (n == self.max_node):
                    flag = False
                else:
                    self.add_node(cur_graph, n, tail_action.item() - n)  #
                    flag = self.add_edge(cur_graph, start_action.item(), n)
            else:
                flag = self.add_edge(cur_graph, start_action.item(), tail_action.item())

            if flag == True:  # 加边成功时
                validity = self.check_validity(cur_graph)  # 检查有效性
                if validity == False:  # 添加边无效时
                    return torch.tensor(self.roll_out_penalty)  # 返回一个负奖励作为这次回滚的总体奖励
            else:
                return torch.tensor(self.roll_out_penalty)  # 返回一个负奖励作为这次回滚的总体奖励

        X_new, A_new = self.read_from_graph_raw(cur_graph)
        X_new = torch.from_numpy(X_new)
        A_new = torch.from_numpy(A_new)
        logits, probs = self.gnnNets(X_new.float(), A_new.float())
        reward = probs[self.target_class] - 0.5
        return reward

    def add_node(self, graph, idx, node_type):
        graph.add_node(idx, label=node_type)
        return

    def add_edge(self, graph, start_id, tail_id):
        if graph.has_edge(start_id, tail_id) or start_id == tail_id:
            return False
        else:
            graph.add_edge(start_id, tail_id)
            return True

    # 处理合成的
    def read_from_graph(self, graph):
        n = graph.number_of_nodes()
        F = np.zeros((self.max_node + self.node_type, self.node_type))
        attr = nx.get_node_attributes(graph, "label")
        attr = list(attr.values())
        nb_clss = self.node_type
        targets = np.array(attr).reshape(-1)
        one_hot_feature = np.eye(nb_clss)[targets]
        F[:n, :] = one_hot_feature
        F[n:n + self.node_type, :] = np.eye(self.node_type)

        E = np.zeros([self.max_node + self.node_type, self.max_node + self.node_type])
        E[:n, :n] = np.asarray(nx.to_numpy_array(graph))
        E[:self.max_node + self.node_type, :self.max_node + self.node_type] += np.eye(self.max_node + self.node_type)
        return F, E

    # 处理未合成的
    def read_from_graph_raw(self, graph):
        # 返回只有节点类型的图的邻接矩阵和特征集
        n = graph.number_of_nodes()
        #  F = np.zeros((self.max_node+self.node_type, 1))
        attr = nx.get_node_attributes(graph, "label")
        attr = list(attr.values())  # 节点类型list
        nb_clss = self.node_type  # 原图节点类型数
        targets = np.array(attr).reshape(-1)
        one_hot_feature = np.eye(nb_clss)[targets]  # 返回一个大小为节点类型数的对角矩阵
        #  F[:n+1,0] = 1

        E = np.zeros([n, n])
        E[:n, :n] = np.asarray(nx.to_numpy_array(graph))  # 将图转化为加权邻接矩阵
        #   E[:n,:n] += np.eye(n)

        return one_hot_feature, E

    def graph_reset(self):
        self.graph.clear()
        self.graph.add_node(0, label=0)  # self.dict = {0:'C', 1:'N', 2:'O', 3:'F', 4:'I', 5:'Cl', 6:'Br'}
        self.step = 0
        return
