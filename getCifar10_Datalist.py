from torch_geometric.data import Data
from cifar10 import *
from getEdge import *

from torch_geometric.data import Data

def getDatalist():
    train_datas, train_y, test_datas, test_y = getCifar10()
    e = torch.LongTensor(getEdge(32))

    #50000个训练数据
    #获得边32*32图对应的边，其中每个节点与对应周围最多八个点连接
    train_data_list = []
    for i in range(len(train_y)):
        x = torch.FloatTensor(train_datas[i])
        y = torch.LongTensor([train_y[i]])
        d = Data(x=x, edge_index=e, y=y)
        train_data_list.append(d)

    #10000个测试数据
    test_data_list = []
    for i in range(len(test_y)):
        x = torch.FloatTensor(test_datas[i])
        y = torch.LongTensor([test_y[i]])
        d = Data(x=x, edge_index=e, y=y)
        test_data_list.append(d)

    return train_data_list, test_data_list



