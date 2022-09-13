def getEdge(row):
    #上下左右方式连接
    idx1 = []
    idx2 = []
    for i in range(row):
        for j in range(row):
            #当前点为i*row+j
            #上
            if i-1 < 0 : pass
            else:
                idx1.append(i*row+j)
                idx2.append((i-1)*row+j)
            #下
            if i+1 > row-1:pass
            else:
                idx1.append(i*row+j)
                idx2.append((i+1)*row+j)
            #左
            if j-1 < 0: pass
            else:
                idx1.append(i*row+j)
                idx2.append(i*row + (j-1))
            #右
            if j+1 > row-1: pass
            else:
                idx1.append(i*row+j)
                idx2.append(i*row + (j+1))
            #左上
            if j-1 < 0 or i-1 < 0: pass
            else:
                idx1.append(i*row+j)
                idx2.append((i-1) * row + (j - 1))
            #左下
            if j-1 < 0 or i+1 > row-1: pass
            else:
                idx1.append(i*row+j)
                idx2.append((i+1) * row + (j - 1))
            #右上
            if j+1 > row-1 or i-1 < 0: pass
            else:
                idx1.append(i*row+j)
                idx2.append((i-1)*row + (j+1))
            #右下
            if j+1 > row-1 or i+1 > row-1: pass
            else:
                idx1.append(i*row+j)
                idx2.append((i+1)*row + (j+1))

    return [idx1,idx2]

