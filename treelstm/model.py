import torch
import torch.nn as nn

from . import Constants


# module forConstituencyTreeLSTM
# class ConstituencyTreeLSTM(nn.Module):
#     def __init__(self, in_dim, mem_dim):
#         super(ConstituencyTreeLSTM, self).__init__()
#         self.in_dim = in_dim
#         self.mem_dim = mem_dim
#         self.ioux = nn.Linear(self.in_dim, 3 * self.mem_dim)
#         # self.iouh = nn.Linear(self.mem_dim, 3 * self.mem_dim)
#         self.iouhL = nn.Linear(self.mem_dim, 3 * self.mem_dim)
#         self.iouhR = nn.Linear(self.mem_dim, 3 * self.mem_dim)
#         self.fx = nn.Linear(self.in_dim, self.mem_dim)
#         self.fhL = nn.Linear(self.mem_dim, self.mem_dim)
#         self.fhR = nn.Linear(self.mem_dim, self.mem_dim)
        
    # def node_forward(self, inputs, child_c, child_h):
    #     print(torch.sum(child_h,dim=0,keepDim=True))
        
        
    #     iou = self.ioux(inputs) + self.iouhL(child_hl) + self.iouhR(child_hr)

    #     i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
    #     i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)


    #     fL = torch.sigmoid(
    #         self.fhL(child_hl) +
    #         self.fx(inputs)
    #     )

    #     fR = torch.sigmoid(
    #         self.fhR(child_hr) +
    #         self.fx(inputs)
    #     )

    #     fcL = torch.mul(fL, child_c)
    #     fcR = torch.mul(fR, child_c)

    #     c = torch.mul(i, u) + fcL + fcR
    #     h = torch.mul(o, torch.tanh(c))
    #     return c, h

    # old child-sum tree, didn't change name cause lazy - daniel
class ConstituencyTreeLSTM(nn.Module):
    def __init__(self, in_dim, mem_dim):
        super(ConstituencyTreeLSTM, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.ioux = nn.Linear(self.in_dim, 3 * self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        self.iouhL = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        self.iouhR = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        self.fx = nn.Linear(self.in_dim, self.mem_dim)
        self.fhL = nn.Linear(self.mem_dim, self.mem_dim)
        self.fhR = nn.Linear(self.mem_dim, self.mem_dim)

        self.fh = nn.Linear(self.mem_dim, self.mem_dim)

    def leaf_forward(self, inputs, child_c, child_h):        
        child_h_sum = torch.sum(child_h, dim=0, keepdim=True)

        iou = self.ioux(inputs) + self.iouh(child_h_sum)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)

        f = torch.sigmoid(
            self.fh(child_h) +
            self.fx(inputs).repeat(len(child_h), 1)
        )
        fc = torch.mul(f, child_c)

        c = torch.mul(i, u) + torch.sum(fc, dim=0, keepdim=True)
        h = torch.mul(o, torch.tanh(c))
        return c, h

    def node_forward(self, inputs, children):
        print(children)
        if len(children) == 2: 
            childL_c = children[0][0]
            childL_h = children[0][1]
        
            childR_c = children[1][0]
            childR_h = children[1][1]
            
            iou = self.ioux(inputs) + self.iouhL(childL_h) + self.iouhR(childR_h)

            i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
            i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)


            fL = torch.sigmoid(
                self.fhL(childL_h) +
                self.fx(inputs)
            )

            fR = torch.sigmoid(
                self.fhR(childR_h) +
                self.fx(inputs)
            )

            fcL = torch.mul(fL, childL_c)
            fcR = torch.mul(fR, childR_c)

            c = torch.mul(i, u) + fcL + fcR
            h = torch.mul(o, torch.tanh(c))
            return c, h
        elif len(children) == 1:
            childL_c = children[0][0]
            childL_h = children[0][1]
            
            iou = self.ioux(inputs) + self.iouhL(childL_h)

            i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
            i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)


            fL = torch.sigmoid(
                self.fhL(childL_h) +
                self.fx(inputs)
            )


            fcL = torch.mul(fL, childL_c)

            c = torch.mul(i, u) + fcL
            h = torch.mul(o, torch.tanh(c))
            return c, h
        else:
            return None


    def forward(self, tree, inputs):   
        print("TREE SIZE:", tree.size())
        print("TREE INDEX:", tree.idx)
        print("NUM CHILDREN:", tree.num_children) 
        for child in tree.children:
            print("CHILD INDEX:", child.idx)
        assert(tree.num_children <= 2)

        for idx in range(tree.num_children):
            self.forward(tree.children[idx], inputs)
            
        if tree.num_children == 0:
            child_c = inputs[0].detach().new(1, self.mem_dim).fill_(0.).requires_grad_()
            child_h = inputs[0].detach().new(1, self.mem_dim).fill_(0.).requires_grad_()
            tree.state = self.leaf_forward(inputs[tree.idx], child_c, child_h)
        else:
            # get left child and right child separately
            # list of individual child (c,h)
            children =  list(map(lambda x: x.state, tree.children))
            # child_c, child_h = torch.cat(child_c, dim=0), torch.cat(child_h, dim=0)
            tree.state = self.node_forward(inputs[tree.idx], children)
        # print('INDEX: ', tree.idx)
        # print('STATE: ', tree.state)
        return tree.state


# module for distance-angle similarity
class Similarity(nn.Module):
    def __init__(self, mem_dim, hidden_dim, num_classes):
        super(Similarity, self).__init__()
        self.mem_dim = mem_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.wh = nn.Linear(2 * self.mem_dim, self.hidden_dim)
        self.wp = nn.Linear(self.hidden_dim, self.num_classes)

    def forward(self, lvec, rvec):
        mult_dist = torch.mul(lvec, rvec)
        abs_dist = torch.abs(torch.add(lvec, -rvec))
        vec_dist = torch.cat((mult_dist, abs_dist), 1)

        out = torch.sigmoid(self.wh(vec_dist))
        out = torch.log_softmax(self.wp(out), dim=1)
        return out


# putting the whole model together
class SimilarityTreeLSTM(nn.Module):
    def __init__(self, vocab_size, in_dim, mem_dim, hidden_dim, num_classes, sparsity, freeze):
        super(SimilarityTreeLSTM, self).__init__()
        self.emb = nn.Embedding(vocab_size, in_dim, padding_idx=Constants.PAD, sparse=sparsity)
        if freeze:
            self.emb.weight.requires_grad = False
        self.constituencytreelstm = ConstituencyTreeLSTM(in_dim, mem_dim)
        self.similarity = Similarity(mem_dim, hidden_dim, num_classes)

    def forward(self, ltree, linputs, rtree, rinputs):
        linputs = self.emb(linputs)
        rinputs = self.emb(rinputs)
        lstate, lhidden = self.constituencytreelstm(ltree, linputs)
        rstate, rhidden = self.constituencytreelstm(rtree, rinputs)
        output = self.similarity(lstate, rstate)
        return output
