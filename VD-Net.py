
class cls_model(nn.Module):
    def __init__(self, vec_size):
        super(cls_model, self).__init__()
        self.vec_size = vec_size
        self.obj_linear1 = Linear(self.vec_size, 60)
        self.obj_linear2 = Linear(70, 240)

        self.sub_linear1 = Linear(self.vec_size, 60)
        self.sub_linear2 = Linear(70, 240)

        self.rela_linear1 = Linear(480+22, 400)
        self.rela_linear2 = Linear(400, rela_set_num, bias=False)
        self.drop = 0.001
        self.bn_norm = nn.BatchNorm1d(480+22, momentum=0.5)

    def forward(self, x, training=True):
        obj = x[:, 0, :self.vec_size]
        obj_pos = x[:, 0, self.vec_size:]
        sub = x[:, 1, :self.vec_size]
        sub_pos = x[:, 1, self.vec_size:]
        pos = x[:, 2, -22:]
        obj_l1 = F.dropout(self.obj_linear1(obj), p=self.drop, training=training)
        obj_l2 = F.dropout(self.obj_linear2(torch.cat([obj_l1, obj_pos], 1)), p=self.drop, training=training)

        sub_l1 = F.dropout(self.sub_linear1(sub), p=self.drop, training=training)
        sub_l2 = F.dropout(self.sub_linear2(torch.cat([sub_l1, sub_pos], 1)), p=self.drop, training=training)

        rela_in = torch.cat([obj_l2, sub_l2, pos], 1)
        rela_in = self.bn_norm(rela_in)
        rela_l1 = self.rela_linear1(rela_in)
        rela_l2 = self.rela_linear2(rela_l1)
        return rela_l2