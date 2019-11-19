# for subject and object position embedding, the 10 dims embedding is [x1, x2, y1, y2, ins_w, ins_h, center_x / ins_w, center_y / ins_h, center_x / img_w, center_y / img_h],
# in the final version, we use the same setting as our paper ([x1, y1, ins_h, ins_w]) and padding the rest 6 dims as 0.

# for the joint embedding, the 22 dims version is 18 dims as in paper ([offset_Ox1, offset_Ox2, ...]) and [subject_x1, subject_y1, object_x1, object_y1].
# in the final version, we use the same setting as our paper and padding the rest dims with 0. 


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
