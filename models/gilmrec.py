# coding: utf-8

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.abstract_recommender import GeneralRecommender


class CrossAttn(nn.Module):
    def __init__(self, embedding_dim):
        super(CrossAttn, self).__init__()
        self.embedding_dim = embedding_dim
        self.Wq = nn.Linear(embedding_dim, embedding_dim)
        self.Wk = nn.Linear(embedding_dim, embedding_dim)

    def cross_attention(self, query_X, support_X):
        Q = self.Wq(query_X)  # query
        K = self.Wk(support_X)  # key
        attention_scores = torch.matmul(Q, K.transpose(0, 1)) / torch.sqrt(
            torch.tensor(self.embedding_dim, dtype=torch.float32))
        attention_weights = torch.sum(attention_scores,dim=1)

        min_val = attention_weights.min()
        max_val = attention_weights.max()
        attn = (attention_weights - min_val) / (max_val - min_val)

        return attn

class GILMRec(GeneralRecommender):
    def __init__(self, config, dataset):
        super(GILMRec, self).__init__(config, dataset)

        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.n_mm_layer = config['n_mm_layers']
        self.config=config
        self.n_ui_layers = config['n_ui_layers']
        self.alpha = config['alpha']
        self.beta = config['beta']
        self.reg_weight = config['reg_weight']
        self.invariant_mix = "mean"
        self.topt = config['topt']
        self.n_nodes = self.n_users + self.n_items
        self.cf_model = config['cf_model']
        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.adj = self.scipy_matrix_to_sparse_tenser(self.interaction_matrix, torch.Size((self.n_users, self.n_items)))
        self.num_inters, self.norm_adj = self.get_norm_adj_mat()
        self.num_inters = torch.FloatTensor(1.0 / (self.num_inters + 1e-7)).to(self.device)

        # init user and item ID embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        # 加载物品的模态特征
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=True)
            self.item_image_trs = nn.Parameter(nn.init.xavier_uniform_(
                torch.zeros(self.v_feat.shape[1], self.feat_embed_dim)))
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=True)
            self.item_text_trs = nn.Parameter(nn.init.xavier_uniform_(
                torch.zeros(self.t_feat.shape[1], self.feat_embed_dim)))
        # self.mlp=torch
        # 计算物品相似度
        self.v_matrix = self.get_modal_sim(self.image_embedding.weight)
        self.t_matrix = self.get_modal_sim(self.text_embedding.weight)
        # self.v_matrix_norm=F.softmax(self.v_matrix, dim=1)
        # self.t_matrix_norm=F.softmax(self.t_matrix, dim=1)
        self.image_invariant_matrix, self.text_invariant_matrix = self.get_invariant_matrix(self.v_matrix,
                                                                                            self.t_matrix, self.topt)

        # self.attn_1 = CrossAttn(self.embedding_dim)
        # self.attn_2 = CrossAttn(self.embedding_dim)
        # self.embs_layer = nn.Linear(192, 64)

    def get_modal_sim(self, item_feature):

        item_feature = item_feature.half()
        normed_feature = F.normalize(item_feature, dim=1)
        modal_sim_A = torch.mm(normed_feature, normed_feature.T)
        return modal_sim_A

    def get_invariant_matrix(self, image_sim_A, text_sim_A, k):


        image_invariant_matrix = torch.topk(image_sim_A, k, dim=1).indices
        text_invariant_matrix = torch.topk(text_sim_A, k, dim=1).indices

        return image_invariant_matrix, text_invariant_matrix


    def scipy_matrix_to_sparse_tenser(self, matrix, shape):
        row = matrix.row
        col = matrix.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(matrix.data)
        return torch.sparse.FloatTensor(i, data, shape).to(self.device)

    def get_norm_adj_mat(self):
        A = sp.dok_matrix((self.n_nodes, self.n_nodes), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()

        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        A._update(data_dict)

        sumArr = (A > 0).sum(axis=1)

        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D

        L = sp.coo_matrix(L)
        return sumArr, self.scipy_matrix_to_sparse_tenser(L, torch.Size(
            (self.n_nodes, self.n_nodes)))


    def cge(self):
        if self.cf_model == 'mf':

            cge_embs = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
        if self.cf_model == 'lightgcn':
            ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
            cge_embs = [ego_embeddings]
            for _ in range(self.n_ui_layers):
                ego_embeddings = torch.sparse.mm(self.norm_adj,
                                                 ego_embeddings)
                cge_embs += [ego_embeddings]
            cge_embs = torch.stack(cge_embs, dim=1)
            cge_embs = cge_embs.mean(dim=1, keepdim=False)
        return cge_embs


    def mge(self, str='v'):
        if str == 'v':
            item_feats = torch.mm(self.image_embedding.weight,
                                  self.item_image_trs)
        elif str == 't':
            item_feats = torch.mm(self.text_embedding.weight,
                                  self.item_text_trs)
        user_feats = torch.sparse.mm(self.adj, item_feats) * self.num_inters[
                                                             :self.n_users]
        mge_feats = torch.concat([user_feats, item_feats], dim=0)
        for _ in range(self.n_mm_layer):
            mge_feats = torch.sparse.mm(self.norm_adj,
                                        mge_feats)
        return mge_feats, item_feats

    def forward(self):

        cge_embs = self.cge()

        if self.v_feat is not None and self.t_feat is not None:
            v_mge_feats, v_feats = self.mge('v')
            t_mge_feats, t_feats = self.mge('t')
            mg_embs = F.normalize(v_mge_feats) + F.normalize(t_mge_feats)

        u_embs, i_embs = torch.split(cge_embs + mg_embs, [self.n_users, self.n_items], dim=0)


        if self.invariant_mix == "mean":
            image_invariant_embs = v_feats[self.image_invariant_matrix].mean(dim=1)
            text_invariant_embs = t_feats[self.text_invariant_matrix].mean(dim=1)
        elif self.invariant_mix == "median":  # 中位数
            image_invariant_embs = v_feats[self.image_invariant_matrix].median(dim=1).values
            text_invariant_embs = t_feats[self.text_invariant_matrix].median(dim=1).values
        elif self.invariant_mix == "mode":  # 众数
            image_invariant_embs = v_feats[self.image_invariant_matrix].mode(dim=1).values
            text_invariant_embs = t_feats[self.text_invariant_matrix].mode(dim=1).values
        elif self.invariant_mix == "max":
            image_invariant_embs = v_feats[self.image_invariant_matrix].max(dim=1).values
            text_invariant_embs = t_feats[self.text_invariant_matrix].max(dim=1).values
        elif self.invariant_mix == "min":
            image_invariant_embs = v_feats[self.image_invariant_matrix].min(dim=1).values
            text_invariant_embs = t_feats[self.text_invariant_matrix].min(dim=1).values

        else:
            pass

       #去topt
        # v_invariant_embs =torch.matmul(self.v_matrix_norm,v_feats)
        # t_invariant_embs =torch.matmul(self.t_matrix_norm,t_feats)

        # alph = self.attn_1.cross_attention(image_invariant_embs,image_invariant_embs)
        # beta = self.attn_2.cross_attention(text_invariant_embs,text_invariant_embs)

        if self.config['mod'] == 'non':
            pass
        else:
            i_embs = ((F.normalize(image_invariant_embs) + F.normalize(text_invariant_embs)) * self.alpha + i_embs) / 2

            # i_embs = ((F.normalize(v_invariant_embs) + F.normalize(t_invariant_embs) )* self.alpha + i_embs) / 2
            # i_embs = (F.normalize(torch.mm(alph ,image_invariant_embs)) + F.normalize(torch.mm(beta,text_invariant_embs))+ i_embs)/2


        self.u_embs = u_embs
        self.i_embs = i_embs
        return u_embs, i_embs

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)
        bpr_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        return bpr_loss

    def reg_loss(self, *embs):
        reg_loss = 0
        for emb in embs:
            reg_loss += torch.norm(emb, p=2)
        reg_loss /= embs[-1].shape[0]
        return reg_loss

    def calculate_loss(self, interaction):
        ua_embeddings, ia_embeddings = self.forward()

        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]

        batch_bpr_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)

        batch_reg_loss = self.reg_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)

        loss = batch_bpr_loss + self.reg_weight * batch_reg_loss
        return loss


    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_embs, item_embs = self.forward()
        scores = torch.matmul(user_embs[user], item_embs.T)
        return scores

    def predict_all(self):
        user_embs, item_embs = self.forward()
        return item_embs,user_embs
