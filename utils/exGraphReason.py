import torch.nn as nn
from torch.autograd import Variable
from torch.nn.init import xavier_uniform_
import dgl
from utils.gcn import RGCNModel
import torch

class transformerEntailment(nn.Module):
    def __init__(self, config,transformer_encoder):
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(0.1)
        self.relation_key_pair = {'Comment': 0, 'Clarification_question': 1, 'Elaboration': 2, 'Acknowledgement': 3,
                                  'Continuation': 4, 'Explanation': 5, 'Conditional': 6, 'Question-answer_pair': 7,
                                  'Alternation': 8, 'Q-Elab': 9, 'Result': 10, 'Background': 11, 'Narration': 12,
                                  'Correction': 13, 'Parallel': 14, 'Contrast': 15}
        # if config.d_model == 1024:
        #     encoder_layer = TransformerEncoderLayer(config.d_model, 16, 4 * self.config.d_model)
        # else:
        #     encoder_layer = TransformerEncoderLayer(config.d_model, 12, 4 * self.config.d_model)
        #
        # encoder_norm = nn.LayerNorm(config.d_model)
        self.transformer_encoder = transformer_encoder

        self.relation_embeds = nn.Embedding(18, config.d_model)
        self.edge_embeds = nn.Embedding(6, config.d_model)

        self.GCN = RGCNModel(config.d_model, 6, 1, True)

        # self._reset_transformer_parameters()



    def _reset_transformer_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for name, param in self.named_parameters():
            if 'transformer' in name and param.dim() > 1:
                xavier_uniform_(param)

    def forward(self, input_ids,hidden_states, rule_idx, relationInput, user_idx, scenario, entailment_len):
        tenc_input, tenc_mask, tenc_input_gcn = [], [], []
        rule_mask = []

        for idx in range(len(rule_idx)):
            G = dgl.DGLGraph().to(hidden_states.device)
            relation = []
            edge_type = []  # in total six type of edges
            edges = []

            relationTemp = relationInput[idx]
            rule_idxTemp = rule_idx[idx]
            user_idxTemp = user_idx[idx]

            for item in relationTemp:
                if item['type'] not in relation:
                    relation.append(item['type'])
            G.add_nodes(rule_idxTemp.shape[0] + 1 + len(relation))  # total utterance nodes in the graph

            # Graph Construction
            for item in relationTemp:
                # add default_in and default_out edges
                G.add_edges(item['y'], relation.index(item['type']) + rule_idxTemp.shape[0] + 1)
                edge_type.append(0)
                edges.append([item['y'], relation.index(item['type']) + rule_idxTemp.shape[0] + 1])
                # G.edges[item['y'], relation.index(item['type'])+e['entail']['rule_idx'].shape[0]+1].data['rel_type'] = self.edge_embeds(Variable(torch.LongTensor([0,]).to(self.device)))
                G.add_edges(relation.index(item['type']) + rule_idxTemp.shape[0] + 1, item['x'])
                edge_type.append(1)
                edges.append([relation.index(item['type']) + rule_idxTemp.shape[0] + 1, item['x']])
                # G.edges[relation.index(item['type'])+e['entail']['rule_idx'].shape[0]+1, item['x']].data['rel_type'] = self.edge_embeds(Variable(torch.LongTensor([1,]).to(self.device)))

                # add reverse_out and reverse_in edges
                G.add_edges(relation.index(item['type']) + rule_idxTemp.shape[0] + 1, item['y'])
                edge_type.append(2)
                edges.append([relation.index(item['type']) + rule_idxTemp.shape[0] + 1, item['y']])
                # G.edges[relation.index(item['type'])+e['entail']['rule_idx'].shape[0]+1, item['y']].data['rel_type'] = self.edge_embeds(Variable(torch.LongTensor([2,]).to(self.device)))
                G.add_edges(item['x'], relation.index(item['type']) + rule_idxTemp.shape[0] + 1)
                edge_type.append(3)
                edges.append([item['x'], relation.index(item['type']) + rule_idxTemp.shape[0] + 1])
                # G.edges[item['x'], relation.index(item['type'])+e['entail']['rule_idx'].shape[0]+1].data['rel_type'] = self.edge_embeds(Variable(torch.LongTensor([3,]).to(self.device)))

            # add self edges
            for x in range(rule_idxTemp.shape[0] + 1 + len(relation)):
                G.add_edges(x, x)
                edge_type.append(4)
                edges.append([x, x])
                # G.edges[x,x].data['rel_type'] = self.edge_embeds(Variable(torch.LongTensor([4,]).to(self.device)))

            # add global edges
            for x in range(rule_idxTemp.shape[0] + 1 + len(relation)):
                if x != rule_idxTemp.shape[0]:
                    G.add_edges(rule_idxTemp.shape[0], x)
                    edge_type.append(5)
                    edges.append([rule_idxTemp.shape[0], x])
                    # G.edges[x,x].data['rel_type'] = self.edge_embeds(Variable(torch.LongTensor([5,]).to(self.device)))

            # add node feature
            for i in range(rule_idxTemp.shape[0] + 1 + len(relation)):
                if i < rule_idxTemp.shape[0]:
                    G.nodes[[i]].data['h'] = torch.index_select(hidden_states[idx], 0,
                                                                torch.LongTensor([rule_idxTemp[i], ]).to(
                                                                    hidden_states.device))
                elif i == rule_idxTemp.shape[0]:
                    if scenario[idx] != -1:
                        G.nodes[[i]].data['h'] = torch.index_select(hidden_states[idx], 0, torch.LongTensor(
                            [user_idxTemp[1], ]).to(hidden_states.device))
                    else:
                        G.nodes[[i]].data['h'] = self.relation_embeds(
                            Variable(torch.LongTensor([16, ]).to(hidden_states.device)))

                else:
                    index_relation = self.relation_key_pair[relation[i - rule_idxTemp.shape[0] - 1]]
                    G.nodes[[i]].data['h'] = self.relation_embeds(
                        Variable(torch.LongTensor([index_relation, ]).to(hidden_states.device)))

            edge_norm = []
            for e1, e2 in edges:
                if e1 == e2:
                    edge_norm.append(1)
                else:
                    edge_norm.append(1 / (G.in_degrees(e2) - 1))

            edge_type = torch.tensor(edge_type).to(hidden_states.device)
            edge_norm = torch.tensor(edge_norm).unsqueeze(1).float().to(hidden_states.device)
            G.edata.update({'rel_type': edge_type, })
            G.edata.update({'norm': edge_norm})
            X = self.GCN(G)[0]  # [bz, hdim]

            tenc_idx = torch.cat([rule_idxTemp, user_idxTemp], dim=-1).to(hidden_states.device)
            gcn_user = torch.index_select(hidden_states[idx], 0, user_idxTemp.to(hidden_states.device))
            gcn_rule_idx = torch.LongTensor([i for i in range(rule_idxTemp.shape[0])]).to(hidden_states.device)
            gcn_rule = torch.index_select(X, 0, gcn_rule_idx)
            tenc_input_gcn.append(torch.cat([gcn_rule, gcn_user], dim=0))

            tenc_input.append(torch.index_select(hidden_states[idx], 0, tenc_idx))
            tenc_mask.append(torch.tensor([False] * tenc_idx.shape[0], dtype=torch.bool))
            rule_mask.append(torch.tensor([1] * rule_idxTemp.shape[0], dtype=torch.bool))

        tenc_input_gcn_padded = torch.nn.utils.rnn.pad_sequence(tenc_input_gcn).to(hidden_states.device)
        tenc_input_padded = torch.nn.utils.rnn.pad_sequence(tenc_input).to(hidden_states.device)  # [seqlen, N, dim]
        tenc_mask_padded = torch.nn.utils.rnn.pad_sequence(tenc_mask, batch_first=True, padding_value=True).to(
            hidden_states.device)
        # tenc_out = self.transformer_encoder(tenc_input_padded, src_key_padding_mask=tenc_mask_padded)
        tenc_out_gcn = self.transformer_encoder(tenc_input_gcn_padded, src_key_padding_mask=tenc_mask_padded).transpose(0,1)

        # entailment_hidden = torch.transpose(tenc_out + tenc_out_gcn, 0, 1).contiguous()  # [bz * seqlen * dim]

        # entailment_hidden = self.transformer_encoder(
        #     self.dropout(
        #         hidden_states.masked_select(
        #             entailment_mask.unsqueeze(dim=-1).bool()
        #         ).reshape(hidden_states.shape[0], -1, hidden_states.shape[-1]).transpose(0, 1)
        #     )
        #     , src_key_padding_mask=edu_attention_mask
        # ).transpose(0, 1)

        # max_num_entailment_in_batch = max(entailment_len)
        #
        # entail_score_mask = nn.utils.rnn.pad_sequence(rule_mask, batch_first=True).unsqueeze(-1).to(
        #     hidden_states.device)  # [bz * seqlen * 1]
        # max_num_rule_in_batch = entail_score_mask.shape[1]
        #
        # assert max_num_rule_in_batch == max_num_entailment_in_batch
        # entailment_score = self.entailment_classify(
        #     self.dropout(entailment_hidden[:, :max_num_entailment_in_batch, :])
        # )
        #
        # # entail_score_mask = rule_mask.unsqueeze(-1)
        #
        # entail_state = torch.matmul(entailment_score, self.entail_emb)
        # cat_state = torch.cat([entail_state, entailment_hidden[:, :max_num_entailment_in_batch, :]], dim=-1)
        #
        # selfattn_unmask = self.w_selfattn(self.dropout(cat_state))
        # # print("----------------")
        # # print("max_num_entailment_in_batch:", max_num_entailment_in_batch)
        # # print("cat_state", cat_state.shape)
        # # print("entailment_label", entailment_label.shape)
        # # print("rule_mask", rule_mask.shape)
        #
        # selfattn_unmask.masked_fill_(~entail_score_mask, -1e9)
        # selfattn_weight = F.softmax(selfattn_unmask, dim=1)
        # selfattn = torch.sum(selfattn_weight * cat_state, dim=1)
        # score = self.w_output(self.dropout(selfattn))

        return tenc_out_gcn
