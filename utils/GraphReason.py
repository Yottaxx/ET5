import torch.nn as nn
from torch.autograd import Variable
from torch.nn.init import xavier_uniform_
import dgl
from transformers import ElectraModel, ElectraConfig

from utils.gcn import RGCNModel
import torch
import math


class MHA(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.attention_head_size = (config.hidden_size) // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1,
                         3)  # (batch_size, seq_len, all_head_size) -> (batch_size, num_attention_heads, seq_len, attention_head_size)

    def forward(self, input_ids_a, input_ids_b, attention_mask=None, head_mask=None, output_attentions=False):
        mixed_query_layer = self.query(input_ids_a)
        mixed_key_layer = self.key(input_ids_b)
        mixed_value_layer = self.value(input_ids_b)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        # (batch_size * num_choice, num_attention_heads, seq_len, attention_head_size) -> (batch_size * num_choice, seq_len, num_attention_heads, attention_head_size)

        # Should find a better way to do this
        w = (
            self.dense.weight.t()
                .view(self.num_attention_heads, self.attention_head_size, self.hidden_size)
                .to(context_layer.dtype)
        )
        b = self.dense.bias.to(context_layer.dtype)

        projected_context_layer = torch.einsum("bfnd,ndh->bfh", context_layer, w) + b
        projected_context_layer_dropout = self.dropout(projected_context_layer)
        layernormed_context_layer = self.LayerNorm(input_ids_a + projected_context_layer_dropout)
        return (layernormed_context_layer, attention_probs) if output_attentions else (layernormed_context_layer,)


class FuseLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.linear1 = nn.Linear(4 * config.d_model, config.d_model)
        self.linear2 = nn.Linear(4 * config.d_model,config.d_model)
        self.linear3 = nn.Linear(2 * config.d_model, config.d_model)
        self.activation = nn.ReLU()
        self.gate = nn.Sigmoid()

    def forward(self, orig, input1, input2):
        out1 = self.activation(self.linear1(torch.cat([orig, input1, orig - input1, orig * input1], dim=-1)))
        out2 = self.activation(self.linear2(torch.cat([orig, input2, orig - input2, orig * input2], dim=-1)))
        fuse_prob = self.gate(self.linear3(torch.cat([out1, out2], dim=-1)))

        return fuse_prob * input1 + (1 - fuse_prob) * input2



class transformerEntailmentWhole(nn.Module):
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
        self.num_decoupling = 1

        if config.d_model == 1024:
            electra_config = ElectraConfig.from_pretrained("google/electra-large-discriminator", cache_dir=None)
        else:
            electra_config = ElectraConfig.from_pretrained("google/electra-base-discriminator", cache_dir=None)

        self.fuse = FuseLayer(config)

        self.localMHA = nn.ModuleList([MHA(electra_config) for _ in range(self.num_decoupling)])
        self.globalMHA = nn.ModuleList([MHA(electra_config) for _ in range(self.num_decoupling)])

        self.transformer_encoder = transformer_encoder

        self.relation_embeds = nn.Embedding(18, config.d_model)
        self.edge_embeds = nn.Embedding(6, config.d_model)

        self.relation_embeds.weight.data.normal_(mean=0.0, std=config.initializer_factor*10)
        self.edge_embeds.weight.data.normal_(mean=0.0, std=config.initializer_factor*10)

        self.GCN = RGCNModel(config.d_model, 6, 1, True)

        self._reset_transformer_parameters()


    def _reset_transformer_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for name, param in self.named_parameters():
            if 'transformer' in name and param.dim() > 1:
                xavier_uniform_(param)

    def forward(self, input_ids,hidden_states, rule_idx, relationInput, user_idx, scenario, entailment_len):
        tenc_input, tenc_mask, tenc_input_gcn,tenc_input_rule = [], [], [],[]
        rule_mask = []
        userinfo_input, rule_input = [], []

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

            tenc_input_ = []
            tenc_input_global = []
            inp_ = user_idxTemp[0]
            inp = input_ids
            ruleidx = rule_idxTemp
            # construct mask matrix for multihead attention
            M1 = M2 = torch.zeros(inp_, inp_)

            for id_ in range(len(ruleidx) - 1):
                M1[ruleidx[id_]:ruleidx[id_ + 1], ruleidx[id_]:ruleidx[id_ + 1]] = 1.0
            M1[ruleidx[-1]:inp_, ruleidx[-1]:inp_] = 1.0

            M2 = 1.0 - M1
            M1 = (1.0 - M1) * -10000
            M2 = (1.0 - M2) * -10000

            M1 = M1.unsqueeze(0).unsqueeze(1)
            M2 = M2.unsqueeze(0).unsqueeze(1)

            s = [i for i in range(user_idxTemp[0])]
            s = torch.LongTensor(s)

            rule_selected = torch.index_select(hidden_states[idx], 0, s.to(hidden_states.device))
            rule_selected = rule_selected.unsqueeze(0)

            local_word_level = self.localMHA[0](rule_selected, rule_selected, attention_mask=M1.to(hidden_states.device))[0]
            global_word_level = self.globalMHA[0](rule_selected, rule_selected, attention_mask=M2.to(hidden_states.device))[0]

            for t in range(1, self.num_decoupling):
                local_word_level = \
                self.localMHA[t](local_word_level, local_word_level, attention_mask=M1.to(hidden_states.device))[0]
                global_word_level = \
                self.globalMHA[t](global_word_level, global_word_level, attention_mask=M2.to(hidden_states.device))[0]

            context_word_level = self.fuse(rule_selected, local_word_level, global_word_level)

            rule_input.append(
                torch.index_select(context_word_level.squeeze(0), 0, rule_idxTemp.to(hidden_states.device)))

            userinfo_input.append(torch.index_select(hidden_states[idx], 0, user_idxTemp.to(hidden_states.device)))

            for i in rule_input[-1]:
                tenc_input_.append(i)
            for j in userinfo_input[-1]:
                tenc_input_.append(j)
            tenc_input_rule.append(torch.Tensor([t.cpu().detach().numpy() for t in tenc_input_]))

            tenc_input.append(torch.index_select(hidden_states[idx], 0, tenc_idx))
            tenc_mask.append(torch.tensor([False] * tenc_idx.shape[0], dtype=torch.bool))
            rule_mask.append(torch.tensor([1] * rule_idxTemp.shape[0], dtype=torch.bool))


        tenc_input_gcn_padded = torch.nn.utils.rnn.pad_sequence(tenc_input_gcn).to(hidden_states.device)
        tenc_input_padded = torch.nn.utils.rnn.pad_sequence(tenc_input).to(hidden_states.device)  # [seqlen, N, dim]

        tenc_input_rule_padded = torch.nn.utils.rnn.pad_sequence(tenc_input_rule).to(hidden_states.device)

        tenc_mask_padded = torch.nn.utils.rnn.pad_sequence(tenc_mask, batch_first=True, padding_value=True).to(
            hidden_states.device)
        # tenc_out = self.transformer_encoder(tenc_input_padded, src_key_padding_mask=tenc_mask_padded)
        tenc_out_gcn = self.transformer_encoder(tenc_input_gcn_padded, src_key_padding_mask=tenc_mask_padded).transpose(0,1)

        tenc_out_rule = self.transformer_encoder(tenc_input_rule_padded, src_key_padding_mask=tenc_mask_padded).transpose(0,1)

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

        return tenc_out_gcn,tenc_out_rule
