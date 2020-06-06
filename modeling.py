'''
# Code inspired from:
    # https://github.com/pengshuang/Text-Similarity/blob/master/models/ESIM.py
'''
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
class ESIM(nn.Module):

    def __init__(self,args,lan_1_vocab,lan_2_vocab):
        super(ESIM, self).__init__()
        self.args =args
        self.dropout=args.dropout
        self.hidden_size = args.hidden_size
        self.embeds_dim = args.embeds_dim
        self.lan_1_vocab = lan_1_vocab
        self.lan_2_vocab = lan_2_vocab
        self.embeds_1 = nn.Embedding(lan_1_vocab,args.embeds_dim)
        self.embeds_2 = nn.Embedding(lan_2_vocab,args.embeds_dim)
        self.bn_embeds = nn.BatchNorm1d(args.embeds_dim)
        self.lstm_lan_1 = nn.LSTM(args.embeds_dim, args.hidden_size, batch_first=True, bidirectional=True)
        self.lstm_lan_2 = nn.LSTM(args.embeds_dim, args.hidden_size, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(args.hidden_size*8, args.hidden_size, batch_first=True, bidirectional=True)

        self.fc = nn.Sequential(
            nn.BatchNorm1d(args.hidden_size * 8),
            nn.Linear(args.hidden_size * 8, args.linear_size),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(args.linear_size),
            nn.Dropout(args.dropout),
            nn.Linear(args.linear_size, args.linear_size),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(args.linear_size),
            nn.Dropout(args.dropout),
            nn.Linear(args.linear_size, 3),
            nn.Softmax(dim=-1)
        )
        #self.loss = nn.CrossEntropyLoss(reduction='none')
    # Code inspired from:
    # https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py.
    def replace_masked(self,tensor, mask, value):
        """
        Replace the all the values of vectors in 'tensor' that are masked in
        'masked' by 'value'.

        Args:
            tensor: The tensor in which the masked vectors must have their values
                replaced.
            mask: A mask indicating the vectors which must have their values
                replaced.
            value: The value to place in the masked vectors of 'tensor'.

        Returns:
            A new tensor of the same size as 'tensor' where the values of the
            vectors masked in 'mask' were replaced by 'value'.
        """
        mask = mask.unsqueeze(1).transpose(2, 1)
        reverse_mask = 1.0 - mask
        values_to_add = value * reverse_mask
        return tensor * mask + values_to_add
    def soft_attention_align(self, x1, x2, mask1,mask2):
        '''
        x1: batch_size * seq_len * dim
        x2: batch_size * seq_len * dim
        '''
        # attention: batch_size * seq_len * seq_len
        attention = torch.matmul(x1, x2.transpose(1, 2))
        mask1 = mask1.float().masked_fill_(mask1, float('-inf'))
        mask2 = mask2.float().masked_fill_(mask2, float('-inf'))

        #weight
        weight1 = F.softmax(attention + mask2.unsqueeze(1), dim=-1)
        x1_align = torch.matmul(weight1, x2)
        weight2 = F.softmax(attention.transpose(1, 2) + mask1.unsqueeze(1), dim=-1)
        x2_align = torch.matmul(weight2, x1)

        # x_align: batch_size * seq_len * hidden_size
        return x1_align, x2_align


    # def apply_multiple(self, x):
    #     # input: batch_size * seq_len * (2 * hidden_size)
    #     p1 = F.avg_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
    #     p2 = F.max_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
    #     # output: batch_size * (4 * hidden_size)
    #     return torch.cat([p1, p2], 1)

    def apply_multiple(self, v_ai, premises_mask):
        # input: batch_size * seq_len * (2 * hidden_size)
        p1 = torch.sum(v_ai * premises_mask.unsqueeze(1)
                                                .transpose(2, 1), dim=1)\
            / torch.sum(premises_mask, dim=1, keepdim=True)
        p2 , _ = self.replace_masked(v_ai, premises_mask, -1e7).max(dim=1)
        # p1 = F.avg_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
        # p2 = F.max_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
        # output: batch_size * (4 * hidden_size)
        return torch.cat([p1, p2], 1)

    def submul(self, x1, x2):
        mul = x1 * x2
        sub = x1 - x2
        return torch.cat([sub, mul], -1)

    def forward(self, input_a_id, input_b_id,input_a_mask,input_b_mask,input_a_length,input_b_length,label):
        # batch_size * seq_len
        sent1, sent2 = input_a_id, input_b_id
        mask1, mask2 = input_a_mask.eq(0), input_b_mask.eq(0)
        sent1_length = input_a_length
        sent2_length = input_b_length
        labels = label
        # embeds: batch_size * seq_len => batch_size * seq_len * embeds_dim
        x1 = self.bn_embeds(self.embeds_1(sent1).transpose(1, 2).contiguous()).transpose(1, 2)
        x2 = self.bn_embeds(self.embeds_2(sent2).transpose(1, 2).contiguous()).transpose(1, 2)

        # sort the sentence
        sorted_sent1_lengths, indices_1 = torch.sort(sent1_length, descending=True)
        _, desorted_indices_1 = torch.sort(indices_1, descending=False)

        sorted_sent2_lengths, indices_2 = torch.sort(sent2_length, descending=True)
        _, desorted_indices_2 = torch.sort(indices_2, descending=False)
        x1 = x1[indices_1]
        x2 = x2[indices_2]
        # print(x1.size())
        # print(x2.size())
        x1 = pack_padded_sequence(x1,sorted_sent1_lengths,batch_first=True)
        x2 = pack_padded_sequence(x2,sorted_sent2_lengths,batch_first=True)

        # batch_size * seq_len * dim => batch_size * seq_len * 2*hidden_size
        o1, _ = self.lstm_lan_1(x1)
        o2, _ = self.lstm_lan_2(x2)

        o1 = pad_packed_sequence(o1, batch_first=True)[0]
        o2 = pad_packed_sequence(o2, batch_first=True)[0]
        o1 = o1[desorted_indices_1]
        o2 = o2[desorted_indices_2]
        # print(o1.size())
        # print(o2.size())
        max_sent1_length = o1.size()[1]
        max_sent2_length = o2.size()[1]
        #we need to cut the mask to makesure mask2 is the same size with attention
        mask1 = mask1[:,:max_sent1_length]
        mask2 = mask2[:,:max_sent2_length]
        input_a_mask = input_a_mask[:,:max_sent1_length]
        input_b_mask = input_b_mask[:,:max_sent2_length]
        # print(mask1.size())
        # print(mask2.size())
        # Attention
        # batch_size * seq_len * hidden_size
        q1_align, q2_align = self.soft_attention_align(o1, o2, mask1, mask2)

        # Compose
        # batch_size * seq_len * (8 * hidden_size)
        q1_combined = torch.cat([o1, q1_align, self.submul(o1, q1_align)], -1)
        q2_combined = torch.cat([o2, q2_align, self.submul(o2, q2_align)], -1)

        q1_combined = q1_combined[indices_1]
        q2_combined = q2_combined[indices_2]
        q1_combined = pack_padded_sequence(q1_combined,sorted_sent1_lengths,batch_first=True)
        q2_combined = pack_padded_sequence(q2_combined,sorted_sent2_lengths,batch_first=True)
        # batch_size * seq_len * (2 * hidden_size)
        q1_compose, _ = self.lstm2(q1_combined)
        q2_compose, _ = self.lstm2(q2_combined)

        q1_compose = pad_packed_sequence(q1_compose, batch_first=True)[0]
        q2_compose = pad_packed_sequence(q2_compose, batch_first=True)[0]
        q1_compose = q1_compose[desorted_indices_1]
        q2_compose = q2_compose[desorted_indices_2]
        # Aggregate
        # input: batch_size * seq_len * (2 * hidden_size)
        # output: batch_size * (4 * hidden_size)
        q1_rep = self.apply_multiple(q1_compose,input_a_mask)
        q2_rep = self.apply_multiple(q2_compose,input_b_mask)
        #print(q1_rep)
        #print(q1_rep)
        # Classifier
        x = torch.cat([q1_rep, q2_rep], -1)

        similarity = self.fc(x)
        #similarity = torch.argmax(similarity, -1)
        #print(similarity)
        #print(labels)
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(similarity.view(-1,3), labels.view(-1))
        return loss, similarity

class ESIM2(nn.Module):

    def __init__(self,args,lan_vocab):
        super(ESIM2, self).__init__()
        self.args =args
        self.dropout=args.dropout
        self.hidden_size = args.hidden_size
        self.embeds_dim = args.embeds_dim
        self.lan_vocab = lan_vocab
        self.embeds = nn.Embedding(lan_vocab,args.embeds_dim)
        self.bn_embeds = nn.BatchNorm1d(args.embeds_dim)
        self.lstm1 = nn.LSTM(args.embeds_dim, args.hidden_size, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(args.hidden_size*8, args.hidden_size, batch_first=True, bidirectional=True)

        self.fc = nn.Sequential(
            nn.BatchNorm1d(args.hidden_size * 8),
            nn.Linear(args.hidden_size * 8, args.linear_size),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(args.linear_size),
            nn.Dropout(args.dropout),
            nn.Linear(args.linear_size, args.linear_size),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(args.linear_size),
            nn.Dropout(args.dropout),
            nn.Linear(args.linear_size, 3),
            nn.Softmax(dim=-1)
        )
        #self.loss = nn.CrossEntropyLoss(reduction='none')
    def soft_attention_align(self, x1, x2, mask1,mask2):
        '''
        x1: batch_size * seq_len * dim
        x2: batch_size * seq_len * dim
        '''
        # attention: batch_size * seq_len * seq_len
        attention = torch.matmul(x1, x2.transpose(1, 2))
        mask1 = mask1.float().masked_fill_(mask1, float('-inf'))
        mask2 = mask2.float().masked_fill_(mask2, float('-inf'))

        #weight
        weight1 = F.softmax(attention + mask2.unsqueeze(1), dim=-1)
        x1_align = torch.matmul(weight1, x2)
        weight2 = F.softmax(attention.transpose(1, 2) + mask1.unsqueeze(1), dim=-1)
        x2_align = torch.matmul(weight2, x1)

        # x_align: batch_size * seq_len * hidden_size
        return x1_align, x2_align

    # Code inspired from:
    # https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py.
    def replace_masked(self,tensor, mask, value):
        """
        Replace the all the values of vectors in 'tensor' that are masked in
        'masked' by 'value'.

        Args:
            tensor: The tensor in which the masked vectors must have their values
                replaced.
            mask: A mask indicating the vectors which must have their values
                replaced.
            value: The value to place in the masked vectors of 'tensor'.

        Returns:
            A new tensor of the same size as 'tensor' where the values of the
            vectors masked in 'mask' were replaced by 'value'.
        """
        mask = mask.unsqueeze(1).transpose(2, 1)
        reverse_mask = 1.0 - mask
        values_to_add = value * reverse_mask
        return tensor * mask + values_to_add

    def apply_multiple(self, v_ai, premises_mask):
        # input: batch_size * seq_len * (2 * hidden_size)
        p1 = torch.sum(v_ai * premises_mask.unsqueeze(1)
                                                .transpose(2, 1), dim=1)\
            / torch.sum(premises_mask, dim=1, keepdim=True)
        p2 , _ = self.replace_masked(v_ai, premises_mask, -1e7).max(dim=1)
        # p1 = F.avg_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
        # p2 = F.max_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
        # output: batch_size * (4 * hidden_size)
        return torch.cat([p1, p2], 1)


    def submul(self, x1, x2):
        mul = x1 * x2
        sub = x1 - x2
        return torch.cat([sub, mul], -1)

    def forward(self, input_a_id, input_b_id,input_a_mask,input_b_mask,input_a_length,input_b_length,label):
        # batch_size * seq_len
        sent1, sent2 = input_a_id, input_b_id
        mask1, mask2 = input_a_mask.eq(0), input_b_mask.eq(0)
        sent1_length = input_a_length
        sent2_length = input_b_length
        labels = label
        # embeds: batch_size * seq_len => batch_size * seq_len * embeds_dim
        x1 = self.bn_embeds(self.embeds(sent1).transpose(1, 2).contiguous()).transpose(1, 2)
        x2 = self.bn_embeds(self.embeds(sent2).transpose(1, 2).contiguous()).transpose(1, 2)
        # sort the sentence
        sorted_sent1_lengths, indices_1 = torch.sort(sent1_length, descending=True)
        _, desorted_indices_1 = torch.sort(indices_1, descending=False)

        sorted_sent2_lengths, indices_2 = torch.sort(sent2_length, descending=True)
        _, desorted_indices_2 = torch.sort(indices_2, descending=False)
        x1 = x1[indices_1]
        x2 = x2[indices_2]
        # print(x1.size())
        # print(x2.size())
        x1 = pack_padded_sequence(x1,sorted_sent1_lengths,batch_first=True)
        x2 = pack_padded_sequence(x2,sorted_sent2_lengths,batch_first=True)


        # batch_size * seq_len * dim => batch_size * seq_len * hidden_size
        o1, _ = self.lstm1(x1)
        o2, _ = self.lstm1(x2)

        o1 = pad_packed_sequence(o1, batch_first=True)[0]
        o2 = pad_packed_sequence(o2, batch_first=True)[0]
        o1 = o1[desorted_indices_1]
        o2 = o2[desorted_indices_2]
        # print(o1.size())
        # print(o2.size())
        max_sent1_length = o1.size()[1]
        max_sent2_length = o2.size()[1]
        #we need to cut the mask to makesure mask2 is the same size with attention
        mask1 = mask1[:,:max_sent1_length]
        mask2 = mask2[:,:max_sent2_length]
        input_a_mask = input_a_mask[:,:max_sent1_length]
        input_b_mask = input_b_mask[:,:max_sent2_length]
        # Attention
        # batch_size * seq_len * hidden_size
        q1_align, q2_align = self.soft_attention_align(o1, o2, mask1, mask2)

        # Compose
        # batch_size * seq_len * (8 * hidden_size)
        q1_combined = torch.cat([o1, q1_align, self.submul(o1, q1_align)], -1)
        q2_combined = torch.cat([o2, q2_align, self.submul(o2, q2_align)], -1)

        q1_combined = q1_combined[indices_1]
        q2_combined = q2_combined[indices_2]
        q1_combined = pack_padded_sequence(q1_combined,sorted_sent1_lengths,batch_first=True)
        q2_combined = pack_padded_sequence(q2_combined,sorted_sent2_lengths,batch_first=True)
        # batch_size * seq_len * (2 * hidden_size)
        q1_compose, _ = self.lstm2(q1_combined)
        q2_compose, _ = self.lstm2(q2_combined)
        #print(q1_compose.size())
        q1_compose = pad_packed_sequence(q1_compose, batch_first=True)[0]
        q2_compose = pad_packed_sequence(q2_compose, batch_first=True)[0]

        q1_compose = q1_compose[desorted_indices_1]
        q2_compose = q2_compose[desorted_indices_2]
        # Aggregate
        # input: batch_size * seq_len * (2 * hidden_size)
        # output: batch_size * (4 * hidden_size)
        q1_rep = self.apply_multiple(q1_compose,input_a_mask)
        q2_rep = self.apply_multiple(q2_compose,input_b_mask)


        # Classifier
        x = torch.cat([q1_rep, q2_rep], -1)
        similarity = self.fc(x)
        #similarity = torch.argmax(similarity, -1)
        #print(similarity)
        #print(labels)
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(similarity.view(-1,3), labels.view(-1))
        return loss, similarity