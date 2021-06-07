"""MADE and ResMADE."""
import time

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedLinear(nn.Linear):

    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer('mask', torch.ones(out_features, in_features))

        self.masked_weight = None

    def set_mask(self, mask):
        """Accepts a mask of shape [in_features, out_features]."""
        self.mask.data.copy_(torch.from_numpy(mask.astype(np.uint8).T))

    def forward(self, input):
        if self.masked_weight is None:
            return F.linear(input, self.mask * self.weight, self.bias)
        else:
            # ~17% speedup for Prog Sampling.
            return F.linear(input, self.masked_weight, self.bias)


class MaskedResidualBlock(nn.Module):

    def __init__(self, in_features, out_features, activation):
        assert in_features == out_features, [in_features, out_features]
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(MaskedLinear(in_features, out_features, bias=True))
        self.layers.append(MaskedLinear(in_features, out_features, bias=True))
        self.activation = activation

    def set_mask(self, mask):
        self.layers[0].set_mask(mask)
        self.layers[1].set_mask(mask)

    def forward(self, input):
        out = input
        out = self.activation(out)
        out = self.layers[0](out)
        out = self.activation(out)
        out = self.layers[1](out)
        return input + out


class MADE(nn.Module):

    def __init__(
            self,
            nin,
            hidden_sizes,
            nout,
            num_masks=1,
            natural_ordering=True,
            input_bins=None,
            activation=nn.ReLU,
            do_direct_io_connections=False,
            input_encoding=None,
            output_encoding='one_hot',
            embed_size=32,
            input_no_emb_if_leq=True,
            residual_connections=False,
            column_masking=False,
            seed=11123,
            fixed_ordering=None,
    ):
        """MADE.

        Args:
          nin: integer; number of input variables.  Each input variable
            represents a column.
          hidden sizes: a list of integers; number of units in hidden layers.
          nout: integer; number of outputs, the sum of all input variables'
            domain sizes.
          num_masks: number of orderings + connectivity masks to cycle through.
          natural_ordering: force natural ordering of dimensions, don't use
            random permutations.
          input_bins: classes each input var can take on, e.g., [5, 2] means
            input x1 has values in {0, ..., 4} and x2 in {0, 1}.  In other
            words, the domain sizes.
          activation: the activation to use.
          do_direct_io_connections: whether to add a connection from inputs to
            output layer.  Helpful for information flow.
          input_encoding: input encoding mode, see EncodeInput().
          output_encoding: output logits decoding mode, either 'embed' or
            'one_hot'.  See logits_for_col().
          embed_size: int, embedding dim.
          input_no_emb_if_leq: optimization, whether to turn off embedding for
            variables that have a domain size less than embed_size.  If so,
            those variables would have no learnable embeddings and instead are
            encoded as one hot vecs.
          residual_connections: use ResMADE?  Could lead to faster learning.
          column_masking: if True, turn on column masking during training time,
            which enables the wildcard skipping optimization during inference.
            Recommended to be set for any non-trivial datasets.
          seed: seed for generating random connectivity masks.
          fixed_ordering: variable ordering to use.  If specified, order[i]
            maps natural index i -> position in ordering.  E.g., if order[0] =
            2, variable 0 is placed at position 2.
        """
        super().__init__()
        print('fixed_ordering', fixed_ordering, 'seed', seed,
              'natural_ordering', natural_ordering)
        self.nin = nin
        assert input_encoding in [None, 'one_hot', 'binary', 'embed']
        self.input_encoding = input_encoding
        assert output_encoding in ['one_hot', 'embed']
        self.embed_size = self.emb_dim = embed_size
        self.output_encoding = output_encoding
        self.activation = activation
        self.nout = nout
        self.hidden_sizes = hidden_sizes
        self.input_bins = input_bins
        self.input_no_emb_if_leq = input_no_emb_if_leq
        self.do_direct_io_connections = do_direct_io_connections
        self.column_masking = column_masking
        self.residual_connections = residual_connections

        self.fixed_ordering = fixed_ordering
        if fixed_ordering is not None:
            assert num_masks == 1
            print('** Fixed ordering {} supplied, ignoring natural_ordering'.
                  format(fixed_ordering))

        assert self.input_bins
        encoded_bins = list(
            map(self._get_output_encoded_dist_size, self.input_bins))
        self.input_bins_encoded = list(
            map(self._get_input_encoded_dist_size, self.input_bins))
        self.input_bins_encoded_cumsum = np.cumsum(
            list(map(self._get_input_encoded_dist_size, self.input_bins)))
        print('encoded_bins (output)', encoded_bins)
        print('encoded_bins (input)', self.input_bins_encoded)

        hs = [nin] + hidden_sizes + [sum(encoded_bins)]
        self.net = []
        for h0, h1 in zip(hs, hs[1:]):
            if residual_connections:
                if h0 == h1:
                    self.net.extend([
                        MaskedResidualBlock(
                            h0, h1, activation=activation(inplace=False))
                    ])
                else:
                    self.net.extend([
                        MaskedLinear(h0, h1),
                    ])
            else:
                self.net.extend([
                    MaskedLinear(h0, h1),
                    activation(inplace=True),
                ])
        if not residual_connections:
            self.net.pop()
        self.net = nn.Sequential(*self.net)

        if self.input_encoding is not None:
            # Input layer should be changed.
            assert self.input_bins is not None
            input_size = 0
            for i, dist_size in enumerate(self.input_bins):
                input_size += self._get_input_encoded_dist_size(dist_size)
            new_layer0 = MaskedLinear(input_size, self.net[0].out_features)
            self.net[0] = new_layer0

        if self.output_encoding == 'embed':
            assert self.input_encoding == 'embed'

        if self.input_encoding == 'embed':
            self.embeddings = nn.ModuleList()
            for i, dist_size in enumerate(self.input_bins):
                if dist_size <= self.embed_size and self.input_no_emb_if_leq:
                    embed = None
                else:
                    embed = nn.Embedding(dist_size, self.embed_size)
                self.embeddings.append(embed)

        # Learnable [MASK] representation.
        if self.column_masking:
            self.unk_embeddings = nn.ParameterList()
            for i, dist_size in enumerate(self.input_bins):
                self.unk_embeddings.append(
                    nn.Parameter(torch.zeros(1, self.input_bins_encoded[i])))

        self.natural_ordering = natural_ordering
        self.num_masks = num_masks
        self.seed = seed if seed is not None else 11123
        self.init_seed = self.seed

        self.direct_io_layer = None
        self.logit_indices = np.cumsum(encoded_bins)
        self.m = {}

        self.update_masks()
        self.orderings = [self.m[-1]]

        # Optimization: cache some values needed in EncodeInput().
        self.bin_as_onehot_shifts = None

    def _build_or_update_direct_io(self):
        assert self.nout > self.nin and self.input_bins is not None
        direct_nin = self.net[0].in_features
        direct_nout = self.net[-1].out_features
        if self.direct_io_layer is None:
            self.direct_io_layer = MaskedLinear(direct_nin, direct_nout)
        mask = np.zeros((direct_nout, direct_nin), dtype=np.uint8)

        if self.natural_ordering:
            curr = 0
            for i in range(self.nin):
                dist_size = self._get_input_encoded_dist_size(
                    self.input_bins[i])
                # Input i connects to groups > i.
                mask[self.logit_indices[i]:, curr:dist_size] = 1
                curr += dist_size
        else:
            # Inverse: ord_idx -> natural idx.
            inv_ordering = [None] * self.nin
            for natural_idx in range(self.nin):
                inv_ordering[self.m[-1][natural_idx]] = natural_idx

            for ord_i in range(self.nin):
                nat_i = inv_ordering[ord_i]
                # x_(nat_i) in the input occupies range [inp_l, inp_r).
                inp_l = 0 if nat_i == 0 else self.input_bins_encoded_cumsum[
                    nat_i - 1]
                inp_r = self.input_bins_encoded_cumsum[nat_i]
                assert inp_l < inp_r

                for ord_j in range(ord_i + 1, self.nin):
                    nat_j = inv_ordering[ord_j]
                    # Output x_(nat_j) should connect to input x_(nat_i); it
                    # occupies range [out_l, out_r) in the output.
                    out_l = 0 if nat_j == 0 else self.logit_indices[nat_j - 1]
                    out_r = self.logit_indices[nat_j]
                    assert out_l < out_r
                    mask[out_l:out_r, inp_l:inp_r] = 1
        mask = mask.T
        self.direct_io_layer.set_mask(mask)

    def _get_input_encoded_dist_size(self, dist_size):
        if self.input_encoding == 'embed':
            if self.input_no_emb_if_leq:
                dist_size = min(dist_size, self.embed_size)
            else:
                dist_size = self.embed_size
        elif self.input_encoding == 'one_hot':
            pass
        elif self.input_encoding == 'binary':
            dist_size = max(1, int(np.ceil(np.log2(dist_size))))
        elif self.input_encoding is None:
            return 1
        else:
            assert False, self.input_encoding
        return dist_size

    def _get_output_encoded_dist_size(self, dist_size):
        if self.output_encoding == 'embed':
            if self.input_no_emb_if_leq:
                dist_size = min(dist_size, self.embed_size)
            else:
                dist_size = self.embed_size
        elif self.output_encoding == 'one_hot':
            pass
        elif self.output_encoding == 'binary':
            dist_size = max(1, int(np.ceil(np.log2(dist_size))))
        return dist_size

    def update_masks(self, invoke_order=None):
        """Update m() for all layers and change masks correspondingly.

        No-op if "self.num_masks" is 1.
        """
        if self.m and self.num_masks == 1:
            return
        L = len(self.hidden_sizes)

        ### Precedence of several params determining ordering:
        #
        # invoke_order
        # orderings
        # fixed_ordering
        # natural_ordering
        #
        # from high precedence to low.

        if invoke_order is not None:
            found = False
            for i in range(len(self.orderings)):
                if np.array_equal(self.orderings[i], invoke_order):
                    found = True
                    break
            assert found, 'specified={}, avail={}'.format(
                ordering, self.orderings)
            # orderings = [ o0, o1, o2, ... ]
            # seeds = [ init_seed, init_seed+1, init_seed+2, ... ]
            rng = np.random.RandomState(self.init_seed + i)
            self.seed = (self.init_seed + i + 1) % self.num_masks
            self.m[-1] = invoke_order
        elif hasattr(self, 'orderings'):
            # Cycle through the special orderings.
            rng = np.random.RandomState(self.seed)
            self.seed = (self.seed + 1) % self.num_masks
            self.m[-1] = self.orderings[self.seed % 4]
        else:
            rng = np.random.RandomState(self.seed)
            self.seed = (self.seed + 1) % self.num_masks
            self.m[-1] = np.arange(
                self.nin) if self.natural_ordering else rng.permutation(
                    self.nin)
            if self.fixed_ordering is not None:
                self.m[-1] = np.asarray(self.fixed_ordering)

        if self.nin > 1:
            for l in range(L):
                if self.residual_connections:
                    # Sequential assignment for ResMade: https://arxiv.org/pdf/1904.05626.pdf
                    self.m[l] = np.array([(k - 1) % (self.nin - 1)
                                          for k in range(self.hidden_sizes[l])])
                else:
                    # Samples from [0, ncols - 1).
                    self.m[l] = rng.randint(self.m[l - 1].min(),
                                            self.nin - 1,
                                            size=self.hidden_sizes[l])
        else:
            # This should result in first layer's masks == 0.
            # So output units are disconnected to any inputs.
            for l in range(L):
                self.m[l] = np.asarray([-1] * self.hidden_sizes[l])

        masks = [self.m[l - 1][:, None] <= self.m[l][None, :] for l in range(L)]
        masks.append(self.m[L - 1][:, None] < self.m[-1][None, :])

        if self.nout > self.nin:
            # Last layer's mask needs to be changed.

            if self.input_bins is None:
                k = int(self.nout / self.nin)
                # Replicate the mask across the other outputs
                # so [x1, x2, ..., xn], ..., [x1, x2, ..., xn].
                masks[-1] = np.concatenate([masks[-1]] * k, axis=1)
            else:
                # [x1, ..., x1], ..., [xn, ..., xn] where the i-th list has
                # input_bins[i - 1] many elements (multiplicity, # of classes).
                mask = np.asarray([])
                for k in range(masks[-1].shape[0]):
                    tmp_mask = []
                    for idx, x in enumerate(zip(masks[-1][k], self.input_bins)):
                        mval, nbins = x[0], self._get_output_encoded_dist_size(
                            x[1])
                        tmp_mask.extend([mval] * nbins)
                    tmp_mask = np.asarray(tmp_mask)
                    if k == 0:
                        mask = tmp_mask
                    else:
                        mask = np.vstack([mask, tmp_mask])
                masks[-1] = mask

        if self.input_encoding is not None:
            # Input layer's mask should be changed.

            assert self.input_bins is not None
            # [nin, hidden].
            mask0 = masks[0]
            new_mask0 = []
            for i, dist_size in enumerate(self.input_bins):
                dist_size = self._get_input_encoded_dist_size(dist_size)
                # [dist size, hidden]
                new_mask0.append(
                    np.concatenate([mask0[i].reshape(1, -1)] * dist_size,
                                   axis=0))
            # [sum(dist size), hidden]
            new_mask0 = np.vstack(new_mask0)
            masks[0] = new_mask0

        layers = [
            l for l in self.net if isinstance(l, MaskedLinear) or
            isinstance(l, MaskedResidualBlock)
        ]
        assert len(layers) == len(masks), (len(layers), len(masks))
        for l, m in zip(layers, masks):
            l.set_mask(m)

        if self.do_direct_io_connections:
            self._build_or_update_direct_io()

    def name(self):
        n = 'made'
        if self.residual_connections:
            n += '-resmade'
        n += '-hidden' + '_'.join(str(h) for h in self.hidden_sizes)
        n += '-emb' + str(self.embed_size)
        if self.num_masks > 1:
            n += '-{}masks'.format(self.num_masks)
        if not self.natural_ordering:
            n += '-nonNatural'
        n += ('-no' if not self.do_direct_io_connections else '-') + 'directIo'
        n += '-{}In{}Out'.format(self.input_encoding, self.output_encoding)
        if self.input_no_emb_if_leq:
            n += '-inputNoEmbIfLeq'
        if self.column_masking:
            n += '-colmask'
        return n

    def Embed(self, data, natural_col=None, out=None):
        if data is None:
            if out is None:
                return self.unk_embeddings[natural_col]
            out.copy_(self.unk_embeddings[natural_col])
            return out

        bs = data.size()[0]
        y_embed = []
        data = data.long()

        if natural_col is not None:
            # Fast path only for inference.  One col.

            coli_dom_size = self.input_bins[natural_col]
            # Embed?
            if coli_dom_size >= self.embed_size or not self.input_no_emb_if_leq:
                res = self.embeddings[natural_col](data.view(-1,))
                if out is not None:
                    out.copy_(res)
                    return out
                return res
            else:
                if out is None:
                    out = torch.zeros(bs, coli_dom_size, device=data.device)

                out.scatter_(1, data, 1)
                return out
        else:
            for i, coli_dom_size in enumerate(self.input_bins):
                # Wildcard column? use -1 as special token.
                # Inference pass only (see estimators.py).
                skip = data[0][i] < 0

                # Embed?
                if coli_dom_size >= self.embed_size or not self.input_no_emb_if_leq:
                    col_i_embs = self.embeddings[i](data[:, i])
                    if not self.column_masking:
                        y_embed.append(col_i_embs)
                    else:
                        dropped_repr = self.unk_embeddings[i]

                        def dropout_p():
                            return np.random.randint(0, self.nin) / self.nin

                        # During training, non-dropped 1's are scaled by
                        # 1/(1-p), so we clamp back to 1.
                        batch_mask = torch.clamp(
                            torch.dropout(torch.ones(bs, 1, device=data.device),
                                          p=dropout_p(),
                                          train=self.training), 0, 1)
                        y_embed.append(batch_mask * col_i_embs +
                                       (1. - batch_mask) * dropped_repr)
                else:
                    if skip:
                        y_embed.append(self.unk_embeddings[i])
                        continue
                    y_onehot = torch.zeros(bs,
                                           coli_dom_size,
                                           device=data.device)
                    y_onehot.scatter_(1, data[:, i].view(-1, 1), 1)
                    if self.column_masking:

                        def dropout_p():
                            return np.random.randint(0, self.nin) / self.nin

                        # During training, non-dropped 1's are scaled by
                        # 1/(1-p), so we clamp back to 1.
                        batch_mask = torch.clamp(
                            torch.dropout(torch.ones(bs, 1, device=data.device),
                                          p=dropout_p(),
                                          train=self.training), 0, 1)
                        y_embed.append(batch_mask * y_onehot +
                                       (1. - batch_mask) *
                                       self.unk_embeddings[i])
                    else:
                        y_embed.append(y_onehot)
            return torch.cat(y_embed, 1)

    def ToOneHot(self, data):
        assert not self.column_masking, 'not implemented'
        bs = data.size()[0]
        y_onehots = []
        data = data.long()
        for i, coli_dom_size in enumerate(self.input_bins):
            if coli_dom_size <= 2:
                y_onehots.append(data[:, i].view(-1, 1).float())
            else:
                y_onehot = torch.zeros(bs, coli_dom_size, device=data.device)
                y_onehot.scatter_(1, data[:, i].view(-1, 1), 1)
                y_onehots.append(y_onehot)

        # [bs, sum(dist size)]
        return torch.cat(y_onehots, 1)

    def ToBinaryAsOneHot(self, data, threshold=0, natural_col=None, out=None):
        if data is None:
            if out is None:
                return self.unk_embeddings[natural_col]
            out.copy_(self.unk_embeddings[natural_col])
            return out

        bs = data.size()[0]
        data = data.long()
        if self.bin_as_onehot_shifts is None:
            # This caching gives very sizable gains.
            self.bin_as_onehot_shifts = [None] * self.nin
            const_one = torch.ones([], dtype=torch.long, device=data.device)
            for i, coli_dom_size in enumerate(self.input_bins):
                # Max with 1 to guard against cols with 1 distinct val.
                one_hot_dims = max(1, int(np.ceil(np.log2(coli_dom_size))))
                self.bin_as_onehot_shifts[i] = const_one << torch.arange(
                    one_hot_dims, device=data.device)

        if natural_col is None:
            # Train path.

            assert out is None
            y_onehots = [None] * self.nin
            for i, coli_dom_size in enumerate(self.input_bins):
                if coli_dom_size > threshold:
                    # Bit shift in PyTorch + GPU is 27% faster than np.
                    data_np = data.narrow(1, i, 1)
                    binaries = (data_np & self.bin_as_onehot_shifts[i]) > 0
                    y_onehots[i] = binaries

                    if self.column_masking:
                        dropped_repr = self.unk_embeddings[i]

                        def dropout_p():
                            return np.random.randint(0, self.nin) / self.nin

                        # During training, non-dropped 1's are scaled by
                        # 1/(1-p), so we clamp back to 1.
                        batch_mask = torch.clamp(
                            torch.dropout(torch.ones(bs, 1, device=data.device),
                                          p=dropout_p(),
                                          train=self.training), 0, 1)
                        binaries = binaries.to(torch.float32,
                                               non_blocking=True,
                                               copy=False)
                        y_onehots[i] = batch_mask * binaries + (
                            1. - batch_mask) * dropped_repr

                else:
                    # Encode as plain one-hot.
                    y_onehot = torch.zeros(bs,
                                           coli_dom_size,
                                           device=data.device)
                    y_onehot.scatter_(1, data[:, i].view(-1, 1), 1)
                    y_onehots[i] = y_onehot

            res = torch.cat(y_onehots, 1)
            return res.to(torch.float32, non_blocking=True, copy=False)

        else:
            # Inference path.
            natural_idx = natural_col
            coli_dom_size = self.input_bins[natural_idx]
            if coli_dom_size > threshold:
                # Bit shift in PyTorch + GPU is 27% faster than np.
                data_np = data
                if out is None:
                    res = (data_np & self.bin_as_onehot_shifts[natural_idx]) > 0
                    return res.to(torch.float32, non_blocking=True, copy=False)
                else:
                    out.copy_(
                        (data_np & self.bin_as_onehot_shifts[natural_idx]) > 0)
                    return out
            else:
                assert False, 'inference'
                if out is None:
                    y_onehot = torch.zeros(bs,
                                           coli_dom_size,
                                           device=data.device)
                    y_onehot.scatter_(1, data, 1)
                    res = y_onehot
                    return res.to(torch.float32, non_blocking=True, copy=False)

                out.scatter_(1, data, 1)
                return out

    def EncodeInput(self, data, natural_col=None, out=None):
        """"Warning: this could take up a significant portion of a forward pass.

        Args:
          natural_col: if specified, 'data' has shape [N, 1] corresponding to
              col-'natural-col'.  Otherwise 'data' corresponds to all cols.
          out: if specified, assign results into this Tensor storage.
        """
        if self.input_encoding == 'binary':
            return self.ToBinaryAsOneHot(data, natural_col=natural_col, out=out)
        elif self.input_encoding == 'embed':
            return self.Embed(data, natural_col=natural_col, out=out)
        elif self.input_encoding is None:
            return data
        elif self.input_encoding == 'one_hot':
            return self.ToOneHot(data)
        else:
            assert False, self.input_encoding

    def forward(self, x):
        """Calculates unnormalized logits.

        If self.input_bins is not specified, the output units are ordered as:
            [x1, x2, ..., xn], ..., [x1, x2, ..., xn].
        So they can be reshaped as thus and passed to a cross entropy loss:
            out.view(-1, model.nout // model.nin, model.nin)

        Otherwise, they are ordered as:
            [x1, ..., x1], ..., [xn, ..., xn]
        And they can't be reshaped directly.

        Args:
          x: [bs, ncols].
        """
        x = self.EncodeInput(x)

        if self.direct_io_layer is not None:
            residual = self.direct_io_layer(x)
            return self.net(x) + residual

        return self.net(x)

    def forward_with_encoded_input(self, x):

        if self.direct_io_layer is not None:
            residual = self.direct_io_layer(x)
            return self.net(x) + residual

        return self.net(x)

    def logits_for_col(self, idx, logits):
        """Returns the logits (vector) corresponding to log p(x_i | x_(<i)).

        Args:
          idx: int, in natural (table) ordering.
          logits: [batch size, hidden] where hidden can either be sum(dom
            sizes), or emb_dims.

        Returns:
          logits_for_col: [batch size, domain size for column idx].
        """
        assert self.input_bins is not None

        if idx == 0:
            logits_for_var = logits[:, :self.logit_indices[0]]
        else:
            logits_for_var = logits[:, self.logit_indices[idx - 1]:self.
                                    logit_indices[idx]]
        if self.output_encoding != 'embed':
            return logits_for_var

        embed = self.embeddings[idx]

        if embed is None:
            # Can be None for small-domain columns.
            return logits_for_var

        # Otherwise, dot with embedding matrix to get the true logits.
        # [bs, emb] * [emb, dom size for idx]
        return torch.matmul(logits_for_var, embed.weight.t())

    def nll(self, logits, data):
        """Calculates -log p(data), given logits (the conditionals).

        Args:
          logits: [batch size, hidden] where hidden can either be sum(dom
            sizes), or emb_dims.
          data: [batch size, nin].

        Returns:
          nll: [batch size].
        """
        if data.dtype != torch.long:
            data = data.long()
        nll = torch.zeros(logits.size()[0], device=logits.device)
        for i in range(self.nin):
            logits_i = self.logits_for_col(i, logits)
            nll += F.cross_entropy(logits_i, data[:, i], reduction='none')

        return nll

    def sample(self, num=1, device=None):
        assert self.natural_ordering
        assert self.input_bins and self.nout > self.nin
        with torch.no_grad():
            sampled = torch.zeros((num, self.nin), device=device)
            indices = np.cumsum(self.input_bins)
            for i in range(self.nin):
                logits = self.forward(sampled)
                s = torch.multinomial(
                    torch.softmax(self.logits_for_i(i, logits), -1), 1)
                sampled[:, i] = s.view(-1,)
        return sampled


if __name__ == '__main__':
    # Checks for the autoregressive property.
    rng = np.random.RandomState(14)
    # (nin, hiddens, nout, input_bins, direct_io)
    configs_with_input_bins = [
        (2, [10], 2 + 5, [2, 5], False),
        (2, [10, 30], 2 + 5, [2, 5], False),
        (3, [6], 2 + 2 + 2, [2, 2, 2], False),
        (3, [4, 4], 2 + 1 + 2, [2, 1, 2], False),
        (4, [16, 8, 16], 2 + 3 + 1 + 2, [2, 3, 1, 2], False),
        (2, [10], 2 + 5, [2, 5], True),
        (2, [10, 30], 2 + 5, [2, 5], True),
        (3, [6], 2 + 2 + 2, [2, 2, 2], True),
        (3, [4, 4], 2 + 1 + 2, [2, 1, 2], True),
        (4, [16, 8, 16], 2 + 3 + 1 + 2, [2, 3, 1, 2], True),
    ]
    for nin, hiddens, nout, input_bins, direct_io in configs_with_input_bins:
        print(nin, hiddens, nout, input_bins, direct_io, '...', end='')
        model = MADE(nin,
                     hiddens,
                     nout,
                     input_bins=input_bins,
                     natural_ordering=True,
                     do_direct_io_connections=direct_io)
        model.eval()
        print(model)
        for k in range(nout):
            inp = torch.tensor(rng.rand(1, nin).astype(np.float32),
                               requires_grad=True)
            loss = model(inp)
            l = loss[0, k]
            l.backward()
            depends = (inp.grad[0].numpy() != 0).astype(np.uint8)

            depends_ix = np.where(depends)[0].astype(np.int32)
            var_idx = np.argmax(k < np.cumsum(input_bins))
            prev_idxs = np.arange(var_idx).astype(np.int32)

            # Asserts that k depends only on < var_idx.
            print('depends', depends_ix, 'prev_idxs', prev_idxs)
            assert len(torch.nonzero(inp.grad[0, var_idx:])) == 0
        print('ok')
    print('[MADE] Passes autoregressive-ness check!')
