import torch.jit as jit
import torch
import torch.nn as nn
from torch.nn import Parameter
from typing import List, Tuple
from torch import Tensor
from collections import namedtuple

# copied from
# https://github.com/pytorch/pytorch/blob/master/benchmarks/fastrnns/custom_lstms.py
# with CoupledLSTMCell implemented

LSTMState = namedtuple('LSTMState', ['hx', 'cx'])


def reverse(lst: List[Tensor]) -> List[Tensor]:
    return lst[::-1]


class CoupledLSTMCell(nn.Module): #jit.ScriptModule):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.randn(3 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.randn(3 * hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.randn(3 * hidden_size))
        self.bias_hh = Parameter(torch.randn(3 * hidden_size))

    #@jit.script_method
    def forward(self, input: Tensor, state: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        hx, cx = state
        gates = (torch.mm(input, self.weight_ih.t()) + self.bias_ih +
                 torch.mm(hx, self.weight_hh.t()) + self.bias_hh)
        ingate, cellgate, outgate = gates.chunk(3, 1)
        #print("ingate", ingate.shape)

        ingate = torch.sigmoid(ingate)
        forgetgate = 1.-ingate #torch.sigmoid(forgetgate)  #1. - ingate
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, (hy, cy)


class LSTMLayer(nn.Module): #jit.ScriptModule):
    def __init__(self, cell, *cell_args):
        super().__init__()
        self.cell = cell(*cell_args)

    #@jit.script_method
    def forward(self, input: Tensor, state: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        inputs = input.unbind(0)
        outputs = [] #torch.jit.annotate(List[Tensor], [])
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state)
            outputs += [out]
        return torch.stack(outputs), state


class ReverseLSTMLayer(nn.Module): #jit.ScriptModule):
    def __init__(self, cell, *cell_args):
        super().__init__()
        self.cell = cell(*cell_args)

    #@jit.script_method
    def forward(self, input: Tensor, state: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        inputs = reverse(input.unbind(0))
        outputs = [] #jit.annotate(List[Tensor], [])
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state)
            outputs += [out]
        return torch.stack(reverse(outputs)), state


class BidirLSTMLayer(nn.Module): #jit.ScriptModule):
    #__constants__ = ['directions']

    def __init__(self, cell, *cell_args):
        super().__init__()
        self.directions = nn.ModuleList([
            LSTMLayer(cell, *cell_args),
            ReverseLSTMLayer(cell, *cell_args),
        ])

    #@jit.script_method
    def forward(self, input: Tensor, states: List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]:
        outputs = [] #jit.annotate(List[Tensor], [])
        output_states = [] #jit.annotate(List[Tuple[Tensor, Tensor]], [])
        # XXX: enumerate https://github.com/pytorch/pytorch/issues/14471
        i = 0
        for direction in self.directions:
            state = states[i]
            out, out_state = direction(input, state)
            outputs += [out]
            output_states += [out_state]
            i += 1
        return torch.cat(outputs, -1), output_states


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        #print(input_size, hidden_size)
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = BidirLSTMLayer(CoupledLSTMCell, self.input_size, self.hidden_size)

    def forward(self, X):
        batch_size = X.shape[0]
        #print("LSTM input shape", X.shape)
        #inp = torch.randn(seq_len, batch_size, input_size)
        states = [
            LSTMState(
                torch.zeros(batch_size, self.hidden_size, dtype=X.dtype),
                torch.zeros(batch_size, self.hidden_size, dtype=X.dtype)
            ),
            LSTMState(
                torch.zeros(batch_size, self.hidden_size, dtype=X.dtype),
                torch.zeros(batch_size, self.hidden_size, dtype=X.dtype)
            )
        ]
        #print(X.permute(1,0,2).shape)
        out, out_state = self.rnn(X.permute(1,0,2), states)
        #print(out.shape, [(len(x), x[0].shape, x[1].shape) for hy, cy in out_state])
        (hf, cf), (hb, cb) = out_state
        hn, cn = torch.cat([hf, hb], dim=1)[None, :, :], torch.cat([cf, cb], dim=1)[None, :, :]
        #print(out.shape, hn.shape, cn.shape)
        return out, (hn, cn)