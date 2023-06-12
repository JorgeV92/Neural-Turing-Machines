import torch
import torch.nn as nn

class Controller(nn.Module):
    def __init__(self, controller_type, input_size, output_size, hidden_size):
        super(Controller, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        if controller_type == 'LSTM':
            self.controller = LSTMController(input_size, hidden_size)
        elif controller_type == 'FFN':
            self.controller = FFNController(input_size, hidden_size)
        elif controller_type == 'GRU':
            self.controller = GRUController(input_size, hidden_size)
        else:
            raise ValueError(f"Invalid controller type: {controller_type}")

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, prev_state):
        out, state = self.controller(x, prev_state)
        out = self.fc(out)
        return out, state

    def init_state(self, batch_size):
        return self.controller.init_state(batch_size)


class LSTMController(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMController, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1)
        self.hidden_size = hidden_size

    def forward(self, x, prev_state):
        out, state = self.lstm(x, prev_state)
        return out, state

    def init_state(self, batch_size):
        return (torch.zeros(1, batch_size, self.hidden_size),
                torch.zeros(1, batch_size, self.hidden_size))


class FFNController(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(FFNController, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return x
    
    
class GRUController(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRUController, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=1)
        self.hidden_size = hidden_size

    def forward(self, x, prev_state):
        out, state = self.gru(x, prev_state)
        return out, state

    def init_state(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)
