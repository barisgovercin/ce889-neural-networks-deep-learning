import weights as W     # contains trained weights
import scaler  as S     # contains x_min, x_max, y_min, y_max


class NeuralNetHolder:

    def __init__(self):
        # loading the sizes
        self.input_size = W.input_size
        self.hidden_size = W.hidden_size
        self.output_size = W.output_size

        # loading the weights and biases
        self.hidden_weights = W.hidden_weights
        self.hidden_biases = W.hidden_biases
        self.output_weights = W.output_weights
        self.output_biases = W.output_biases
        

    def sigmoidCalc(self, value):
        return 1.0 / (1.0 + (2.0 ** (-value)))
    
    def getTargetValues(self, input_row):   # time, pos_x, pos_y, vel_x, vel_y, x_target, y_target
        parts = input_row.split(",")
        clean = []

        for part in parts:
            stripped = part.strip()
            if stripped != "":
                clean.append(stripped)

        if len(clean) >= 6:  # if game loop sends full format 4 and 5 are target values
            x_target = float(clean[4])
            y_target = float(clean[5])
        else:   # if it is short format 0 and 1 are target values
            x_target = float(clean[0])
            y_target = float(clean[1])

        return x_target, y_target
    
    def forwardPass(self, inputs):  # take the inputs => compute tthe hidden layer => compute the output layer then return the final outputs
        # hidden layer
        hidden_outputs = []
        for h in range(self.hidden_size):
            total = 0.0
            for inp in range(self.input_size):
                weight = self.hidden_weights[h][inp]
                value = inputs[inp]
                total += weight * value

            total += self.hidden_biases[h]
            activation = self.sigmoidCalc(total)
            hidden_outputs.append(activation)   # h1_out, h2_out, h3_out

        # output layer (linear)
        final_outputs = []
        for o in range(self.output_size):
            total = 0.0
            for h in range(self.hidden_size):
                weight = self.output_weights[o][h]
                h_value = hidden_outputs[h]
                total += weight * h_value

            total += self.output_biases[o]
            final_outputs.append(total)     # new_vel_x, new_vel_y

        return final_outputs
    

    def predict(self, input_row):
        # parse raw x_target, y_target
        x_target, y_target = self.getTargetValues(input_row)

        # apply min-max normalisation with scaler.py
        if S.X_MAX != S.X_MIN:
            x_norm = (x_target - S.X_MIN) / (S.X_MAX - S.X_MIN)
        else:
            x_norm = 0.0

        if S.Y_MAX != S.Y_MIN:
            y_norm = (y_target - S.Y_MIN) / (S.Y_MAX - S.Y_MIN)
        else:
            y_norm = 0.0

        # limit values to [0,1]
        if x_norm < 0.0:
            x_norm = 0.0
        if x_norm > 1.0:
            x_norm = 1.0

        if y_norm < 0.0:
            y_norm = 0.0
        if y_norm > 1.0:
            y_norm = 1.0

        # feed forward
        outputs = self.forwardPass([x_norm, y_norm])

        # outputs = [new_vel_x, new_vel_y]
        return outputs
