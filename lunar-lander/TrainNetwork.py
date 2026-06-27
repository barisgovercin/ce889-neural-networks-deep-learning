_random_seed = 1234567

def randNumGenerator():     # used this instead of the random library [0-1)
    global _random_seed
    _random_seed = (1103515245 * _random_seed + 12345) % 2147483648
    return _random_seed / 2147483648.0


def randomBounder(lower_bound, upper_bound):        # returns a random float in [lower_bound, upper_bound).
    r = randNumGenerator()
    return lower_bound + (upper_bound - lower_bound) * r


def sigmoidCalc(number):        # 1/(1 + 2^(-x))
    denom = 1.0 + (2.0 ** (-number))
    return 1.0 / denom


def mseCalc(pairs):     # MSE = average( (target - output)^2 )
    if len(pairs) == 0:
        return 0.0

    total_squared = 0.0
    count = 0

    for pair in pairs:
        targets = pair[0]   # velx, vely real
        outputs = pair[1]   # velx, vely predicted

        for index in range(len(targets)):
            diff = targets[index] - outputs[index]
            total_squared += diff * diff
            count += 1

    return total_squared / float(count)


class MLP:

    def __init__(self, input_size, hidden_size, output_size, learning_rate, momentum):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.momentum = momentum

        # weights and biases, then the weight vectors of its weight will be stored in these lists
        self.hidden_weights = []
        self.hidden_biases = []
        self.output_weights = []
        self.output_biases = []

        # velocities for momentum, then the momentum values ​​​​of the previous steps will be stored in these lists
        self.hidden_weight_velocities = []
        self.hidden_bias_velocities = []
        self.output_weight_velocities = []
        self.output_bias_velocities = []

        # hidden layer initialisation
        for hidden_neuron_index in range(hidden_size):
            weight_row = []
            velocity_row = []

            for input_index in range(input_size):
                weight_row.append(randomBounder(-0.5, 0.5))
                velocity_row.append(0.0)

            self.hidden_weights.append(weight_row)
            self.hidden_weight_velocities.append(velocity_row)

            self.hidden_biases.append(randomBounder(-0.5, 0.5))
            self.hidden_bias_velocities.append(0.0)

        # output layer initialisation
        for output_neuron_index in range(output_size):
            weight_row = []
            velocity_row = []

            for hidden_index in range(hidden_size):
                weight_row.append(randomBounder(-0.5, 0.5))
                velocity_row.append(0.0)

            self.output_weights.append(weight_row)
            self.output_weight_velocities.append(velocity_row)

            self.output_biases.append(randomBounder(-0.5, 0.5))
            self.output_bias_velocities.append(0.0)


    def forwardPass(self, inputs):      # input xnorm ynorm => weighted sum => sigmoid => returns (hidden_outputs, final_outputs),
        # hidden layer (sigmoid)
        hidden_outputs = []
        for hidden_neuron_index in range(self.hidden_size):
            net = 0.0
            for input_index in range(self.input_size):
                weight = self.hidden_weights[hidden_neuron_index][input_index]
                input_value = inputs[input_index]   # xnorm ynorm 
                net += weight * input_value

            net += self.hidden_biases[hidden_neuron_index]
            activation = sigmoidCalc(net)
            hidden_outputs.append(activation)   # h1, h2, h3

        # output layer (linear)     inputs hidden_outputs => weighted sum => returns final outputs
        final_outputs = []
        for output_neuron_index in range(self.output_size):
            net = 0.0
            for hidden_index in range(self.hidden_size):
                weight = self.output_weights[output_neuron_index][hidden_index]
                hidden_value = hidden_outputs[hidden_index]     # h1, h2, h3
                net += weight * hidden_value

            net += self.output_biases[output_neuron_index]
            final_outputs.append(net)   # vx_pred, vy_pred

        return hidden_outputs, final_outputs


    def backpropagation(self, inputs, targets):     # forward pass => compute deltas => update weights/biases with momentum
        hidden_outputs, final_outputs = self.forwardPass(inputs)

        # output deltas, (linear) gets the true vel's and predicted vel's returns output_deltas to use weight updates
        output_deltas = []
        for output_neuron_index in range(self.output_size):
            error = targets[output_neuron_index] - final_outputs[output_neuron_index]   # error = target - prediction
            output_deltas.append(error)     # vx_delta, vy_delta   

        # hidden deltas, gets the output_deltas, weights and hidden_outputs an returns 3 hidden deltas to use weight updates
        hidden_deltas = []
        for hidden_neuron_index in range(self.hidden_size):
            error_sum = 0.0
            for output_neuron_index in range(self.output_size):
                w = self.output_weights[output_neuron_index][hidden_neuron_index]
                error_sum += output_deltas[output_neuron_index] * w     # sum of (output_delta * corresponding weight)

            h_out = hidden_outputs[hidden_neuron_index]
            delta = error_sum * h_out * (1.0 - h_out)
            hidden_deltas.append(delta)   # h1_delta, h2_delta, h3_delta

        # updating output weights and biases
        for output_neuron_index in range(self.output_size):

            # weights
            for hidden_index in range(self.hidden_size):
                grad = output_deltas[output_neuron_index] * hidden_outputs[hidden_index]    # gradient = output_delta * hidden_output

                old_vel = self.output_weight_velocities[output_neuron_index][hidden_index]
                new_vel = self.momentum * old_vel + self.learning_rate * grad   # new_vel = momentum * old_vel + learning_rate * gradient

                self.output_weight_velocities[output_neuron_index][hidden_index] = new_vel
                self.output_weights[output_neuron_index][hidden_index] += new_vel   # weight update

            # bias
            bias_grad = output_deltas[output_neuron_index]
            old_bias_vel = self.output_bias_velocities[output_neuron_index]
            new_bias_vel = self.momentum * old_bias_vel + self.learning_rate * bias_grad    # new_vel for bias = momentum * old_vel + learning_rate * bias_gradient

            self.output_bias_velocities[output_neuron_index] = new_bias_vel
            self.output_biases[output_neuron_index] += new_bias_vel     # bias update

        # updating hidden weights and biases
        for hidden_neuron_index in range(self.hidden_size):

            # weights
            for input_index in range(self.input_size):
                grad = hidden_deltas[hidden_neuron_index] * inputs[input_index]   # gradient = hidden_delta * input_value

                old_vel = self.hidden_weight_velocities[hidden_neuron_index][input_index]
                new_vel = self.momentum * old_vel + self.learning_rate * grad   # new_vel = momentum * old_vel + learning_rate * gradient

                self.hidden_weight_velocities[hidden_neuron_index][input_index] = new_vel
                self.hidden_weights[hidden_neuron_index][input_index] += new_vel    # weight update

            # bias
            bias_grad = hidden_deltas[hidden_neuron_index]
            old_bias_vel = self.hidden_bias_velocities[hidden_neuron_index]
            new_bias_vel = self.momentum * old_bias_vel + self.learning_rate * bias_grad    # new_vel for bias = momentum * old_vel + learning_rate * bias_gradient

            self.hidden_bias_velocities[hidden_neuron_index] = new_bias_vel
            self.hidden_biases[hidden_neuron_index] += new_bias_vel    # bias update

        return final_outputs    # velx vely predictions used to control lander
    

    def train(self, train_set, val_set, epoch_count):       # training the network for number of epochs
        for epoch_index in range(epoch_count):
            # shuffle train set for each epoch
            length = len(train_set)
            for i in range(length):
                j = int(randNumGenerator() * length)
                tmp = train_set[i]
                train_set[i] = train_set[j]
                train_set[j] = tmp

            # training pass
            train_pairs = []
            for item in train_set:
                inputs = item[0]    # xnorm, ynorm
                targets = item[1]   # velx, vely
                outputs = self.backpropagation(inputs, targets) # forward runs, error calc, weight and bias updates, returns velx_pred, vely_pred
                train_pairs.append([targets, outputs])  # real vel's and predicted vel's

            train_mse = mseCalc(train_pairs)

            # validation pass
            val_pairs = []
            for item in val_set:
                inputs = item[0]    # xnorm, ynorm
                targets = item[1]   # velx, vely
                _, outputs = self.forwardPass(inputs)   # no learning in val so no hidden outputs thats why _ placeholder
                val_pairs.append([targets, outputs])

            val_mse = mseCalc(val_pairs)

            print("Epoch", epoch_index + 1, "/", epoch_count, "- Train MSE:", train_mse, "- Val MSE:", val_mse)


def readNormData(path):     # read the normalised CSV file.
    rows = []

    try:
        file_handle = open(path, "r")
    except:
        print("File cannot opened:", path)
        return rows

    # skip header
    file_handle.readline()

    for line in file_handle:
        line = line.strip()
        if line != "":
            parts = line.split(",")
            if len(parts) >= 4:
                x_norm = float(parts[0])
                y_norm = float(parts[1])
                vel_y = float(parts[2])
                vel_x = float(parts[3])

                inputs = [x_norm, y_norm]
                targets = [vel_x, vel_y]
                rows.append([inputs, targets])

    file_handle.close()
    return rows


def saveWeights(net, filename):
    try:
        file_handle = open(filename, "w")
    except:
        print("Weights cannot be written:", filename)
        return
    
    file_handle.write("input_size = " + str(net.input_size) + "\n")
    file_handle.write("hidden_size = " + str(net.hidden_size) + "\n")
    file_handle.write("output_size = " + str(net.output_size) + "\n\n")

    file_handle.write("hidden_weights = " + str(net.hidden_weights) + "\n\n")
    file_handle.write("hidden_biases = " + str(net.hidden_biases) + "\n\n")
    file_handle.write("output_weights = " + str(net.output_weights) + "\n\n")
    file_handle.write("output_biases = " + str(net.output_biases) + "\n")

    file_handle.close()
    print("Weights saved to:", filename)


def main():
    input_size = 2
    hidden_size = 3            # number of neurons in hidden layer in MATLAB
    output_size = 2

    learning_rate = 0.01       # net.trainParam.lr in MATLAB
    momentum = 0.6             # net.trainParam.mc in MATLAB
    epochs = 111               # net.trainParam.epochs in MATLAB

    train_data_path = "ce889_data_train_norm.csv"
    val_data_path = "ce889_data_val_norm.csv"

    print("Loading training data from:", train_data_path)
    train_data = readNormData(train_data_path)

    print("Loading validation data from:", val_data_path)
    val_data = readNormData(val_data_path)

    if len(train_data) == 0 or len(val_data) == 0:
        print("No data, cannot train.")
        return

    print("Total train samples:", len(train_data))
    print("Total val samples:", len(val_data))

    print("\nHyperparameters:")
    print(" Hidden size:", hidden_size)
    print(" Learning rate:", learning_rate)
    print(" Momentum:", momentum)
    print(" Epochs:", epochs)

    # create and train the network
    network = MLP(input_size, hidden_size, output_size, learning_rate, momentum)

    network.train(train_data, val_data, epochs)

    print("\nTraining finished.")
    saveWeights(network, "weights.py")


if __name__ == "__main__":
    main()
