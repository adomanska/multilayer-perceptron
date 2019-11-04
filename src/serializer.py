import numpy as np

class Serializer:
    def serialize(self, nn, filename):
        best_acc_index = np.argmax(nn.accuracies)
        print(best_acc_index)
        weights = [layer.weights_history[-1] for layer in nn.layers]
        biases = [layer.biases_history[-1] for layer in nn.layers]
        np.savez(filename, *weights, *biases)

    def deserialize(self, filename):
        npzfile = np.load(filename, allow_pickle=True)
        weights_files = npzfile.files[0 : int(len(npzfile.files) / 2)]
        biases_files = npzfile.files[int(len(npzfile.files) / 2) ::]
        print(npzfile.files)
        weights = [npzfile[w_file] for w_file in weights_files]
        biases = [npzfile[b_file] for b_file in biases_files]
        return weights, biases
