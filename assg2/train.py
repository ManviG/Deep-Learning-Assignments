'''
Deep Learning Programming Assignment 1
--------------------------------------
Name: Manvi Garg
Roll No.: 13EX20015

======================================
Complete the functions in this file.
Note: Do not change the function signatures of the train
and test functions
'''
import numpy as np
import pickle as pkl
import time
import math

start_time = time.time()

dh = 200
m = 10
# buffer_w2 = np.zeros((batch_size, dh, d))
# buffer_w1 = np.zeros((batch_size, m, dh))


def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1.0-sigmoid(z))


class Perceptron(object):
    """docstring for Perceptron"""
    def __init__(self, sizes):
        self.sizes = sizes
        self.num_layers = len(sizes)
        biases = []
        weights = []
        # activations = []
        # activations_prime = []
        for i in range(self.num_layers - 1) :
            biases.append(np.zeros((sizes[i+1],1)))
            w = np.random.randn(sizes[i+1],sizes[i])
            weights.append(np.random.randn(sizes[i+1],sizes[i]) / math.sqrt(sizes[i]))
            # weights.append(np.random.ranf((sizes[i+1],sizes[i])))
            # print w.shape
            # activations.append(np.zeros((sizes[i+1],1)))
            # activations_prime.append(np.zeros((sizes[i+1],1)))

        self.biases = biases
        self.weights = weights
        # self.activations = activations
        # self.activations_prime = activations_prime
    

    def back_propagation(self,training_vector, target_vector, eta = 0.1):

        ## Forward Propagation
        outputs = []
        activations = []
        activations_prime = []
        prev_layer_vector = training_vector

        activations.append(training_vector)
        outputs.append(training_vector)
        for i in range(self.num_layers - 1):
            # print self.weights[i].shape, prev_layer_vector.shape, self.biases[i].shape
            # print
            output = (np.dot(self.weights[i], prev_layer_vector) + self.biases[i])
            outputs.append(output)
            activations.append(sigmoid(output))
            prev_layer_vector = sigmoid(output)

        delta = []
        # print len(activations), len(outputs)

        # for i in range(len(activations)):
        #     print activations[i].shape, outputs[i].shape
        # print

        # print len(prev_layer_vector), len(target_vector)
        assert len(prev_layer_vector) == len(target_vector)

        del_k = np.multiply(np.multiply(activations[-1], (1 - activations[-1])), (activations[-1] - target_vector))

        # del_k = np.zeros((prev_layer_vector.shape))
        # for k in range(len(prev_layer_vector)):
        #     del_k[k] = activations[-1][k]*(1 - activations[-1][k])*(activations[-1][k] - target_vector[k])
        delta.append(del_k)

        del_next_layer = del_k
        for i in reversed(range(self.num_layers-1)):
            if(i==0):
                break
            del_j = np.zeros((activations[i].shape))
            # print 'i = ' , i , activations[i].shape[0], activations[i+1].shape[0], del_j.shape, del_next_layer.shape

            for hidden_node in range(activations[i].shape[0]):
                # print hidden_node
                summ = 0
                for node_next_layer in range(del_next_layer.shape[0]):
                    # print node_next_layer, 
                    summ += del_next_layer[node_next_layer]*self.weights[i][node_next_layer][hidden_node]
                del_j[hidden_node] = summ*(activations[i][hidden_node])*(1 - activations[i][hidden_node])
                # print

            del_next_layer = del_j
            delta.append(del_j)

        # print len(delta)

        ## Update weights and biases
        del_b = []
        del_W = []
        for k in reversed(range(self.num_layers-1)):
            del_b.append(-1*(eta)*(delta[k]))
        self.biases = [x+y for x,y in zip(self.biases, del_b)]

        delta.reverse()
        l = 0
        for k in reversed(range(self.num_layers-1)):
            curr_weight_mat = self.weights[k]   ### iterate on weights in reverse order
            curr_weight_mat = curr_weight_mat - (np.multiply(delta[k], outputs[k].transpose()))*eta
            # print curr_weight_mat.shape, delta[l].shape, outputs[k].shape
            # for i in range(curr_weight_mat.shape[0]):
            #     for j in range(curr_weight_mat.shape[1]):
            #         curr_weight_mat[i][j] = curr_weight_mat[i][j] - (eta*delta[k][i]*outputs[k][j])
            self.weights[k] = curr_weight_mat
            # print self.weights[k].shape,
            l+=1

        # self.weights = [x + y]
        print
        print len(self.weights), len(self.biases)
        
        # print self.weights.shape, self.biases.shape


    def feed_forward(self,test_vector):
        # print test_vector.shape
        prev_layer_vector = test_vector

        for i in range(self.num_layers - 1):

            output = (np.dot(self.weights[i], prev_layer_vector) + self.biases[i])
            # outputs.append(output)
            # activations.append(sigmoid(output))
            prev_layer_vector = sigmoid(output)
            # print output, prev_layer_vector
            # for i in range(prev_layer_vector.shape[0]):
            #     for j in range(prev_layer_vector.shape[1]):
            #         a = np.exp(-output[i][j])
            #         print output[i][j], prev_layer_vector[i][j], a

        return prev_layer_vector


    def train(self, training_vectors, target_vectors):
        l = len(target_vectors)
        i = 0
        iters = 5
        for x in range(1,iters):
            for input_vec,output_vec in zip(training_vectors,target_vectors):
                self.back_propagation(input_vec,output_vec)
                i+=1
                print i,'done of',l
                print("--- %s seconds ---" % (time.time() - start_time))


MLP = Perceptron((784,100,10))


def normalize(training_vector):
    _min = np.min(training_vector)
    _max = np.max(training_vector)

    training_vector = (training_vector - _min)/(_max - _min)

    return training_vector

def train(trainX, trainY):
    '''
    Complete this function.
    '''

    training_vectors = []
    target_vectors = []
    # Change the 2D image into a one D vector
    for image in trainX:
        image.shape = (784,1) # 28*28 = 784
        training_vectors.append(image)
    training_vectors = np.array(training_vectors)

    # normalize(training_vectors)
    for i in range(len(training_vectors)):
        training_vectors[i] = normalize(training_vectors[i])

    for i in range(len(trainY)):
        target = np.zeros((10,1))
        target[trainY[i]][0] = 1
        target_vectors.append(target)
    target_vectors = np.array(target_vectors)

    print target_vectors.shape
    print training_vectors.shape

    print ('-----------------------Training ---------------\n\n')
    MLP.train(training_vectors,target_vectors)
    pkl.dump(MLP, open("model.p", "wb"))
    # pkl.dump(MLP.weights, open("weights/weights.p","wb"))
    

def test(testX):
    '''
    Complete this function.
    This function must read the weight files and    
    return the predicted labels.
    The returned object must be a 1-dimensional numpy array of
    length equal to the number of examples. The i-th element
    of the array should contain the label of the i-th test
    example.
    '''

    print ('----------------Testing----------------\n\n')
    MLP = pkl.load(open("model.p", "rb"))

    test_vectors = []
    for image in testX:
        image.shape = (784,1) # 28*28 = 784
        test_vectors.append(image)
    test_vectors = np.array(test_vectors)

    for i in range(len(test_vectors)):
        test_vectors[i] = normalize(test_vectors[i])


    labels = []
    i = 0
    for test_eg in test_vectors:
        # print 'ans = ', testY[i], 
        a = MLP.feed_forward(test_eg)
        print a.transpose()
        # print np.argmax(a),
        i+=1
        labels.append(np.argmax(a))

    labels = np.array(labels)

    return labels
