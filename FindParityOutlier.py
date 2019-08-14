from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense, LSTM
import random
import math
import numpy as np
import sys

random.seed(87136784)

EVEN = 0
ODD = 1

def make_examples(num_examples, min_n, max_n, min_val, max_val):
    """
    Returns a num_examples x (max_n) matrix of random examples.
    Each row is an example of a possible list to feed into the network. Generating each example is
        done as follows:
        
        1. size := random size for this list in [min_n, max_n)
        2. parity := random choice of even or odd
        3. Fill a list with 'size' random numbers of parity 'parity' in range [min_val, max_val)
        4. number := a random number of parity opposite of 'parity' in range [min_val, max_val)
        5. Overwrite a random position in the list with 'number'
        6. Continue to append -1's to the end of the list until it is of size max_n
        7. Append an 'ODD' to the end of the list if 'parity' was even, 'EVEN' if odd (since
            we are looking for the loner parity)
        
    num_examples - the number of examples in the dataset
    min_n - the minimum possible number of numbers per example (inclusive)
    max_n - the maximum possible number of numbers per example (exclusive)
    min_val - the minimum possible number (inclusive)
    max_val - the maximum possible number (exclusive)
    """
    ret = []
    
    for i in range(num_examples):
        new_arr = []
        
        # The size of this example
        size = random.randrange(min_n, max_n)
        
        # Decide if we will be mainly even or odd
        parity = random.choice([EVEN, ODD])
        
        for j in range(max_n):
            
            # If we are > size, use -1
            if j > size:
                new_arr.append(-1)
                continue;
            
            # Otherwise make an even/odd number based on parity
            num = random.randrange(min_val, max_val)
            if (parity == EVEN and num % 2 != 0) or (parity == ODD and num % 2 == 0):
                num += 1
            new_arr.append(num)
        
        # Insert the one outlier parity into a random place
        num = random.randrange(min_val, max_val)
        if (parity == EVEN and num % 2 == 0) or (parity == ODD and num % 2 != 0):
            num += 1
        new_arr[random.randrange(0, size)] = num
        
        # Append the parity value
        new_arr.append(EVEN if parity == ODD else ODD)
        
        ret.append(new_arr)
    
    return np.array(ret)


def convert_examples(examples, num_bits, return_truth_values=True):
    """
    Takes the input array (a 2D array where each row is a list acting as a single datapoint
        for the RNN), and converts it into a 3D array of shape 
        (examples.shape[0], num_bits, examples.shape[1])
        
    Each value in each row is converted into a binary array (unless it is a -1, then it is converted
        into a row of -1's), and that is set as the values at the axis=1 dimension
    
    For example:
        The array: [[2, 4, 1, 10, 0],
                    [1, 6, 7, -1, 1],
                    [2, 5, 3, -1, 1],
                    [4, 4, 3,  4, 0]]
        with num_bits=4 and return_truth_values=True would return the arrays:
        
        X = 
        [
        
        [[0.0, 0.0, 1.0, 0.0],        rows, cols at z=0
         [0.0, 0.0, 0.0, 1.0],
         [0.0, 0.0, 1.0, 0.0],
         [0.0, 1.0, 0.0, 0.0]],
         
        [[0.0, 1.0, 0.0, 0.0],        rows, cols at z=1
         [0.0, 1.0, 1.0, 0.0],
         [0.0, 1.0, 0.0, 1.0],
         [0.0, 1.0, 0.0, 0.0]],
         
        [[0.0, 0.0, 0.0, 1.0],        rows, cols at z=2
         [0.0, 1.0, 1.0, 1.0],
         [0.0, 0.0, 1.0, 1.0],
         [0.0, 0.0, 1.0, 1.0]],
        
        [[ 1.0,  0.0,  1.0,  0.0],    rows, cols at z=3
         [-1.0, -1.0, -1.0, -1.0],
         [-1.0, -1.0, -1.0, -1.0],
         [ 0.0,  1.0,  0.0,  0.0]]
         
         ]
         
         and Y = 
         [0, 1, 1, 0]
    
    examples - the examples to convert (should be a 'num_examples' by 'max_n' array of ints)
    num_bits - the number of bits to use when converting
    return_truth_values - if True, then it is assumed that the correct answer for outlying
        parity (0 if even, 1 if odd) is the last column in 'examples'. We will then slice
        off that column before converting, and return it as the second object in a tuple
        
        If False, then the values are simply converted and only the X data is returned
    """
    if return_truth_values:
        Y = examples[:, -1]
        X = np.zeros([examples.shape[0], num_bits, examples.shape[1] - 1])
        for i in range(examples.shape[0]):
            for j in range(examples.shape[1] - 1):
                X[i, :, j] = int_to_bits(examples[i, j], num_bits)
        return X, Y
    else:
        X = np.zeros([examples.shape[0], num_bits, examples.shape[1]])
        for i in range(examples.shape[0]):
            for j in range(examples.shape[1]):
                X[i, :, j] = int_to_bits(examples[i, j], num_bits)
        return X

    
def int_to_bits(num, num_bits):
    """
    Converts a number to a 1D numpy array of bits of length num_bits. 
    If num >= 2^num_bits, then an array of all 1's is returned. 
    If num < 0, then an array of all -1's is returned
    
    num_bits - the number of bits to use
    num - the number to convert
    """
    
    # Check for negative number
    if num < 0:
        return np.full([num_bits, ], -1.0)
    
    # Check for num >= 2^num_bits (IE: not enough bits to store full number)
    if num >= 2**num_bits:
        return np.full([num_bits, ], 1)
    
    # Otherwise, do actual binary
    ret = [float(c) for c in "{0:b}".format(num)]
    while len(ret) < num_bits:
        ret = [0.0, ] + ret
        
    return np.array(ret)


def bits_in(num):
    """
    Returns the number of bits needed to fully store num
    """
    return math.floor(math.log(num, 2)) + 1


def predict_outlier_parity(l):
    """
    Returns 'EVEN' if the parity of the outlying value is even, 'ODD' if odd. Uses an advanced
        machine learning AI, implemented as a Recurrent Neural Network, to accurately predict
        whether the parity is even or odd.
    
    The network is a Recurrent Neural Network using an LSTM layer, ending with two FullyConnected
        layers, utilizing ReLU activation, dropout, a binary cross-entropy loss function, and the
        RMSProp optimizer.
    
    l - the list to find the outlier's parity in. Should have a length >= 3, and have exactly one
        value that is a different parity from the rest
    """
    num_examples = 10000
    min_n = 3
    max_n = math.ceil(1.1 * len(l)) + 4
    min_val = 0
    max_val = 2 * max(l)
    num_bits = bits_in(max_val)
    
    print("Creating example dataset...\n")

    a = make_examples(num_examples, min_n, max_n, min_val, max_val)
    X, Y = convert_examples(a, num_bits)
    
    print("Building Model...")
    
    model = Sequential()
    model.add(LSTM(units=max(256, num_bits), input_shape=(num_bits, max_n), activation="relu"))
    model.add(Dense(50, activation="relu"))
    model.add(Dropout(.5))
    model.add(Dense(1))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model.summary()
    
    print("\nTraining Model...\n")

    model.fit(X, Y, epochs=10, batch_size=32)
    
    print("\nDone training, converting input list and predicting...")
    
    # Convert the input into a list readable to the RNN
    testX = []
    for i in range(max_n):
        try:
            testX.append(l[i])
        except:
            testX.append(-1)
    testX = convert_examples(np.reshape(np.array(testX), [1, -1]), num_bits, return_truth_values=False)
    
    # Predict what the outlier's parity is
    p = model.predict(testX)[0]
    
    # If the value is closer to EVEN than it is ODD, return EVEN, otherwise return ODD
    return EVEN if abs(p - EVEN) <= abs(p - ODD) else ODD


def predict_outlier_parity_better(l, num_times=5):
    """
    Predicts the outlier's parity even MORE accurately using a group consensus model
    
    As an added efficiency bonus, we can stop after at least num_times / 2 models have outputted
      the same answer. This greatly increases performance since most of the time, the models
      agree.
      
    num_times - the number of times to run the RNN
    """
    num_even = 0
    num_odd = 0
    for i in range(num_times):
        print("\n\nRunning Model: %d / %d\n\n" % ((i + 1), num_times))
        if predict_outlier_parity(l) == EVEN:
            print("\n\nModel Prediction: EVEN")
            num_even += 1
        else:
            print("\n\nModel Prediction: ODD")
            num_odd += 1
        
        if num_even >= math.ceil(num_times / 2.0) or num_odd >= math.ceil(num_times / 2.0):
            print("\n\nOver 50% of the models have agreed, stopping early...\n\n")
            break
    
    return EVEN if num_even > num_odd else ODD


def find_outlier_value(l, num_times=None):
    """
    Finds the value in the given list that has a parity different than the rest. The list should
        be at least of length 3, and have only non-negative integers that are all of one parity
        (odd/even), with exacly 1 integer that is of the opposite parity (even/odd)
    
    l - the list of values to find the outlier in
    num_times - if num_times is > 1, then predict_outlier_parity_better() will be
        called with num_times=num_times, making the results more accurate
    """
    
    if num_times is not None and num_times > 1:
        parity = predict_outlier_parity_better(l, num_times=num_times)
    else:
        parity = predict_outlier_parity(l)
    print("EVEN!" if parity == EVEN else "ODD!")

    print("\n")

    # And now, search through the list finding the first even or odd number based on parity
    for v in l:
        if (parity == EVEN and v % 2 == 0) or (parity == ODD and v % 2 != 0):
            return v


# If this is called from command line, read in the args to make the list, then find the
#   outlier value
#
# USAGE:
# python FindOutlierValue.py comma_separated_list_of_values [num_times]
#
# comma_separated_list_of_values - a comma separated list of at least 3 non-negative integer values.
#   This should be a single string, and should have all values of a single parity (odd/even) with
#   exactly one value that is of the opposite parity (even/odd)
#
# num_times (optional) - the number of times to run the RNN, higher values take longer, but are more
#   likely to be accurate
if __name__ == "__main__":
    args = sys.argv
    
    # The 0 th index arg is the name of this program, so the 1 st index is comma_separated_list_of_values,
    #   split that list on commas, and convert to ints
    l = [int(s.strip()) for s in args[1].split(",")]
    
    # num_times is the 2 nd index arg
    if len(args) > 2:
        num_times = int(args[2])
    else:
        num_times = None
    
    
    # There's no need, in my opinion, to do any error checking on the command line inputs. Just
    #   don't enter bad inputs on the command line. It's not that hard!
    
    
    bad_val = find_outlier_value(l, num_times=num_times)

    print("The bad value is: %d" % bad_val)