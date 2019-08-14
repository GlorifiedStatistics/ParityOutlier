# Parity Outlier

This project is a solution to the "Bad Coding Challenge #17" posted on [r/badcode](https://www.reddit.com/r/badcode/). The problem is as follows:

> Write a method that accepts an array of integers where every item is either odd or even except for a single outlier and return the outlier - e.g. [1, 3, 3, 7, 9, 13, 33, 21, 6] are all odd except for 6. How to handle invalid input is left undefined.
>
> outlier([1, 3, 3, 7, 9, 13, 33, 21, 6]) //6
>
> outlier([2, 2, 4, 2, 11, 10, 6]) //11

Once you know what the parity is that you are looking for, it is rather simple to just search through the list, and return the first value with that parity. The problem is finding the parity of the outlier. I thought about this problem for hours and couldn't find a solution. So instead, I decided to have the computer figure it out for me.

## My Solution

Use a Recurrent Neural Network (RNN) to iterate through the list, and return to me the parity of the outlier I am looking for. Then, I can just search the list for the first value of that parity! (Since there should be only one)

I train the RNN with thousands of examples of parity outliers. Because I am generating the examples myself, I already know what the outlying parity is. The network learns how to find that parity, and tells me. I then do a linear search through the array until I find the first value with that parity. Accuracy on a vaidation set reaches around 90%!

### Installation Requirements

I used python 3 with packages tensorflow (with Keras), numpy, and jupyterlab

### Files

There are three files in this repository ( 4 if you count the README ;) ):

#### FindParityOutlier.ipynb

This is a jupyter notebook with all of my code in it. The last cell is where you can enter in a list, and then call find_outlier_value() to find the outlier value.

```python
def find_outlier_value(l, num_times=None):
    """
    Finds the value in the given list that has a parity different than the rest. The list should
        be at least of length 3, and have only non-negative integers that are all of one parity
        (odd/even), with exacly 1 integer that is of the opposite parity (even/odd)
    
    l - the list of values to find the outlier in
    num_times - if num_times is > 1, then the network is created multiple times with different
        training sets, and the most common answer is chosen in order to increase accuracy
    """
```

#### FindParityOutlier.py

A python script version of the jupyter notebook. This can be imported in order to get access to all of the functions inside, or it can be called from command line like so:

> python FindOutlierValue.py comma_separated_list_of_values [num_times]

comma_separated_list_of_values - a comma separated list of at least 3 non-negative integer values. This should be a single string, and should have all values of a single parity (odd/even) with exactly one value that is of the opposite parity (even/odd)

num_times (optional) - the number of times to run the RNN, higher values take longer, but are more likely to be accurate. For the sake of efficiency, the code stops once num_times/2 networks agree on the right answer, since there is no need to test the rest of them at that point.

#### ValidateFindParityOutlier.ipynb

Another jupyter notebook to asses the accuracy of the RNN on a validation set. The categorical accuracy on a validation set is around 90%.

### Data Conversion

The data first has to be converted into a format the RNN can use to learn. Each list of values is expanded (filled with -1's) until it is the same size as the number of steps in the RNN. The values are then converted
into a (number_of_lists x size) matrix. Each value is then converted into a binary number (the maximum number of bits necessary to store all numbers in binary is computed first, and that many bits are used for the conversion). Any -1's are simply converted into all -1's. This then becomes a row, and the matrix is extended to be a tensor of shape (number_of_lists, number_of_bits, size). For example:

Assume we had the lists:
    [2, 4, 1, 10]
    [1, 6, 7]
    [2, 5, 3]
    [4, 4, 3, 4]

The smaller lists would have -1's appended until they are the right size:
    [2, 4, 1, 10]
    [1, 6, 7, -1]
    [2, 5, 3, -1]
    [4, 4, 3,  4]

The largest number in these arrays is '10' which needs 4 bits to store, so converting all numbers into 4-bit representations (or -1's) would look like:

``` python
X = 
    [
    
        [[0.0, 0.0, 1.0, 0.0],        # rows, cols at z=0
         [0.0, 0.0, 0.0, 1.0],
         [0.0, 0.0, 1.0, 0.0],
         [0.0, 1.0, 0.0, 0.0]],
         
        [[0.0, 1.0, 0.0, 0.0],        # rows, cols at z=1
         [0.0, 1.0, 1.0, 0.0],
         [0.0, 1.0, 0.0, 1.0],
         [0.0, 1.0, 0.0, 0.0]],
         
        [[0.0, 0.0, 0.0, 1.0],        # rows, cols at z=2
         [0.0, 1.0, 1.0, 1.0],
         [0.0, 0.0, 1.0, 1.0],
         [0.0, 0.0, 1.0, 1.0]],
         
        [[ 1.0,  0.0,  1.0,  0.0],    # rows, cols at z=3
         [-1.0, -1.0, -1.0, -1.0],
         [-1.0, -1.0, -1.0, -1.0],
         [ 0.0,  1.0,  0.0,  0.0]]
         
    ]
```

This data is then fed into the RNN to train on

### Neural Network Architecture

The RNN uses an LSTM layer with at least 256 neurons (more if we need more than 256 bits to store the largest integer), which feeds into a Dense layer with 50 neurons, and finally feeds into another Dense layer with a single neuron. That neuron will take the value '0' if the parity of the outlier is even, or '1' if odd. The network also uses Dropout on the middle Dense layer to help reduce overfitting. The loss function is a binary cross-entropy loss.

In order to make sure we are being completely accurate, a new RNN is created and trained on a new random set of data for each list we want to check.