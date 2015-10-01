__author__ = 'adminuser'

import numpy
import copy
from numpy import random as rand

class IOHMM:
    """
    Creates and maintains an HMM.

    N: The number of hidden states
    M: The number of observable symbols.
    T: The state transition matrix, an N*N matrix.
    E: The emission probability matrix E must be N*M.

                |a_11 a_12 ... a_1N|
                |a_21 a_22 ... a_2N|
           T =  | .    .        .  |
                | .         .   .  |
                |a_N1 a_N2 ... a_NN|

           a_ij = P(q_t = i| q_t-1=j )


                |b_11 b_12 ... b_1M|
                |b_21 b_22 ... b_2M|
           E =  | .    .        .  |
                | .         .   .  |
                |b_N1 b_N2 ... b_NM|

           b_ik = P(o_t = v_k | q_t = s_i)
           q_t is the state at time t,
           v_k is the k th symbol of ovservation,
           s_i is the i th symbol of state.

    """

    def __init__(self, n_states, input, output, **args):
        """
        Initialize an HMM.

        :param n_state: number of hidden states
        :param args:
               V - list of all observable symbols.
               Pi - Initial state probability matrix.
               A - transition matrix.
               B - emission probability matrix.

        :return:
        """

        # Number of hidden states
        self.N = n_states

        #Initialize observable INPUT symbol set parameters
        self.U = input
        self.K = len(self.U)
        self.input_map = dict(zip(self.U, range(len(self.U))))

        # Initialize observable symbols set parameters
        self.V = output
        self.M = len(self.V)
        self.output_map = dict( zip ( self.V, range( len( self.V ) )) )



        # Initialize transmission probability matrix
        if 'T' in args:
            self.T = args['T']

            if numpy.shape( self.T ) != (self.K, self.N, self.N):
                raise ValueError("The transmission probability matrix dimension mismatches the given states number.")

            for i in xrange(self.K):
                if not numpy.array_equal(self.T[i].sum(1), numpy.array([1.0,1.0])):
                    raise ValueError("The sum of each row in the transmission matrix should equal to 1")

        else:
            raw_T = rand.uniform(0,1, self.K * self.N * self.N).reshape(self.K, self.N, self.N)
            for i in xrange(self.K):
                raw_T[i] = (raw_T[i].T/raw_T[i].sum(1)).T
            self.T =  raw_T





        # Initialize emission probability matrix
        if 'E' in args:
            self.E = args['E']

            if numpy.shape(self.E) != (self.K, self.N, self.M):
                raise ValueError("The emission probaility matrix dimension mismatches the given states number and "
                                 "output symbols number")

            #if not numpy.array_equal(self.E.sum(1), numpy.array([1.0] * len(self.E.sum(1)))):
            #   raise ValueError("The sum of each row in the emission probability matrix should equal to 1")
        else:
            raw_E = rand.uniform(0,1,self.N * self.M).reshape(self.N, self.M)
            raw_E_sum = raw_E.sum(axis = 1, keepdims = True)
            self.E = raw_E.astype(float)/raw_E_sum

        if 'F' in args:
            self.F = args['F']
            for i in self.F.keys():
                self.T[i,:] = self.F[i]
            if sum(self.F[i])!= 1:
               raise Exception("The probability transferring from this state should sum up to 1.")
        else:
            self.F = {}




        # Initialize th
        if 'Pi' in args:
            self.Pi = args['Pi']

            if len(self.Pi) != self.N:
                raise ValueError("The initial state probability dimension mismatches the given states number.")

            if self.Pi.sum() != 1:
                raise ValueError("The initial state probability does not add up to 1.")

        else:
            raw_Pi = numpy.array([1] * self.N)
            self.Pi = raw_Pi.astype(float)/raw_Pi.sum()

    def print_iohmm(self, label):
        """
        Print out the HMM elements
        """
        print "\n"*2+ "*"*24 + "\n" + label  + "\n" + "*"*24 + "\n"
        print "\n1) Numerber of hidden states:" + str(self.N)
        print "\n3) The input mapping in IOHMM:" + str(self.input_map)
        print "\n5) The symbol mapping in IOHMM:" + str(self.symbol_map)
        print "\n6) The transmission proability matrix T:\n" + str(self.T)
        print "\n7) The emission probability matrix E:\n" + str(self.E)
        print "\n8) The initial state probability Pi: \n" + str(self.Pi)




if __name__ == '__main__':
    pass