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
                                 "input/output symbols number")

            for i in xrange(self.K):
                if not numpy.array_equal(self.E[i].sum(1), numpy.array([1.0,1.0])):
                    raise ValueError("The sum of each row in the emission matrix should equal to 1")
        else:
            raw_E = rand.uniform(0,1,self.K*self.N * self.M).reshape(self.K, self.N, self.M)
            for i in xrange(self.K):
                raw_E[i] = (raw_E[i].T/raw_E[i].sum(1)).T
            self.E = raw_E



        if 'F' in args:
            self.F = args['F']
            for key in self.F.keys():
                if sum(self.F[key])!= 1:
                   raise ValueError("The probability transferring from this state should sum up to 1.")
                self.T[key[0],key[1],:] = self.F[key]
        else:
            self.F = {}

        # Initialize Initial State Distribution
        if 'Pi' in args:
            self.Pi = args['Pi']

            if len(self.Pi) != self.N:
                raise ValueError("The initial state probability dimension mismatches the given states number.")

            if numpy.sum(self.Pi)!= 1:
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
        print "\n5) The output mapping in IOHMM:" + str(self.output_map)
        print "\n6) The transmission proability matrix T:\n" + str(self.T)
        print "\n7) The emission probability matrix E:\n" + str(self.E)
        print "\n8) The initial state probability Pi: \n" + str(self.Pi)

    def toIndex(self, seq, map):
        index_seq = []
        for o in seq:
            if o not in map:
                raise ValueError("The symbol "+o+" is not defined.")
            index_seq.append(map[o])
        return index_seq

    def forward(self, Input, Output, scaling = False, debug = False):

        if debug:
            print "\n"*2+ "*"*23 + "\n" +"*"*2+" FORWARD ALGORITHM "+"*"*2 + "\n" + "*"*23 + "\n"

        input_seq = self.toIndex(Input, self.input_map)
        output_seq = self.toIndex(Output, self.output_map)
        T = len(input_seq)
        # create scaling vector
        if scaling:
            c = numpy.zeros([T],float)



        # Instantiation
        Alpha = numpy.zeros([self.N, T],float)
        Alpha[:,0] = self.Pi * self.E[input_seq[0],:,output_seq[0]]
        if debug:
            print "t=0"
            print Alpha[:,0]



        #Induction
        for t in xrange(1, T):
            Alpha[:,t] = numpy.dot(Alpha[:,t-1],self.T[input_seq[t]])
            if debug:
                print "t=" + str(t)
                print Alpha[:,t]

        if scaling:
            c = 1.0/ Alpha.sum(0)
            Alpha = Alpha * c
            log_prob = -numpy.log(c[T-1])

            if debug:
                print "\nAlpha:"
                print Alpha
                print "\nc:"
                print c
                print "\nP(Obs|iohmm)=" + str(log_prob)
            return (log_prob, Alpha, c)

        else:
            prob = numpy.log(numpy.sum(Alpha[:, T-1]))
            if debug:
                print "\nAlpha:"
                print Alpha
                print "\nP(Obs|iohmm)=" + str(prob)
            return (prob, Alpha)

    def backward(self, Input, Output, debug = False):
        if debug:
            print "\n"*2+ "*"*24 + "\n" +"*"*2+" BACKWARD ALGORITHM "+"*"*2 + "\n" + "*"*24 + "\n"

        input_seq = self.toIndex(Input, self.input_map)
        output_seq = self.toIndex(Output, self.output_map)
        T = len(input_seq)

        # Initialization
        Beta = numpy.zeros([self.N, T], float)
        Beta[:, T-1] = 1.0

        if debug:
            print "t=" + str(T-1)
            print Beta[:,T-1]

        # Induction
        for t in reversed(xrange(T-1)):
            print "\n"
            print self.T[input_seq[t]]
            print self.E[input_seq[t+1],:,output_seq[t+1]]
            print Beta[:,t+1]
            #Beta[:,t] = (self.T[input_seq[t]] * self.E[input_seq[t+1],:,output_seq[t+1]] * Beta[:,t+1]).sum(0)
            Beta[:,t] = numpy.dot(self.T[input_seq[t]] , self.E[input_seq[t+1],:,output_seq[t+1]] * Beta[:,t+1])
            if debug:
                print "t=" + str(t)
                print Beta[:,t]

        if debug:
            print "\nBeta:"
            print Beta

        return Beta






if __name__ == '__main__':
    input = ["elicit", "tell"]
    output = ["wrong", "correct", "told"]
    input_seq = ["elicit", "tell", "elicit", "elicit", "tell"]
    output_seq = ["wrong", "told","correct","wrong","told"]
    T = numpy.array([0.2,0.8,0,1,0.1,0.9,0,1]).reshape(2,2,2)
    E = numpy.array([0.5,0.5,0,0.1,0.9,0,0,0,1,0,0,1]).reshape(2,2,3)
    Pi = numpy.array([0.5,0.5])
    iohmm = IOHMM(2, input=input, output=output, T=T, E=E, Pi=Pi)
    iohmm.print_iohmm("ORIGINAL IOHMM ELEMENTS")
    iohmm.backward(input_seq, output_seq, debug= True)