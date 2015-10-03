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
            #Beta[:,t] = numpy.dot(self.T[input_seq[t]] , self.E[input_seq[t+1],:,output_seq[t+1]] * Beta[:,t+1])
            Beta[:,t] = numpy.dot(self.T[input_seq[t]] * self.E[input_seq[t+1],:,output_seq[t+1]]  , Beta[:,t+1])
            if debug:
                print "t=" + str(t)
                print Beta[:,t]

        if debug:
            print "\nBeta:"
            print Beta

        return Beta

    def bawn_welch(self, Input, Output, **args):

        print "\n"*2+ "*"*24 + "\n" +"*"*1+" Bawn Welch ALGORITHM "+"*"*1 + "\n" + "*"*24 + "\n"

        epochs = args['epochs'] if 'epochs' in args else 20
        updatePi = args['updatePi'] if 'updatePi' in args else True
        updateT = args['updateT'] if 'updateT' in args else True
        updateE = args['updateE'] if 'updateE' in args else True
        debug = args['debug'] if 'debug' in args else False
        epsilon = args['epsilon'] if 'epsilon' in args else 0.001
        Val_seq = args['val'] if 'val' in args else None

        LLS = []



        for epoch in xrange(epochs):

            print "-"*10 + "\n"+ "Epoch "+str(epoch)+"\n"+"-"*10

            exp_si_t0 = numpy.zeros([self.N],float)
            exp_num_from_Si = numpy.zeros([self.K , self.N],float)
            exp_num_in_Si = numpy.zeros([self.K , self.N],float)
            exp_num_Si_Sj = numpy.zeros([self.K ,self.N ,self.N],float)
            exp_num_in_Si_Vk = numpy.zeros([self.K, self.N, self.M], float)
            LogLikelihood = 0

            for i in xrange(len(Input)):
                if debug:
                    print "\nThe input sequence is: "+ str(Input[i])
                    print "\nThe output sequence is: "+str(Output[i])

                log_prob, Alpha = self.forward(Input[i], Output[i], scaling= False, debug= False)

                Beta = self.backward(Input[1], Output[1], debug = False)
                LogLikelihood += log_prob

                if debug:
                    print "\nAlpha:"
                    print Alpha
                    print "\nBeta:"
                    print Beta

                T = len(Input[i])
                input = self.toIndex(Input[i], self.input_map)
                output = self.toIndex(Output[i], self.output_map)

                raw_Gamma = Alpha * Beta
                Gamma = raw_Gamma/raw_Gamma.sum(0)
                if debug:
                    print "\nRaw Gamma:"
                    print raw_Gamma
                    print "\nGamma:"
                    print Gamma


                exp_si_t0 += Gamma[:, 0]
                #exp_si_t0 += Gamma[0,:]
                for each in self.input_map:
                    which = numpy.array([self.input_map[each]==x for x in input])
                    exp_num_in_Si[self.input_map[each],] += Gamma[:,which].sum(1)
                    exp_num_from_Si[self.input_map[each],] += Gamma[:,which[:T-1]].sum(1)

                # The probability in state Si having Observation Oj
                for each_input in self.input_map:
                    for each_output in self.output_map:
                        which_input = numpy.array([self.input_map[each_input]==x for x in input])
                        which_output = numpy.array([self.output_map[each_output]==x for x in output])
                        exp_num_in_Si_Vk[self.input_map[each_input], : , self.output_map[each_output]] += \
                            Gamma[:, which_input & which_output].sum(1)



                if debug:
                    print "\nExpected frequency in state S_i at time 0:\n" + str(exp_si_t0)
                    print "\nExpected number of transition from state S_i:\n" + str(exp_num_from_Si)
                    print "\nExpected number of time in state S_i:\n" + str(exp_num_in_Si)
                    print "\nExpected number of time in state S_i observing V_k:\n" + str(exp_num_in_Si_Vk)

                # Given the parameters, what's the probability of staying at state Si at t
                # and at state Sj at t+1 and observation
                Xi = numpy.zeros([T-1, self.K, self.N, self.N], float)
                for t in xrange(T-1):
                    for i in xrange(self.N):
                        Xi[t, input[t], i, :] = Alpha[i,t] * self.T[input[t],i,:] * self.E[input[t+1],:, output[t+1]] * Beta[:,t+1]


                for t in xrange(T-1):
                    exp_num_Si_Sj[input[t]] += Xi[t,input[t],:,:]
                if debug:
                    print "\nExpected number of transitions from state Si to state Sj: \n" + str(exp_num_Si_Sj)

            # Reestimate model parameters #
            # Reestimate initial state probabilities
            if updatePi:
                self.Pi = exp_si_t0/exp_si_t0.sum()
                if debug:
                    print "\nUpdated Pi:"
                    print self.Pi
            if updateT:
                T_hat = numpy.zeros([self.K, self.N, self.N], float)
                for i in xrange(self.K):
                    for j in xrange(self.N):
                        T_hat[i,j,:] = exp_num_Si_Sj[i,j,:] / exp_num_from_Si[i,j]
                        T_hat[i,j,:] /= T_hat[i,j,:].sum()
                self.T = T_hat
                if debug:
                    print "\nUpdated T"
                    print self.T
            if len(self.F) != 0:
                for key in self.F.keys():
                    self.T[key[0],key[1],:] = self.F[key]
            if updateE:
                E_hat = numpy.zeros([self.K, self.N, self.M])
                for i in xrange(self.K):
                    for j in xrange(self.N):
                        E_hat[i,j,:] = exp_num_in_Si_Vk[i,j,:]/exp_num_in_Si[i,j]
                        E_hat[i,j,:] /= E_hat[i,j,:].sum()
                self.E = E_hat
                if debug:
                    print "\nUpdated E"
                    print self.E
            LLS.append(LogLikelihood)
            print "\nLog Likelihood for this iteration is " + str(LogLikelihood)
            if epoch > 1:
                if(abs(LLS[epoch]-LLS[epoch-1]) < epsilon):
                    print "\nThe loglikelihood improvement falls below threshold, training terminates at " \
                          "epoch " + str(epoch) + "!"

                    break

        if Val_seq != None:
            probs = 0
            for i in xrange(len(Val_seq[0])):
                inp = Val_seq[0][i]
                outp = Val_seq[1][i]
                prob, Alpha = self.forward(inp, outp, scaling = False, debug = False)
                probs += prob

        if debug:
            print "\n"*2+ "*"*24 + "\n" +"*"*1+" Validation  "+"*"*1 + "\n" + "*"*24 + "\n"
            print "Testing sequence loglikelihood is: "+ str(probs)
            print "Testing sequence probability is: "+ str(numpy.e ** probs)
        else:
            print "\nTesting sequence loglikelihood is: "+ str(probs)

        self.print_iohmm("UPDATED HMM ELEMENTS")






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
    input_seqs = []
    input_seqs.append(input_seq)
    input_seqs.append(input_seq)
    output_seqs = []
    output_seqs.append(output_seq)
    output_seqs.append(output_seq)

    iohmm.bawn_welch(input_seqs,output_seqs, debug= False, val = (input_seqs, output_seqs))