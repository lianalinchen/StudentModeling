__author__ = 'adminuser'

import numpy
import copy
from numpy import random as rand

class HMM:
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

    def __init__(self, n_states=1, **args):
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

        # Initialize observable symbols set parameters
        self.V = args['V']
        self.M = len(self.V)
        self.symbol_map = dict( zip ( self.V, range( len( self.V ) )) )



        # Initialize transmission probability matrix
        if 'T' in args:
            self.T = args['T']

            if numpy.shape( self.T ) != (self.N, self.N):
                raise ValueError("The transmission probability matrix dimension mismatches the given states number.")

            if not numpy.array_equal(self.T.sum(1), numpy.array([1.0]*len(self.T.sum(1)))):
                raise ValueError("The sum of each row in the transimission matrix should equal to 1")
        else:
            raw_T = rand.uniform(0,1, self.N * self.N).reshape(self.N, self.N)
            raw_T_sum = raw_T.sum(axis = 1, keepdims = True)
            self.T = raw_T.astype(float)/raw_T_sum




        # Initialize emission probability matrix
        if 'E' in args:
            self.E = args['E']

            if numpy.shape(self.E) != (self.N, self.M):
                raise ValueError("The emission probaility matrix dimension mismatches the given states number and "
                                 "output symbols number")

            if not numpy.array_equal(self.E.sum(1), numpy.array([1.0] * len(self.E.sum(1)))):
                raise ValueError("The sum of each row in the emission probability matrix should equal to 1")
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

    def print_HMM(self, label):
        """
        Print out the HMM elements
        """
        print "\n"*2+ "*"*24 + "\n" + label  + "\n" + "*"*24 + "\n"
        print "\n1) Numerber of hidden states:" + str(self.N)
        print "\n2) Number of observable symbols:" + str(self.V)
        print "\n3) The symbol mapping in HMM:" + str(self.symbol_map)
        print "\n4) The transmission proability matrix T:\n" + str(self.T)
        print "\n5) The emission probability matrix E:\n" + str(self.E)
        print "\n6) The initial state probability Pi: \n" + str(self.Pi)

    def obs_index(self, Obs):
        obs_index_seq= []
        for o in Obs:
            if o not in self.symbol_map:
                raise ValueError("The observation symbol \"" +o+ "\" is not defined in HMM")
            obs_index_seq.append(self.symbol_map[o])
        return obs_index_seq

    def forward(self, Obs, scaling = False, debug = False):
        """
        Calculate the probability of an observation sequence given the model parameters
        P(Obs|hmm)

        Alpha is defined as P(O_1:T,S_T|hmm)

        :param Obs: List. Observation sequence
        :param scaling: boolean. Scale the Alpha matrix to let the column sums to 1
        :param debug: boolean. Whether to print output of each step

        :return:
        """
        if debug:
            print "\n"*2+ "*"*23 + "\n" +"*"*2+" FORWARD ALGORITHM "+"*"*2 + "\n" + "*"*23 + "\n"

        Obs = self.obs_index(Obs)
        T = len(Obs)
        #create scaling vector
        if scaling:
            c = numpy.zeros([T],float)

        # Initialization
        Alpha = numpy.zeros([self.N, T], float)
        Alpha[:,0] =  self.Pi * self.E[:,Obs[0]]


        if debug:
            print "t=0"
            print Alpha[:,0]

        # Induction
        for t in xrange(1,T):
            Alpha[:,t] = numpy.dot(Alpha[:,t-1],self.T) * self.E[:,Obs[t]]
            if debug:
                print "t=" + str(t)
                print Alpha[:,t]



        # Termination
        if scaling:
            c = 1.0/ Alpha.sum(0)

            Alpha= Alpha * c
            log_prob = -numpy.log(c[T-1])

            if debug:
                print "\nAlpha:"
                print Alpha
                print "\nc:"
                print c
                print "\nP(Obs|hmm)=" + str(log_prob)
            return(log_prob, Alpha, c)

        else:
            prob = numpy.log(numpy.sum(Alpha[:,T-1]))
            if debug:
                print "\nAlpha:"
                print Alpha
                print "\nP(Obs|hmm)=" + str(prob)
            return(prob, Alpha)

    def backward(self, Obs, debug = False, **args):
        """
        Calculate the probability of a partial observation sequence from t+1 to T given the model params.

        Beta is defined as P(O_1:T|S_T, hmm)

        :param Obs: Observation sequence
        :return: Beta
        """
        if debug:
            print "\n"*2+ "*"*24 + "\n" +"*"*2+" BACKWARD ALGORITHM "+"*"*2 + "\n" + "*"*24 + "\n"

        Obs = self.obs_index(Obs)
        T = len(Obs)

        # Initialization
        Beta = numpy.zeros([self.N, T], float)
        Beta[:,T-1] = 1.0
        if debug:
            print "t=" + str(T-1)
            print Beta[:,T-1]

        # Induction
        for t in reversed(xrange(T-1)):

            #Beta[:,t] = numpy.dot(self.T, Beta[:,t+1]* self.E[:,Obs[t+1]])
            Beta[:,t] = (self.T * self.E[:,Obs[t+1]] * Beta[:,t+1]).sum(0)
            if debug:
                print "t=" + str(t)
                print Beta[:,t]

        if 'c' in args:
            Beta = Beta * args['c']

        if debug:
            print "\nBeta:"
            print Beta

        return Beta

    def viterbi(self, Obs):
        """
        Find the single best state sequence Q = {q_1, q_2, q_3, ... , q_t} up to time t,
        which accounts for the first t observations and and ends in q_t

        Define delta_t(i) = max P(q1,q2,q3,...,qt=i,O1,O2,...,Ot|hmm)
        We need to keep track of the argument maximize delta_t(i) at each time t and store it in the array psi.

        :param Obs: Observation sequence
        :return:
        """

        print "\n"*2+ "*"*23 + "\n" +"*"*2+" VITERBI ALGORITHM "+"*"*2 + "\n" + "*"*23 + "\n"

        Obs = self.obs_index(Obs)
        T = len(Obs)
        Delta = numpy.zeros([self.N, T],float)
        Psi = numpy.zeros([self.N, T], int)

        # Initialization
        Delta[:,0] = self.Pi * self.E[:,Obs[0]]
        print "t=0"
        print Delta[:,0]

        # Induction
        for t in xrange(1,T):
            nus = (Delta[:,t-1] * self.T.T).T
            Psi[:,t] = nus.argmax(0)
            Delta[:,t] = nus.max(0) * self.E[:,Obs[t]]
            print "t=" + str(t)
            print Delta[:,t]

        # Retrieve State Sequence Q*:
        Q_star = [numpy.argmax(Delta[:,T-1])]
        for t in reversed (xrange(T-1)):
            Q_star.insert(0, Psi[Q_star[0], t+1])

        print "\nDelta:"
        print Delta
        print "\nPsi:"
        print Psi
        print "\nQ_star"
        print Q_star

        return (Q_star, Delta, Psi)

    def bawm_welch(self, Obs_seq, **args):
        """
        Adjust the model parameters to maximize the probability of the observation sequence given the model

        Define:

        Gamma_t(i) = P(O_1:T, q_t = S_i | hmm) as the probability of in state i at time t and having the
        observation sequence.

        Xi_t(i,j) = P(O_1:T, q_t-1 = S_i, q_t = S_j | hmm) as the probability of transiting from state i
        to state j and having the observation sequence.


        :param Obs_seq: A set of observation sequence
        :param args:
            epochs: number of iterations to perform EM, default is 20
        :return:
        """
        print "\n"*2+ "*"*24 + "\n" +"*"*1+" Bawn Welch ALGORITHM "+"*"*1 + "\n" + "*"*24 + "\n"

        epochs = args['epochs'] if 'epochs' in args else 20
        updatePi = args['updatePi'] if 'updatePi' in args else True
        updateT = args['updateT'] if 'updateT' in args else True
        updateE = args['updateE'] if 'updateE' in args else True
        debug = args['debug'] if 'debug' in args else False
        epsilon = args['epsilon'] if 'epsilon' in args else 0.001
        Val_seq = args['val'] if 'val' in args else None

        LLs = []

        for epoch in xrange(epochs):

            print "-"*10 + "\n"+ "Epoch "+str(epoch)+"\n"+"-"*10

            exp_si_t0 = numpy.zeros([self.N],float)
            exp_num_from_Si = numpy.zeros([self.N],float)
            exp_num_in_Si = numpy.zeros([self.N],float)
            exp_num_Si_Sj = numpy.zeros([self.N * self.N],float).reshape(self.N, self.N)
            exp_num_in_Si_Vk = numpy.zeros([self.N, self.M], float)
            LogLikelihood = 0

            for Obs in Obs_seq:
                if debug:
                    print "\nThe observation sequence is: "+ str(Obs)

                #log_prob, Alpha, c = self.forward(Obs, scaling= True, debug= False)
                log_prob, Alpha = self.forward(Obs, scaling= False, debug= False)

                Beta = self.backward(Obs, debug= False)
                LogLikelihood += log_prob

                if debug:
                    print "\nAlpha:"
                    print Alpha
                    print "\nBeta:"
                    print Beta

                T = len(Obs)
                Obs = self.obs_index(Obs)

                raw_Gamma = Alpha * Beta
                Gamma = raw_Gamma/raw_Gamma.sum(0)
                if debug:
                    print "\nGamma"
                    print Gamma

                exp_si_t0+= Gamma[:,0]
                exp_num_from_Si += Gamma[:,:T-1].sum(1)
                exp_num_in_Si += Gamma.sum(1)

                temp = numpy.zeros([self.N, self.M], float)

                for each in self.symbol_map.iterkeys():
                    which = numpy.array([self.symbol_map[each] == x for x in Obs])
                    temp[:,self.symbol_map[each]] = Gamma.T[which,:].sum(0)
                exp_num_in_Si_Vk += temp

                if debug:
                    print "\nExpected frequency in state S_i at time 0:\n" + str(exp_si_t0)
                    print "Expected number of transition from state S_i:\n" + str(exp_num_from_Si)
                    print "Expected number of time in state S_i:\n" + str(exp_num_in_Si)
                    print "Expected number of time in state S_i observing V_k:\n" + str(exp_num_in_Si_Vk)

                Xi = numpy.zeros([T-1, hmm.N, hmm.N],float)
                for t in xrange(T-1):
                    for i in xrange(self.N):
                        Xi[t,i,:] = Alpha[i,t] * self.T[i,:] * self.E[:,Obs[t+1]] * Beta[:,t+1]
                        Xi[t,i,:] /= Xi[t,i,:].sum()
                    #scale = Xi[t,:,:].sum()
                    #Xi[t,:,:] /=  scale
                if debug:
                    print "\nXi:"
                    print Xi

                for t in xrange(T-2):
                    exp_num_Si_Sj += Xi[t,:,:]

                if debug:
                    print "\nExpected number of transitions from state Si to state Sj: \n"+str(exp_num_Si_Sj)



            #Reestimate model parameters #
            #reestimate initial state probabilities
            if updatePi:
                self.Pi = exp_si_t0/exp_si_t0.sum()
                if debug:
                    print "\nUpdated Pi:"
                    print exp_si_t0

            if updateT:
                T_hat = numpy.zeros([self.N, self.N],float).reshape(self.N, self.N)
                for i in xrange(self.N):
                    T_hat[i,:] = exp_num_Si_Sj[i,:]/exp_num_from_Si[i]
                    T_hat[i,:] /= T_hat[i,:].sum()
                self.T = T_hat

                if debug:
                    print "\nUpdated T"
                    print self.T

            if len(self.F) != 0:
                for i in self.F.keys():
                    self.T[i,:] = self.F[i]



            if updateE:
                E_hat = numpy.zeros([self.N, self.M],float).reshape(self.N, self.M)
                for i in xrange(self.N):
                    E_hat[i, :] = exp_num_in_Si_Vk[i,:]/exp_num_in_Si[i]
                    E_hat[i, :] /= E_hat[i, :].sum()
                self.E = E_hat
                if debug:
                    print "\nUpdated E"
                    print self.E

            LLs.append(LogLikelihood)
            print "\nlog likelihood for this iteration is " + str(LogLikelihood)

            if epoch > 1:
                if(abs(LLs[epoch]-LLs[epoch-1]) < epsilon):
                    print "\nThe loglikelihood improvement falls below threshold, training terminates! "
                    if debug is True:
                        self.print_HMM("UPDATED HMM ELEMENTS")
                    break

        # Calculate Loglikelihood of a test sequence
        if Val_seq != None:
            probs = 0
            for Obs in Val_seq:
                L = len(Obs)
                prob, Alpha = self.forward(Obs, scaling= False, debug=False)
                probs += prob

        if debug:
            print "\n"*2+ "*"*24 + "\n" +"*"*1+" Validation  "+"*"*1 + "\n" + "*"*24 + "\n"
            print "Testing sequence loglikelihood is: "+ str(probs)
            print "Testing sequence probability is: "+ str(numpy.e ** probs)
        else:
            print "\nTesting sequence loglikelihood is: "+ str(probs)

        self.print_HMM("Updated HMM")


if __name__ == '__main__':
    symbols = ["wrong", "correct"]
    T = numpy.array([0.9,0.1,0.2,0.8]).reshape(2,2)
    E = numpy.array([0.9,0.1,0.9,0.1]).reshape(2,2)
    F = {1:[0,1]}
    hmm = HMM(2, T=T , E=E, V = symbols, F=F)
    hmm.print_HMM("ORIGINAL HMM ELEMENTS")
    Obs = ["correct","correct","wrong","correct","correct","wrong","wrong","correct","correct","correct","correct","correct"]
    Obs2 = ["correct","correct","wrong","correct","correct","wrong","wrong","correct","correct","correct","correct","correct"]
    Obs3 = ["correct","correct","wrong","correct","correct","wrong","wrong","correct","correct","correct","correct","correct"]
    test = ["correct","correct","wrong","wrong","wrong","wrong"]
    test2 = ["correct","wrong","wrong","wrong","wrong","wrong"]
    #hmm.forward(Obs, scaling= False, debug= True)
    #hmm.backward(Obs, debug= True)
    #hmm.viterbi(Obs)
    Obs_seq = []
    Obs_seq.append(Obs)
    Obs_seq.append(Obs2)
    Obs_seq.append(Obs3)

    val_seq = []
    val_seq.append(test)
    val_seq.append(test2)
    hmm.bawm_welch(Obs_seq, debug = False, epochs= 10, val = val_seq)