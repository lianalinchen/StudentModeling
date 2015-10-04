"""
    Python module for creating, training and applying hidden
    Markov models to discrete or continuous observations.
    Author: Michael Hamilton,  hamiltom@cs.colostate.edu
    Theoretical concepts obtained from Rabiner, 1989.
    """

import numpy, pylab, time, copy, math
from numpy import random as rand
from numpy import linalg
from matplotlib import rc
#rc('text', usetex=True)
import os

class HMM_Classifier:
    """
        A binary hmm classifier that utilizes two hmms: one corresponding
        to the positive activity and one corresponding to the negative
        activity.
        """
    def __init__( self, **args ):
        """
            :Keywords:
            - `neg_hmm` - hmm corresponding to negative activity
            - `pos_hmm` - hmm corresponding to positive activity
            """
        self.neg_hmm = None
        self.pos_hmm = None
        
        if 'neg_hmm' in args:
            self.neg_hmm = args[ 'neg_hmm' ]
        
        if 'pos_hmm' in args:
            self.pos_hmm = args[ 'pos_hmm' ]
    
    def classify( self, sample ):
        """
            Classification is performed by calculating the
            log odds for the positive activity.  Since the hmms
            return a log-likelihood (due to scaling)
            of the corresponding activity, the difference of
            the two log-likelihoods is the log odds.
            """
        # Scream if an hmm is missing
        if self.pos_hmm == None or self.neg_hmm == None:
            raise "pos/neg hmm(s) missing"
        
        pos_ll = forward( self.pos_hmm, sample, scaling=1 )[ 0 ]
        neg_ll = forward( self.neg_hmm, sample, scaling=1 )[ 0 ]
        
        # log odds by difference of log-likelihoods
        return  pos_ll - neg_ll
    
    
    def add_pos_hmm( self, pos_hmm ):
        """
            Add the hmm corresponding to positive
            activity.  Replaces current positive hmm, if it exists.
            """
        self.pos_hmm = pos_hmm
    
    def add_neg_hmm( self, neg_hmm ):
        """
            Add the hmm corresponding to negative
            activity.  Replaces current negative hmm, if it exists.
            """
        self.neg_hmm = neg_hmm

class HMM:
    """
        Creates and maintains a hidden Markov model.  This version assumes the every state can be
        reached DIRECTLY from any other state (ergodic).  This, of course, excludes the start state.
        Hence the state transition matrix, A, must be N X N .  The observable symbol probability
        distributions are represented by an N X M matrix where M is the number of observation
        symbols.
        
        |a_11 a_12 ... a_1N|                   |b_11 b_12 ... b_1M|
        |a_21 a_22 ... a_2N|                   |b_21 b_22 ... b_2M|
        A = | .    .        .  |               B = | .    .        .  |
        | .         .   .  |                   | .         .   .  |
        |a_N1 a_N2 ... a_NN|                   |b_N1 b_N2 ... b_NM|
        
        a_ij = P(q_t = S_j|q_t-1 = S_i)       b_ik = P(v_k at t|q_t = S_i)
        where q_t is state at time t and v_k is k_th symbol of observation sequence
        
        
        """
    def __init__( self, n_states=1, **args ):
        """
            :Keywords:
            - `n_states` - number of hidden states
            - `V` - list of all observable symbols
            - `A` - transition matrix
            - `B` - observable symbol probability distribution
            - `D` - dimensionality of continuous observations
            - `F` - Fixed emission probabilities for the given state ( dict: i -> numpy.array( [n_states] ),
            where i is the state to hold fixed.
            """
        
        self.N = n_states # Number of hidden states
        
        # Initialize observable symbol set parameters
        self.V = args[ 'V' ]
        self.M = len( self.V )
        
        print "v!!!!!!"
        print self.V
        self.symbol_map = dict( zip ( self.V, range( len( self.V ) )) )
        
        
        # Initialize transition probability matrix
        if 'A' in args:
            self.A = args[ 'A' ]
            assert numpy.shape( self.A ) == ( self.N, self.N )
            self.A = (self.A.T /  self.A.T.sum(0)).T
        
        
        else:
            # Randomly initialize matrix and normalize so sum over a row = 1
            raw_A = rand.uniform( size = self.N * self.N ).reshape( ( self.N, self.N ) )
            #raw_A = numpy.array(1.0/self.N).repeat(self.N * self.N).reshape((self.N ,self.N))
            self.A = ( raw_A.T / raw_A.T.sum( 0 ) ).T
            if n_states == 1:
                self.A.reshape( (1,1) )
        
        # Initialize observable symbol probability distributions
        if 'B' in args:
            self.B = args[ 'B' ]
            if n_states > 1:
                assert numpy.shape( self.B ) == ( self.N, self.M )
                self.B = (self.B.T / self.B.T.sum(0)).T
            else:
                self.B = numpy.reshape(self.B, (1,self.M) )
            
            if 'F' in args:
                self.F = args[ 'F' ]
                for i in self.F.keys():
                    self.B[ i,: ] = self.F[ i ]
            else:
                self.F = {}
        
        else:
            # initialize distribution
            B_raw = rand.uniform( 0, 1, self.N * self.M ).reshape( ( self.N, self.M ) )
            #initialize to uniform distribution
            #B_raw = numpy.array(1.0/self.M).repeat(self.N * self.M).reshape((self.N ,self.M))
            self.B = ( B_raw.T / B_raw.T.sum( 0 ) ).T
            
            if 'F' in args:
                self.F = args[ 'F' ]
                for i in self.F.keys():
                    self.B[ i,: ] = self.F[ i ]
            else:
                self.F = {}
        
        
        # Initialize the intitial state distribution
        if 'Pi' in args:
            self.Pi = args[ 'Pi' ]
            assert len( self.Pi ) == self.N
            self.Pi  = (self.Pi.T / self.Pi.T.sum(0)).T
        else:
            # initialize to uniform distribution
            self.Pi = numpy.array ( 1.0 / self.N ).repeat( self.N )
        
        if 'Labels' in args:
            self.Labels = args[ 'Labels' ]
        else:
            self.Labels = range( self.N )
        
        if 'F' in args:
            self.F = args[ 'F' ]
            for i in self.F.keys():
                self.A[ i,: ] = self.F[ i ]
        else:
            self.F = {}
    
    def __repr__( self ):
        retn = '*'*29
        retn += "\nFive elements of HMM model\n"
        retn +='*'*29
        retn += "\nNumber of Hidden States: %d\n" % ( self.N ) + \
            "Observation Symbols: %s\n" % ( self.V ) + \
            "\nA(State Transition Probability):\n %s\n" % ( str( self.A ) ) + \
            "\nB(Observation Probability):\n %s\n" % ( str( self.B ) ) + \
            "\nPi(Initialization Porbability):\n %s" % ( str( self.Pi ) )
        
        return retn


def symbol_index( hmm, Obs ):
    """
        Converts an obeservation symbol sequence into a sequence
        of indices for accessing distribution matrices.
        """
    Obs_ind = []
    for o in Obs:
        Obs_ind.append( hmm.symbol_map[ o ] )
    
    return Obs_ind


def forward( hmm, Obs, scaling=True ):
    """
        Calculate the probability of an observation sequence, Obs,
        given the model, P(Obs|hmm).
        Obs: observation sequence
        hmm: model
        returns: P(Obs|hmm)
        """
    T = len( Obs ) # Number of states in observation sequence
    
    # Get index sequence of observation sequence to access
    # the observable symbol probabilty distribution matrix
    
    Obs = symbol_index( hmm, Obs )
    
    
    # create scaling vector
    if scaling:
        c = numpy.zeros( [ T ], float )
    
    # Base Case:
    Alpha = numpy.zeros( [ hmm.N, T ], float )


    
    Alpha[ :,0 ] = hmm.Pi * hmm.B[ :,Obs[ 0 ] ]
    
    if scaling:
        c[ 0 ] = 1.0 / numpy.sum( Alpha[ :,0 ] )
        Alpha[ :,0 ] = c[ 0 ] * Alpha[ :,0 ]
    
    # Induction Step:
    for t in xrange( 1,T ):
        Alpha[ :,t ] = numpy.dot( Alpha[ :,t-1 ], hmm.A) * hmm.B[ :,Obs[ t ] ]

        
        if scaling:
            #print "\nAlpha[ :,t ]"
            #print Alpha[ :,t ]
            c[ t ] =  1.0 / numpy.sum( Alpha[ :,t ] )
            Alpha[ :,t] = Alpha[ :,t]  * c[ t ]
        
   
    if scaling:
        log_Prob_Obs = -( numpy.sum( numpy.log( c ) ) )
    
        
        return ( log_Prob_Obs, Alpha, c )
    else:
        prob_Obs = numpy.sum( Alpha[ :,T-1 ] )

        return ( prob_Obs, Alpha )


def backward( hmm, Obs, c=None ):
    """
        Calculate the probability of a partial observation sequence
        from t+1 to T, given some state t.
        Obs: observation sequence
        hmm: model
        c: the scaling coefficients from forward algorithm
        returns: B_t(i)
        """
    T = len( Obs ) # Number of states in observation sequence
    
    # Get index sequence of observation sequence to access
    # the observable symbol probabilty distribution matrix
    Obs = symbol_index( hmm, Obs )
    
    # Base Case:
    Beta = numpy.zeros( [ hmm.N, T ], float )
    Beta[ :, T-1 ] = 1.0
    if c is not None:
        Beta [ :,T-1  ] = Beta [ :,T-1 ] * c[ T-1 ]
    
    #print "Inductive step"
    count = 0
    for t in reversed( xrange( T-1 ) ):
        count = count+1
        Beta[ :,t ] = numpy.dot( hmm.A, ( hmm.B[ :,Obs[ t+1 ] ] * Beta[ :,t+1 ] ) )
     
        
        if c is not None:
            Beta[ :,t ] = Beta[ :,t ] * c[ t ]
    
    return Beta


def viterbi( hmm, Obs, scaling=True ):
    """
        Calculate P(Q|Obs, hmm) and yield the state sequence Q* that
        maximizes this probability.
        Obs: observation sequence
        hmm: model
        """
    T = len( Obs ) # Number of states in observation sequence
    
    # Get index sequence of observation sequence to access
    # the observable symbol probabilty distribution matrix
    Obs = symbol_index( hmm, Obs )
    
    # Initialization
    # Delta[ i,j ] = max_q1,q2,...,qt P( q1, q2,...,qt = i, O_1, O_2,...,O_t|hmm )
    # this is the highest prob along a single path at time t ending in state S_i
    Delta = numpy.zeros( [ hmm.N,T ], float)
    
    if scaling:
        Delta[ :,0 ] = numpy.log( hmm.Pi ) + numpy.log( hmm.B[ :,Obs[ 0] ] )
    
    
    else:
        Delta[ :,0 ] = hmm.Pi * hmm.B[ :,Obs[ 0] ]
    
    
    # Track Maximal States
    Psi =  numpy.zeros( [ hmm.N, T ], int )
    
    #print "Inductive Steps:
    if scaling:
        for t in xrange( 1,T ):
            nus =  Delta[ :,t-1 ] + numpy.log( hmm.A )
            Delta[ :,t ] =  nus.max(1) + numpy.log( hmm.B[ :,Obs[ t ] ] )
            Psi[ :,t ] = nus.argmax( 1 )
    else:
        for t in xrange( 1,T ):
            nus =  Delta[ :,t-1 ] * hmm.A
            '''print "hmm.A"
                print hmm.A
                print "hmm.B"
                print hmm.B
                print "Delta"
                print Delta
                print "Delta[:,t-1]"
                print Delta[:,t-1]
                print "nus"
                print nus'''
            Delta[ :,t ] = nus.max( 1 ) * hmm.B[ :,Obs[ t ] ]
            '''print "nus.max( 1 ) "
                print nus.max( 1 )
                print "hmm.B[ :,Obs[ t ] ]"
                print hmm.B[ :,Obs[ t ] ]
                print "Delta[ :,t ] "
                print Delta[ :,t ]'''
            Psi[ :,t ] = nus.argmax(1)
            '''print "Psi[ :,t ] "
                print Psi[ :,t ]'''
    
    # Calculate State Sequence, Q*:
    Q_star =  [ numpy.argmax( Delta[ :,T-1 ] ) ]
    for t in reversed( xrange( T-1 ) ) :
        Q_star.insert( 0, Psi[ Q_star[ 0 ],t+1 ] )
    
    return ( Q_star, Delta, Psi )


def baum_welch( hmm, Obs_seqs, **args ):
    """
        EM algorithm to update Pi, A, and B for the HMM
        :Parameters:
        - `hmm` - hmm model to train
        - `Obs_seqs` - list of observation sequences to train over
        :Return:
        a trained hmm
        
        :Keywords:
        - `epochs` - number of iterations to perform EM, default is 20
        - `val_set` - validation data set, not required but recommended to prevent over-fitting
        - `updatePi` - flag to update initial state probabilities
        - `updateA` - flag to update transition probabilities, default is True
        - `updateB` - flag to update observation emission probabilites for discrete types, default is True
        - `scaling` - flag to scale probabilities (log scale), default is True
        - `graph` - flag to plot log-likelihoods of the training epochs, default is False
        - `normUpdate` - flag to use 1 / -(normed log-likelihood) contribution for each observation
        sequence when updating model parameters, default if False
        - `fname` - file name to save plot figure, default is ll.eps
        - `verbose` - flag to print training times and log likelihoods for each training epoch, default is false
        """
    # Setup keywords
    if 'epochs' in args: epochs = args[ 'epochs' ]
    else: epochs =20
    
    updatePi=updateA=updateB=scaling=graph = 1
    normUpdate=verbose=validating = 0
    
    if 'updatePi' in args: updatePi = args[ 'updatePi' ]
    if 'updateA' in args: updateA = args[ 'updateA' ]
    if 'updateB' in args: updateB = args[ 'updateB' ]
    if 'scaling' in args: scaling = args[ 'scaling' ]
    if 'graph' in args: graph = args[ 'graph' ]
    if 'normUpdate' in args: normUpdate = args[ 'normUpdate' ]
    if 'fname' in args: fname = args[ 'fname' ]
    else: fname = 'll.eps'
    if 'verbose' in args: verbose = args[ 'verbose' ]
    if 'val_set' in args:
        validating = 1
        val_set = args[ 'val_set' ]
    
    K = len( Obs_seqs ) # number of observation sequences
    start = time.time() # start training timer
    LLs = []            # keep track of log likelihoods for each epoch
    val_LLs = []        # keep track of validation log-likelihoods for each epoch
    
    # store best parameters
    best_A = copy.deepcopy( hmm.A )
    best_B = copy.deepcopy( hmm.B )
    best_Pi = copy.deepcopy( hmm.Pi )
    best_epoch = 'N/A'
    
    best_val_LL = None
    
    # Iterate over specified number of EM epochs
    for epoch in xrange( epochs ):
        start_epoch = time.time()                                 # start epoch timer
        print "EPOCH %d"%epoch
        
        LL_epoch = 0                                              # intialize log-likelihood of all seqs given the model
        Expect_si_all = numpy.zeros( [ hmm.N ], float )           # Expectation of being in state i at each time t
        Expect_si_all_TM1 = numpy.zeros( [ hmm.N ], float )       # Expectation of being in state i over all seqs until T-1
        Expect_si_sj_all = numpy.zeros( [ hmm.N, hmm.N ], float ) # Expectation of transitioning from state i to state j at each time t
        Expect_si_sj_all_TM1 = numpy.zeros( [ hmm.N, hmm.N ], float ) # Expectation of transitioning from state i to state j until T-1
        Expect_si_t0_all = numpy.zeros( [ hmm.N ] )               # Expectation of in state Si at time (t=1)
        Expect_si_vk_all = numpy.zeros( [ hmm.N, hmm.M ], float ) # Expectation of being in state i and seeing symbol vk
        ow = 0
        for Obs in Obs_seqs:
            if ow > 0 and ow % 100 == 0:
                print "epoch %d: %d seqs processed" % ( epoch+1, ow )
            ow += 1
            #Obs = list( Obs )
            print "The obeservation sequence is"
            print Obs
            log_Prob_Obs, Alpha, c = forward( hmm=hmm, Obs=Obs, scaling=1 )  # Calculate forward probs, log-likelihood, and scaling vals, c is an array of scaling parameter
            Beta = backward( hmm=hmm, Obs=Obs, c=c )                         # Calculate backward probs
            LL_epoch += log_Prob_Obs                                         # Update overall epoch log-likelihood
            T = len( Obs )                                                   # Number of states in observation sequence
            
            # Determine update weight of the observation for contribution
            # to model parameter maximization
            
            if normUpdate:
                w_k = 1.0 / -( log_Prob_Obs + numpy.log( len( Obs ) ) )
            else:
                w_k = 1.0
            
            # Get index sequence of observation sequence to access
            # the observable symbol probabilty distribution matrix
            Obs_symbols = Obs[ : ]
            Obs = symbol_index( hmm, Obs )
            
            # Calculate gammas: The probability in state Si at time t, given the observation sequence
            # Gamma[ i,t ] = P( q_t = S_i|Obs, hmm)
            print "Alpha \n"
            print Alpha
            print "Beta \n"
            print Beta
            Gamma_raw = Alpha * Beta
            Gamma = Gamma_raw / Gamma_raw.sum( 0 )
            print "Gamma: \n"
            print Gamma
            
            Expect_si_t0_all += w_k * Gamma[ :,0 ]
            Expect_si_all += w_k * Gamma.sum( 1 )
            Expect_si_all_TM1 += w_k * Gamma[ :,:T-1 ].sum( 1 )
            print "Si at t0 " + str(Expect_si_t0_all)
            print "at Si " + str(Expect_si_all)
            print "From Si " + str(Expect_si_all_TM1)
            
            
            # Calculate Xis
            # Xi is an N X N X T-1 matrix corresponding to
            # Xi[ i,j,t ] = P(q_t = S_i, q_t+1 = S_j|Obs, hmm )
            Xi = numpy.zeros( [ hmm.N, hmm.N, T-1 ], float )
            
            for t in xrange( T-1 ):
                for i in xrange( hmm.N ):
                    Xi[ i,:,t ] = Alpha[ i,t ] * hmm.A[ i,: ] * hmm.B[ :, Obs[ t+1 ] ] * Beta[ :,t+1 ]
                
                if not scaling:
                    Xi[ :,:,t ] = Xi[ :,:,t ] / Xi[ :,:,t ].sum()

            print "Xi" + str(Xi[ :,:,:T-1])
            
            

            # Expect_si_sj_all = expected number of transitions from state s_i to state s_j

            Expect_si_sj_all_TM1 += w_k * Xi[ :,:,:T-2].sum( 2 )



            print "Si to Sj t-1" + str(Expect_si_sj_all_TM1)
            
            
            if updateB:
                B_bar = numpy.zeros( [ hmm.N, hmm.M ], float )
                
                for k in xrange( hmm.M ):
                    which = numpy.array( [ hmm.V[ k ] == x for x in Obs_symbols ] )
                    #Gamma.T[ which,: ]: At time t, the probability of being in state i or j, make observation of O1, next loop, calculate the probability making observation of O2
                    B_bar[ :,k ] = Gamma.T[ which,: ].sum( 0 )
                Expect_si_vk_all += w_k * B_bar

        
        ##############    Reestimate model parameters    ###############
        
       
        
        # reestimate transition probabilites
        if updateA:
            A_bar = numpy.zeros( [ hmm.N, hmm.N ], float )
            for i in xrange( hmm.N ):
                A_bar[ i,: ] = Expect_si_sj_all_TM1[ i,: ] / Expect_si_all_TM1[ i ]
            hmm.A = A_bar
            print "Update A:"
            print hmm.A
        
        
        if updateB:
            # reestimate emission probabilites
            # ( observable symbol probability distribution )
            for i in xrange( hmm.N ):
                Expect_si_vk_all[ i,: ] = Expect_si_vk_all [ i,: ] / Expect_si_all[ i ]
            
            hmm.B = Expect_si_vk_all
            print "\nUpdate B"
            print hmm.B
            
            
            for i in hmm.F.keys():
                hmm.B[ i,: ] = hmm.F[ i ]
         # reestimate initial state probabilites
        if updatePi:
            Expect_si_t0_all = Expect_si_t0_all / numpy.sum( Expect_si_t0_all )
            hmm.Pi = Expect_si_t0_all
            print "\nUpdate Pi:"
            print hmm.Pi
        
        LLs.append( LL_epoch )
        print "Log likelihood at Epoch " + str(epoch)+" is : "+str(LL_epoch)
        print LLs
        print "\n"*3
        
        if epoch>=1:
          
            if math.fabs(LLs[-1]-LLs[-2])<0.01:
                 print "Program terminates at epoch:",epoch
                 print "Best validaton log likelihood probability is ",LLs[-1]
                 break
        
    if validating:  
        val_LL_epoch=0
        for v in val_set:
            val_LL_epoch += forward( hmm=hmm, Obs=v, scaling=1 )[0]  # Calculate forward probs, log-likelihood, and scaling vals, c is an array of scaling parameter   

    return val_LL_epoch

######################
#Get the observation sequence#
######################
def getObs():
    f = open(os.path.dirname(os.path.dirname(__file__)) + "/HMM/Sequence/KC1_HMM.txt",'r')
    Obs_seq = []
    Obs = []
    stu= f.readline().split()
    Obs.append(int(stu[1]))
    count = 0
    for line in f:
        count = count+1
        record = line.split()
        if  record[0]==stu[0]:
            Obs.append(int(record[1]))
        else:
            print record[0]
            stu = record
            Obs_seq.append(Obs)
            Obs = []
            Obs.append(int(record[1]))
    Obs_seq.append(Obs)
    print "Count!!!!!!!!!!!!!!!!"
    print count 
    return Obs_seq














##############################
###         Test                            ###########
##############################

def stu_test(graph= True):
    V= [0,1]

    
    A = numpy.array(  [[ 0.7,  0.3],\
 [ 0.3,  0.7]])
    
    B = numpy.array( [[ 0.7  ,0.3],\
 [ 0.2 , 0.8]])
    
    Pi =numpy.array([ 0.5,  0.5])

                
    hmm = HMM (2, A=A,B=B,Pi=Pi, V=V )
    
    
    ######################
    #      Hmm model elements:      #
    ######################
    print hmm.__repr__()
    #Adjust the precision of printing float
    numpy.set_printoptions(precision=4)
    #######################
    #Baum Welch AlgorithmResults:#
    #######################
    print '*'*29
    print " Baum Welch Algorithm Results:"
    print '*'*29
    c = getObs()
    #Leave one out validation 
    looc(hmm,c)
    #updated_hmm = baum_welch(hmm,c, val_set = c ,graph=graph,verbose = True ) 
    
def looc(hmm, c):
    val_sum = 0
    for i in range(len(c)):
        print "#"*15
        train= c[:i]+c[i+1:]
        val = []
        val.append(c[i])
        v = baum_welch(hmm,train,val_set=val,verbose = True )
        print "the probability of validation set is" +str(v)
        val_sum += v
        print val_sum
    print "The average probilaity of validation set is:"
    print val_sum/len(c) 

    
def predict_post_test():
    V= [0,1]

    A = numpy.array ([[   0.8612,  0.1388],\
 [  0.0596 , 0.9404]])
    
    B = numpy.array([[  0.8697 , 0.1303],\
  [  0.0454  ,0.9546]])
     
    Pi = numpy.array ([  0.2865 , 0.7135])
                                        
    hmm = HMM (2, A=A,B=B,Pi=Pi, V=V ) 
    
    c = getObs()
    
    #print c 
    
    count = 0
    for each in c:
        log_probability, Alpha,c = forward(hmm=hmm, Obs=each, scaling=1 )
        print Alpha[1,-1] 
############################################################################
if __name__ == "__main__":
    symbols = ["wrong", "correct"]
    T = numpy.array([0.9,0.1,0.2,0.8]).reshape(2,2)
    E = numpy.array([0.9,0.1,0.9,0.1]).reshape(2,2)
    F = {1:[0,1]}
    hmm = HMM(n_states=2, V = symbols, A = T, B = E)

    Obs = ["correct","correct","wrong","correct","correct","wrong"]
    Obs2 = ["correct","correct","wrong","correct","correct","wrong"]
    Obs3 = ["correct","correct","wrong","correct","correct","wrong"]

    Obs_seq = []
    Obs_seq.append(Obs)
    Obs_seq.append(Obs2)
    Obs_seq.append(Obs3)

    #print forward(hmm,Obs)
    result = baum_welch(hmm,Obs_seq, val_set=Obs_seq)
    print result