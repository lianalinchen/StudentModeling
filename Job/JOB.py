__author__ = 'adminuser'
from SeqGenerator.getSequence import SequenceGenerator
from HMM.HMM import HMM
import os
import numpy


class JOB:
    def __init__(self, filename, type, **args):
        self.filename = filename

        if type.lower() != "hmm" and type.lower() != "iohmm":
            raise ValueError("Please specify what type of model for training")
        self.type = type
        self.sequences = SequenceGenerator(self.filename, self.type)

    def run(self, symbol, **args):
        dic = {}
        if 'T' in args:
            self.T = args['T']
        if 'E' in args:
            self.E = args['E']
        if 'F' in args:
            self.F = args['F']
        if 'CV' in args:
            cv = args['CV']
        else:
            cv = False

        self.output_symbol = symbol
        # Get the training sequence
        s = self.sequences.getTrainSeq(debug=False)
        output = s[0]
        predict = s[1]

        if self.type == "hmm":
            if 'F' in args:
                pass
                #hmm = HMM(2, output=self.output_symbol, T=self.T, E=self.E, F=self.F)
            else:
                hmm = HMM(2, output=self.output_symbol, T=self.T, E=self.E)
            hmm.print_HMM("Original Parameters")
            if cv is True:
                #If using LOOCV
                validation_sum = 0
                dsize = len(output)
                for i in xrange(dsize):
                    train_set = output[:i]+output[i+1:]
                    val_set = []
                    val_set.append(output[i])
                    v = hmm.bawm_welch(train_set, debug=False, val=val_set)
                    validation_sum += v
                print "\nLeave One Out Cross Validation Testing set average Loglikelihood is:"
                print validation_sum/dsize

            else:

                #Training on the whole dataset, get the updated parameters
                updatedPrams = hmm.bawm_welch(output, debug=False,val=predict)






if __name__ == '__main__':

    job = JOB(os.path.dirname(os.path.dirname(__file__)) + "/Sequence/Cordillera/EX1/KC28_HMM.txt","hmm")

    symbols = ["0", "1"]
    Pi = numpy.array([0.5, 0.5])
    T = numpy.array([0.9,0.1,0.1,0.9]).reshape(2,2)
    E = numpy.array([0.9,0.1,0.1,0.9]).reshape(2,2)

    # Train the model, get the updated parameters

    # fixed param
    # param = job.run(symbols, Pi = Pi, T=T, E=E, F=F)

    # Do not fix params
    param = job.run(symbols, Pi = Pi, T=T, E=E)
    # Use the updated parameters to predict the possibility of
    #print "\nTRAINING RESULTS:\n"
    #print job.Pi
    #print job.T
    #print job.E

