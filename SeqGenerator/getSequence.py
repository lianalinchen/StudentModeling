__author__ = 'adminuser'
import os
class SequenceGenerator:
    def __init__(self, filename, type):
        self.filename = filename
        self.type = type

    def getTrainSeq(self, debug):
        if self.type.lower() == "hmm":
            return self.getHMMSeq(debug)

        elif self.type.lower() == "iohmm":
            return self.getIOHMMSeq(debug)

    def getHMMSeq(self, debug = False):
        with open(self.filename,'r') as f:
            read_date = f.readlines()

        seqs = {}

        for each in read_date:
            key = each.split()[0][7:]
            if key not in seqs:
                seqs[key] = []
                seqs[key].append(each.split()[1])
            else:
                seqs[key].append(each.split()[1])


        #Training sequence will have the same length while full sequence will remain its original length.
        train_seq = []
        full_seq = []

        min_length = float("inf")
        for each in seqs:
            length = len(seqs[each])
            if length < min_length:
                min_length = length

        #Sorted the key to sequence is the same order
        for each in sorted(seqs.iterkeys(), key=int):
            train_seq.append(seqs[each][:min_length])
            full_seq.append(seqs[each])
        print "\n\n"

        if debug:
            for each in seq:
                print each
            print "The length of sequence is "+ str(min_length)

        f.close()
        return (train_seq, full_seq)

    def getIOHMMSeq(self, debug = True):
        with open(self.filename,'r') as f:
            read_date = f.readlines()

        inputs = {}
        outputs = {}

        for each in read_date:
            key = each.split()[0]
            if key not in inputs:
                inputs[key] = []
                outputs[key] = []
            inputs[key].append(each.split()[1])
            outputs[key].append(each.split()[2])

        input = []
        output = []
        min_length = float("inf")
        for each in inputs:
            length = len(inputs[each])
            if length < min_length:
                min_length = length


        for each in inputs:
            input.append(inputs[each][:min_length])

        for each in outputs:
            output.append(outputs[each][:min_length])

        if debug:
            print inputs
            for i in xrange(len(input)):
                print input[i]
                print output[i]

        return (input, output)


if __name__ == '__main__':
    sg = SequenceGenerator(os.path.dirname(os.path.dirname(__file__)) + "/Sequence/Cordillera/EX1/KC1_HMM.txt","hmm")
    result =  sg.getHMMSeq(debug= False)
    for each in result[0]:
        print each

    for each in result[1]:
        print each


