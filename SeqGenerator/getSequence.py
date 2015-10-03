__author__ = 'adminuser'

class SequenceGenerator:
    def __init__(self, filename, type):
        self.filename = filename
        self.type = type

    def get(self, debug):
        if self.type.lower() == "hmm":
            return self.getHMMSeq(debug)

        elif self.type.lower() == "iohmm":
            return self.getIOHMMSeq(debug)

    def getHMMSeq(self, debug = False):
        with open(self.filename,'r') as f:
            read_date = f.readlines()

        seqs = {}

        for each in read_date:
            key = each.split()[0]
            if key not in seqs:
                seqs[key] = []
                seqs[key].append(each.split()[1])
            else:
                seqs[key].append(each.split()[1])

        seq = []
        min_length = float("inf")
        for each in seqs:
            length = len(seqs[each])
            if length < min_length:
                min_length = length

        for each in seqs:
            seq.append(seqs[each][:min_length])

        if debug:
            for each in seq:
                print each
            print "The length of sequence is "+ str(min_length)

        f.close()
        return seq

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
    sg = SequenceGenerator("./Sequence/KC1_HMM.txt","hmm")
    print sg.get(debug= False)

    sg2 = SequenceGenerator("./Sequence/KC1_IOHMM.txt","iohmm")
    print sg2.get(debug = False)

