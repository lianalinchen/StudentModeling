import unittest
from unittest import TestCase

import numpy

from IOHMM.IOHMM import IOHMM

__author__ = 'adminuser'


class TestIOHMM(TestCase):

    def testInputOutputSymbol(self):
        input = ["elicit", "tell"]
        output = ["wrong", "correct", "told"]
        iohmm = IOHMM(2, input = input,  output = output)
        self.assertEqual(iohmm.N,2,"The states number is wrong")

        self.assertEqual(iohmm.U, input, "The input symbols are incorrect")
        self.assertEqual(iohmm.K, 2, "The number of input symbols is incorrect")
        self.assertEqual(iohmm.input_map, {"elicit":0, "tell":1}, "The input map is incorrect")

        self.assertEqual(iohmm.V, output, "The output symbols are incorrect")
        self.assertEqual(iohmm.M, 3, "The number of output symbols is incorrect")
        self.assertEqual(iohmm.output_map, {"wrong":0, "correct":1, "told":2}, "The output map is incorrect")

    def testDefaultValue(self):
        input = ["elicit", "tell"]
        output = ["wrong", "correct", "told"]
        iohmm = IOHMM(2, input = input,  output = output)
        # Test Default Transmission probability
        self.assertTrue(numpy.allclose(iohmm.T[0].sum(1), numpy.array([1.0,1.0])))
        self.assertTrue(numpy.allclose(iohmm.T[1].sum(1), numpy.array([1.0,1.0])))
        # Test Default Emission Probability
        self.assertEqual(iohmm.E.shape, (2,2,3))
        self.assertTrue(numpy.allclose(iohmm.E[0].sum(1), numpy.array([1.0,1.0])))
        self.assertTrue(numpy.allclose(iohmm.E[1].sum(1), numpy.array([1.0,1.0])))
        # Fixed probability
        self.assertEqual(iohmm.F,{})
        # Initial State Distribution
        self.assertTrue(numpy.array_equal(iohmm.Pi,[0.5,0.5]))

    def testTransProbabilityInvalid(self):
        input = ["elicit", "tell"]
        output = ["wrong", "correct", "told"]
        T = numpy.array([0.1,0.4,0.2,0.8,0.3,0.7,0.4,0.6]).reshape(2,2,2)
        with self.assertRaises(ValueError) as context:
            IOHMM(2, input = input,  output = output, T=T)
            self.assertTrue("The sum of each row in the transmission matrix should equal to 1"
                            in context.exception )

    def testTransProbabilityValidAssignment(self):
        input = ["elicit", "tell"]
        output = ["wrong", "correct", "told"]
        T = numpy.array([0.1,0.9,0.2,0.8,0.3,0.7,0.4,0.6]).reshape(2,2,2)
        iohmm = IOHMM(2, input = input,  output = output, T=T)
        self.assertTrue(numpy.allclose(iohmm.T, T))

    def testEmissProbabilityInvalid(self):
        input = ["elicit", "tell"]
        output = ["wrong", "correct", "told"]
        E = numpy.array([0.1,0.2,0.2,0.3]).reshape(2,2)
        with self.assertRaises(ValueError) as context:
            IOHMM(2, input = input,  output = output, E=E)
            self.assertTrue("The emission probaility matrix dimension mismatches the given states number and "
                            "input/output symbols number" in context.exception)

        E = numpy.array([0.1,0.3,0.7,0.4,0.5,0.4,0.3,0.5,0.5,0.6,0.5,0.3]).reshape(2,2,3)
        with self.assertRaises(ValueError) as context:
            IOHMM(2, input = input,  output = output, E=E)
            self.assertTrue("The sum of each row in the transmission matrix should equal to 1" in context.exception)

    def testEmissionProbabilityValid(self):
        input = ["elicit", "tell"]
        output = ["wrong", "correct", "told"]
        E = numpy.array([0.1,0.2,0.7,0.4,0.5,0.1,0.3,0.5,0.2,0.6,0.1,0.3]).reshape(2,2,3)
        iohmm = IOHMM(2, input = input, output=output, E=E)
        self.assertTrue(numpy.array_equal(E,iohmm.E))

    def testFixedInvalidProb(self):
        input = ["elicit", "tell"]
        output = ["wrong", "correct", "told"]
        F = {(0,1):numpy.array([0,1,1]),(1,1):numpy.array([2,1])}
        with self.assertRaises(ValueError) as context:
            IOHMM(2, input = input, output=output, F=F)
        self.assertTrue("The probability transferring from this state should sum up to 1."
                            in context.exception)

    def testFixedValidProb(self):
        input = ["elicit", "tell"]
        output = ["wrong", "correct", "told"]
        F = {(0,1):numpy.array([0,1]),(1,1):numpy.array([0,1])}
        iohmm = IOHMM(2, input = input, output=output, F=F)

    def testInitInvalidProb(self):
        input = ["elicit", "tell"]
        output = ["wrong", "correct", "told"]
        Pi = [1,2,3]
        with self.assertRaises(ValueError) as context:
            IOHMM(2, input = input, output=output,Pi=Pi )
            self.assertTrue("The initial state probability dimension mismatches the given states number."
                            in context.exception)

        Pi = [0.1,0.5]
        with self.assertRaises(ValueError) as context:
            IOHMM(2, input = input, output=output,Pi=Pi )
            self.assertTrue("The initial state probability dimension mismatches the given states number."
                            in context.exception)

    def testInitValidProb(self):
        input = ["elicit", "tell"]
        output = ["wrong", "correct", "told"]
        Pi = [0.1,0.9]
        iohmm = IOHMM(2, input = input, output=output,Pi=Pi )
        self.assertEqual(iohmm.Pi, Pi)

    def testToIndex(self):
        input = ["elicit", "tell"]
        output = ["wrong", "correct", "told"]
        iohmm = IOHMM(2, input = input, output=output )
        seq = ["wrong","correct","wrong","correct","correct"]
        self.assertTrue(numpy.array_equal(iohmm.toIndex(seq,iohmm.output_map),
                                          numpy.array([0,1,0,1,1])))

        seq2 = ["wrong","correct","wrong","correct","correct","undefined"]
        with self.assertRaises(ValueError):
            iohmm.toIndex(seq2,iohmm.output_map)

    def testForward(self):
        input = ["elicit", "tell"]
        output = ["wrong", "correct", "told"]
        input_seq = ["elicit", "tell", "elicit", "elicit", "tell"]
        output_seq = ["wrong", "told","correct","wrong","told"]
        T = numpy.array([0.2,0.8,0,1,0.1,0.9,0,1]).reshape(2,2,2)
        E = numpy.array([0.5,0.5,0,0.1,0.9,0,0,0,1,0,0,1]).reshape(2,2,3)
        Pi = numpy.array([0.5,0.5])
        iohmm = IOHMM(2, input=input, output=output, T=T, E=E, Pi=Pi)
        result = iohmm.forward(input_seq, output_seq)
        self.assertEqual(result[0], -1.2039728043259359)
        result1 = iohmm.forward(input_seq, output_seq, scaling=True)
        self.assertEqual(result1[0], -1.2039728043259359 )

    def testBackward(self):
        input = ["elicit", "tell"]
        output = ["wrong", "correct", "told"]
        input_seq = ["elicit", "tell", "elicit", "elicit", "tell"]
        output_seq = ["wrong", "told","correct","wrong","told"]
        T = numpy.array([0.2,0.8,0,1,0.1,0.9,0,1]).reshape(2,2,2)
        E = numpy.array([0.5,0.5,0,0.1,0.9,0,0,0,1,0,0,1]).reshape(2,2,3)
        Pi = numpy.array([0.5,0.5])
        iohmm = IOHMM(2, input=input, output=output, T=T, E=E, Pi=Pi)
        result = iohmm.backward(input_seq, output_seq)
        Beta = numpy.array([0.09, 0.09, 0.18, 1, 1, 0.09, 0.09, 0.1, 1, 1]).reshape(2,5)
        self.assertTrue(numpy.allclose(result, Beta))



















if __name__ == '__main__':
  unittest.main()

