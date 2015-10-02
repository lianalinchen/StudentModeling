from unittest import TestCase
from HMM import HMM
import numpy


__author__ = 'adminuser'


class TestHMM(TestCase):

    def testInputOutputSymbol(self):
        symbols = ["wrong", "correct"]
        hmm = HMM(2, output = symbols)
        self.assertEqual(hmm.N,2,"The states number is wrong")

        self.assertEqual(hmm.V, symbols)
        self.assertEqual(hmm.M, 2, "The number of output symbol is incorrect.")
        self.assertEqual(hmm.symbol_map, {"wrong":0, "correct":1}, "The symbol map is incorrect")

    def testDefaultValue(self):
        symbols = ["wrong", "correct"]
        hmm = HMM(2, output = symbols)

        self.assertTrue(numpy.allclose(hmm.T.sum(1), numpy.array([1.0, 1.0])))
        self.assertTrue(numpy.allclose(hmm.E.sum(1), numpy.array([1.0, 1.0])))
        self.assertEqual(hmm.F, {})

    def testTInvalid(self):
        symbols = ["wrong", "correct"]
        T = numpy.array([0.1,0.2])
        with self.assertRaises(ValueError) as context:
            hmm = HMM(2, output = symbols, T=T)
        self.assertTrue("The transmission probability matrix dimension mismatches "
                        "the given states number." in context.exception)

        T = numpy.array([0.3, 0.9, 0, 0.9]).reshape(2,2)
        with self.assertRaises(ValueError) as context:
            hmm = HMM(2, output = symbols, T=T)
        self.assertTrue("The sum of each row in the transimission matrix should equal to 1"
                        in context.exception)

    def testTValid(self):
        symbols = ["wrong", "correct"]
        T = numpy.array([0.1, 0.9, 0.1, 0.9]).reshape(2,2)
        hmm = HMM(2, output = symbols, T=T)
        self.assertTrue(numpy.array_equal(T, hmm.T))

    def testEInvalid(self):
        symbols = ["wrong", "correct"]
        E = numpy.array([0.1,0.2])
        with self.assertRaises(ValueError) as context:
            hmm = HMM(2, output = symbols, E = E)
        self.assertTrue("The emission probaility matrix dimension mismatches the given states number and "
                        "output symbols number" in context.exception)

        E = numpy.array([0.3, 0.9, 0, 0.9]).reshape(2,2)
        with self.assertRaises(ValueError) as context:
            hmm = HMM(2, output = symbols, E = E)
        self.assertTrue("The sum of each row in the emission probability matrix should equal to 1"
                        in context.exception)

    def testEValid(self):
        symbols = ["wrong", "correct"]
        E = numpy.array([0.1, 0.9, 0.1, 0.9]).reshape(2,2)
        hmm = HMM(2, output = symbols, E=E)
        self.assertTrue(numpy.array_equal(E, hmm.E))

    def testFixedValid(self):
        Fixed = {1: numpy.array([0.1, 0.9])}
        symbols = ["wrong", "correct"]
        hmm = HMM(2, output = symbols, F = Fixed)
        self.assertTrue(numpy.array_equal(hmm.T[1],Fixed[1]))

    def testFixedInvalid(self):
        Fixed = {1: numpy.array([0.1, 0])}
        symbols = ["wrong", "correct"]
        with self.assertRaises(ValueError) as context:
            hmm = HMM(2, output = symbols, F = Fixed)
        self.assertTrue("The probability transferring from this state should sum up to 1." in context.exception)

    def testPiInvalid(self):
        Pi = numpy.array([0.2, 0.2])
        symbols = ["wrong", "correct"]
        with self.assertRaises(ValueError) as context:
            hmm = HMM(2, output = symbols, Pi = Pi)
        self.assertTrue("The initial state probability does not add up to 1." in context.exception)

        Pi = numpy.array([0.2, 0.2,0.6])
        with self.assertRaises(ValueError) as context:
            hmm = HMM(2, output = symbols, Pi = Pi)
        self.assertTrue("The initial state probability dimension mismatches the given states number."
                        in context.exception)

    def testPiValid(self):
        symbols = ["wrong", "correct"]
        Pi = numpy.array([0.5, 0.5])
        hmm = HMM(2, output = symbols, Pi= Pi)
        self.assertTrue(numpy.array_equal(hmm.Pi, Pi))

    def testForward(self):
        symbols = ["wrong", "correct"]
        T = numpy.array([0.4,0.6,0,1]).reshape(2,2)
        E = numpy.array([0.4,0.6,0.1,0.9]).reshape(2,2)
        Pi = numpy.array([0.5,0.5])
        hmm = HMM(2, output = symbols, T=T, E=E, Pi=Pi)
        output_seq = ["correct","correct","wrong","correct","correct"]
        result = hmm.forward(output_seq)
        self.assertEqual(result[0], -2.861525489057914)
        result2 = hmm.forward(output_seq, scaling=True)
        self.assertEqual(result2[0], -2.861525489057914)

    def testBackward(self):
        symbols = ["wrong", "correct"]
        T = numpy.array([0.4,0.6,0,1]).reshape(2,2)
        E = numpy.array([0.4,0.6,0.1,0.9]).reshape(2,2)
        Pi = numpy.array([0.5,0.5])
        hmm = HMM(2, output = symbols, T=T, E=E, Pi=Pi)
        output_seq = ["correct","correct","wrong","correct","correct"]
        result = hmm.backward(output_seq)
        r = numpy.array([0.08125488, 0.156312, 0.6732, 0.78, 1. ,0.0729, 0.081, 0.81, 0.9, 1.  ]).reshape(2,5)
        self.assertTrue(numpy.allclose(result,r))