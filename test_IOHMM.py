import unittest
from unittest import TestCase
import numpy
from IOHMM import IOHMM

__author__ = 'adminuser'


class TestIOHMM(TestCase):

    def testInputOutputSymbol(self):
        input = ["elicit", "tell"]
        output = ["wrong", "correct"]
        iohmm = IOHMM(2, input = input,  output = output)
        self.assertEqual(iohmm.N,2,"The states number is wrong")

        self.assertEqual(iohmm.U, input, "The input symbols are incorrect")
        self.assertEqual(iohmm.K, 2, "The number of input symbols is incorrect")
        self.assertEqual(iohmm.input_map, {"elicit":0, "tell":1}, "The input map is incorrect")

        self.assertEqual(iohmm.V, output, "The output symbols are incorrect")
        self.assertEqual(iohmm.M, 2, "The number of output symbols is incorrect")
        self.assertEqual(iohmm.output_map, {"wrong":0, "correct":1}, "The output map is incorrect")

    def testTransProbabilityDefaultValue(self):
        input = ["elicit", "tell"]
        output = ["wrong", "correct"]
        iohmm = IOHMM(2, input = input,  output = output)
        self.assertTrue(numpy.allclose(iohmm.T[0].sum(1), numpy.array([1.0,1.0])))
        self.assertTrue(numpy.allclose(iohmm.T[1].sum(1), numpy.array([1.0,1.0])))

    def testTransProbabilityInvalid(self):
        input = ["elicit", "tell"]
        output = ["wrong", "correct"]
        T = numpy.array([0.1,0.4,0.2,0.8,0.3,0.7,0.4,0.6]).reshape(2,2,2)
        self.assertRaises(ValueError,  IOHMM(2, input = input,  output = output, T=T) )


    def testTransProbabilityValidAssignment(self):
        input = ["elicit", "tell"]
        output = ["wrong", "correct"]
        T = numpy.array([0.1,0.9,0.2,0.8,0.3,0.7,0.4,0.6]).reshape(2,2,2)
        iohmm = IOHMM(2, input = input,  output = output, T=T)
        self.assertTrue(numpy.allclose(iohmm.T, T))







if __name__ == '__main__':
  unittest.main()

