
import unittest as ut
import numpy as np
from theano import tensor as T
from theano import function # shared, pp


class CustomLossTest(ut.TestCase):

	def setUp(self):
		self.x = T.dvector()
		self.ret = custom_loss(None, self.x)
		self.lossFun = function([self.x], self.ret)

	def test(self):
		probs = np.array([0.5, 0.3, 0.2], dtype='float64')
		self.assertTrue( self.lossFun(probs) == 0.0 )

	def test1(self):
		probs = np.array([1., 0, 0])
		self.assertTrue( self.lossFun(probs) == 0.0 )

	def test_equal_probabilities(self):
		probs = np.array([0.25, 0.25, 0.25, 0.25])
		print('Return symbol, class, type:',self.ret,type(self.ret),self.ret.type)
		# print(pp(self.ret))
		self.assertTrue( self.lossFun(probs) == 1.00 )


def custom_loss(y_true, probs):
	onesLike = T.ones_like(probs)
	print('ones be like!:',onesLike)
	tupSize = T.sum(onesLike) + 1
	print('Tuple length:',tupSize)
	target = probs[0] * onesLike
	print('Repeated p1:',target)

	boolTensor = T.gt(target, probs)
	print('Conditions:',boolTensor)
	comparisonResult = T.switch(boolTensor, T.zeros_like(probs), onesLike)
	print('Compared:',comparisonResult)
	count = T.sum(comparisonResult) - 1 # for the self-comparison of first probability!

	frac_loss = count*1.0/(tupSize-2)
	print('Frac. loss:',frac_loss)
	return frac_loss



