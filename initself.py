
from __future__ import print_function


# before/after
BEFAF = False


class Narek(object):

	def __init__(self, bar=0, baz=False):
		self.bar = bar
		if not BEFAF:
			self.__dict__.update(locals())

	def doStuff(self, x):
		print(x)



k = Narek(bar=0.9, baz=True)
k.doStuff(17)

print(k.bar)

if k.baz:
	print('NAG used!')
else:
	print('NAG NOT used!!')