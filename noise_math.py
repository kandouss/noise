import numpy as np

class StaticNoise:
	def __init__(self,n=256,D=2):
		self.n = n
		self.D = D
		self.G = GradientTable(n=n,D=D)
		self.weight = lambda d: np.prod([1.0-(3*k**2-2*k**3) for k in np.abs(d)])
		self.unit_hypercube = self.gen_unit_hypercube()
	
	def __getitem__(self,p):
		return self.noise_at(p)
	
	def noise_at_frac(self,p):
		p = np.array(p)
		p *= self.n
		return self.noise_at(p)

	def noise_at(self,p):
		p = np.array(p)
		v = 0
		for c in self.unit_hypercube:
			q = np.floor(p) + c
			g = self.G[tuple(np.mod(q,self.n))]
			v += np.dot( g, q-p ) * self.weight( q-p )
		return v

	def gen_unit_hypercube(self):
		pts = []
		fmt_str = '{:0'+str(self.D)+'b}'
		for d in range(2**self.D):
			pts.append(np.array( [int(k) for k in list(fmt_str.format(d))]))
		return pts

class GradientTable:	
	def __init__(self,n=256,D=2):
		self.n = n
		self.D = D
		self.G = self.gen_grad_array(n,D)
		self.P = self.gen_perm_array(n)
	
	def __getitem__(self,p):
		if isinstance(p,tuple):
			p = list(p)
		if isinstance(p,list):
			if max(p) >= self.n or len(p) > self.D:
				raise IndexError
			if len(p) == 1:
				return np.array(self.G[p[0]])
			else:
				p[1] = ( self.P[p[-1]] + p[-2] )%self.n
				p = p[:-1]
				return self.__getitem__(p)
		elif isinstance(p,int):
			return np.array(self.G[p])
		else:
			raise IndexError

	@staticmethod	
	def gen_grad_array(n, D):
		A = np.random.rand(n,D)*2.0-1.0
		for k in range(n):
			while np.sum(A[k]**2.0) > 1.0:
				A[k] = np.random.rand(D)
		A *= np.outer(np.sqrt(np.sum(A**2.0,axis=1))**-1,np.array([1,1]))
		return A

	@staticmethod
	def gen_perm_array(n):
		A = np.arange(n)
		for i in range(n):
			j = np.random.randint(i,n)
			tmp = A[i]
			A[i] = A[j]
			A[j] = tmp
		return A
