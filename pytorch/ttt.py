tt = [0, 123]
ff = 456

def temp(u):
	return 100 + u


b = '{0}, {2}, ---- {2}, {1}'.format('aaa', 'bbb', 'ccc')

print(b)

a = f'''	fdfd
	affadsf /
	afdf \
	{tt[1]}
	{ff}
	{min(10, 11)}

	{"__global const Dtype* Dptr = bias {};".format(ff) if True else ' '}


	fafa
	fsfsaf'''
print(a)