import itertools

el = itertools.product('ABCDEF', repeat=3)
print(len(list(el)))
for e in el:
    print(e)
