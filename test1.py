import deal_data

a = [1, 2, 3, 1, 3, 1]

id1 = [i for i, data in enumerate(a) if data == max(a)]

print(id1)

b = set()

b.add(1)
b.add(1)

print(list(b)[0])
print(len(b))
