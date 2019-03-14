import string

s = ['the', 'is', 'to', 'beautiful']

for data in s:
    if data in string.punctuation:
        print(data)
