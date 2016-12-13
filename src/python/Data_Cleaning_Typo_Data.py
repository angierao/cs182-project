import random
import sys
f1 = open("../../data/clinton_newlines.txt")
f2 = open("../../data/train_correct.txt", "w")
f3 = open("../../data/train_typo.txt", "w")
f2.truncate()
f3.truncate()
keyboard = []
position = [0] * 26
# initialization
def get_index(ch):
    # in the future, we could change mapping
    if ch >= 'A' and ch <= 'Z':
        return ord(ch) - ord('A')
    elif ch >= 'a' and ch <= 'z':
        return ord(ch) - ord('a')
    return -1

def prep():
    global keyboard
    global position
    keyboard = [['q','w','e','r','t','y', 'u', 'i', 'o', 'p'], ['a','s','d','f','g','h','j','k','l'],['z','x','c','v','b','n','m']]
    for i in range(26):
        for x in range(len(keyboard)):
            for y in range(len(keyboard[x])):
                if get_index(keyboard[x][y]) == i:
                    position[i] = (x, y)
    print "position=", position
    print "keyboard=", keyboard

def check_adjacent(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    if abs(x1 - x2) <= 1 and abs(y1 - y2) <= 1:
        return True
    return False

def inject_typo(line, rate):
    new_line = ""
    for i in range(len(line)):
        c1 = get_index(line[i])
        if c1 == -1:
            new_line += line[i]
            continue
        if random.randint(1, 100) <= rate:
            # inject typo
            adjs = []
            for c2 in range(26):
                if check_adjacent(position[c1], position[c2]):
                    adjs.append(c2)
            new_c = adjs[random.randint(0, len(adjs) - 1)]
            new_line += chr(new_c + ord('a'))
        else:
            new_line += line[i]
    return new_line

rate = 5
prep()
for line in f1:
	words = line.lower().split(' ')
	for word in words:
		word = word.replace(".", "")
		word = word.replace(",", "")
		word = word.replace("?", "")
		word = word.replace("\n", "")

		f2.write(word + "\n")
		f3.write(inject_typo(word, rate) + "\n")
f1.close()
f2.close()
f3.close()

"""
f1 = open("../../data/typos10.data")
f2 = open("../../data/train_correct.txt", "w")
f3 = open("../../data/train_typo.txt", "w")
f2.truncate()
f3.truncate()

s2 = ""
s3 = ""
for line in f1:
	if line[0] == '_':
		f2.write(s2 + "\n")
		f3.write(s3 + "\n")
		s2 = ""
		s3 = ""
	else:
		s2 += line[0]
		s3 += line[2]
f1.close()
f2.close()
f3.close()
"""