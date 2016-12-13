import random
transitions = []
transition_prob = []
typos = []
typo_prob = []

# initialization
def get_index(ch):
    # in the future, we could change mapping
    if ch >= 'A' and ch <= 'Z':
        return ord(ch) - ord('A')
    elif ch >= 'a' and ch <= 'z':
        return ord(ch) - ord('a')
    return -1


def initialize_typo_prob():
    # to be implemented
    global typos
    typos = []
    for c1 in range(26):
        typos.append([])
        for c2 in range(26):
            typos[c1].append(0)

cnt=0
def update_typo_prob(sentence1, sentence2):  # correct = sentence1
    n = min(len(sentence1), len(sentence2))
    global cnt
    if cnt<10:
        print "sentence1=",sentence1
        print "sentence2=",sentence2
        cnt += 1
    for i in range(n):
        c1 = get_index(sentence1[i])
        c2 = get_index(sentence2[i])
        if c1 == -1 or c2 == -1:
            continue
        typos[c1][c2] += 1

def finalize_typo_prob(alpha = 0):
    global typo_prob
    typo_prob = []
    for c1 in range(26):
        typo_prob.append([])
        s = sum(typos[c1]) + alpha * 26.0
        for c2 in range(26):
            typo_prob[c1].append((typos[c1][c2] + alpha) / s)

def initialize_transition_prob(): 
    global transitions
    transitions = []
    for c1 in range(26):
        transitions.append([])
        for c2 in range(26):
            transitions[c1].append(0)


# call the following function to update for each sentence
# currently, not ignoring the invlaid code
cnt2=0
def update_transition_prob(sentence):
    global cnt2
    if cnt2 < 10:
        print "transition=",sentence
        cnt2 +=1
    for i in range(len(sentence) - 1):
        c1 = get_index(sentence[i])
        c2 = get_index(sentence[i + 1])
        if c1 != -1 and c2 != -1:
            transitions[c1][c2] += 1

def finalize_transition_prob(alpha = 0):
    # transition_prob[a][b] (a to b)
    global transition_prob
    transition_prob = []
    for c1 in range(26):
        transition_prob.append([])
        s = sum(transitions[c1]) + 26 * alpha
        for c2 in range(26):
            transition_prob[c1].append((float)(transitions[c1][c2] + alpha) / s)

keyboard = []
position = [0] * 26
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

# get train data
rate = 1
prep()
train_file = "../../data/clinton_newlines.txt"
train_typo_file = "../../data/clinton_data_typo.txt"
fi = open(train_file, 'r')
fo = open(train_typo_file, 'w')
fo.truncate()
for line in fi:
    fo.write(inject_typo(line, rate))
fo.close()

# train model
initialize_typo_prob()
initialize_transition_prob()
fi = open(train_file)
fo = open(train_typo_file)

correct = []
for line in fi:
    correct.append(line)
    update_transition_prob(line)
line_num = 0
for line in fo:
    update_typo_prob(correct[line_num], line)
    line_num += 1

fi.close()
fo.close()

finalize_transition_prob(1)
finalize_typo_prob(1)
print transitions
for i in range(26):
    for j in range(26):
        print "tran[%d][%d] = %.02f" % (i, j, transition_prob[i][j])


# calculate maximum likelihood sequence ??? (whatever name)
# dp[i][c] (at ith character, char = c)
# ignore not valid chars
def viterbi(sentence): 
    n = len(sentence)
    dp = []
    ans = []
    
    pos = [-1] * n
    last_char = -1
    r = 0
    for i in range(n):
        if get_index(sentence[i]) != -1:
            pos[i] = r
            c2 = get_index(sentence[i])
            dp.append([0] * 26)
            ans.append([0] * 26)
            if last_char == -1:
                # first
                for c1 in range(26):
                    dp[r][c1] = typo_prob[c1][c2]
                    ans[r][c1] = -1
            else:
                # not first
                for c1 in range(26):
                    dp[r][c1] = 0
                    for c3 in range(26):
                        tmp = dp[r - 1][c3] * transition_prob[c3][c1] * typo_prob[c1][c2]
                        if tmp > dp[r][c1]:
                            dp[r][c1] = tmp
                            ans[r][c1] = c3
               
            last_char = c2
            r += 1
    
    # now recalculate to get the sequence
    if r == 0:
        return ""
    i, c = r - 1, dp[r - 1].index(max(dp[r - 1]))
    output = ""
    while (c != -1):
        output += chr(c + ord('a'))
        c = ans[i][c]
        i -= 1
    output = output[::-1]
    correct_output = ""
    for i in range(len(sentence)):
        if pos[i] == -1:
            correct_output += sentence[i]
        else:
            correct_output += output[pos[i]]
    return correct_output

def fix_typo(sentence):
    return viterbi(sentence)

typo_file = "../../data/clinton_data_typo.txt"
correct_file = "../../data/clinton_newlines.txt" 
output_file = "../../data/clinton_data_typo_correct.txt"
print "hi"
fo = open(output_file, "w")
results = []
original = []

line_num = 0
with open(typo_file) as f:
    for line in f:
        results.append(fix_typo(line))
        original.append(line)
        fo.write(results[-1])
        line_num += 1
        if line_num >= 500:
            break
line_num = 0
fix_num = 0
wrong_num = 0
total_num = 0
with open(correct_file) as f:
    for line in f:
        if (line_num >= 500):
            break
        ori = original[line_num].lower().split(" ")
        res = results[line_num].lower().split(" ")
        cor = line.lower().split(" ")
        total_num += len(cor)
        for i in range(len(cor)):
            if ori[i] != cor[i]:
                wrong_num += 1
            if res[i] != cor[i]:
                fix_num += 1
        line_num += 1
        if (line_num % 10 == 0):
            print "line=", line_num
        if (line_num >= 500):
            break

fo.close()

print fix_num / float(total_num)
print wrong_num / float(total_num)