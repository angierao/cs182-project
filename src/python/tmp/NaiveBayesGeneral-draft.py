""" Feature-based Naive Bayes Implementation. """

# Labels: 0 - Donald Trump; 1 - Hillary
m = 3  # number of features
def getDataFromRaw():

def getFeaturesFromData():

def getFeatures():
""" Get Feature Values. """

def getMean(arr):
    return sum(arr) / float(len(arr))

def getStdev(arr):
    avg = getMean(arr)
    variance = sum([pow(x-avg, 2) for x in numbers])/float(len(numbers)-1)
    return math.sqrt(variance)

def processData():
""" Calculate mean and standard deviation. """
    means = [[], []]
    stdevs = [[], []]
    for label in range(2):
        for f in range(m):
            arr = [entry[1][f] for entry in data if entry[0] == label]
            means[label].append(getMean(arr))
            stdevs[label].append(getStdev(arr))
            

def mylog(x):
    if x == 0:
        return sys.float_info.min
    else:
        return log(x)

def calculateProbability(x, mean, stdev):
""" Use normal distribution to calculate corresponding probability. """
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

def predictData(inputFeatures):
""" Predict data given a list of features. """
    ans = [0] * 2
    for label in range(2): 
       ans[label] += mylog(total[label] / (1.0 * total[0] + total[1]))
       for i in range(m):
           ans[label] += mylog(calculateProbability(inputFeatures[i], means[label][i], stdevs[label][i]))
    if ans[0] > ans[1]:
        return (0, "Donald Trump")
    else:
        return (1, "Hillary Clinton")


    

