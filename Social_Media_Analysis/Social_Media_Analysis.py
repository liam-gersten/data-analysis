# some packages require installations
import matplotlib.pyplot as plt; plt.rcdefaults()
import pandas as pd
import numpy as np
import nltk
nltk.download('vader_lexicon', quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# parses through text and returns specific segment
def parse(fromString, start, stop, startBefore, stopAfter):
    [name, iterate] = ['', False]
    for i in range(len(fromString)):
        if ((fromString[i:i+stopAfter] == stop) and iterate): return name
        if iterate: name += fromString[i]
        if (i >= startBefore) and (fromString[i-startBefore:i+1] == start):
            iterate = True

# parses through tweet/post and returns a list with all hashtags
def findHashtags(message):
    endChars = [' ', '\n', '#', '.', ',', '?', '!', ':', ';', ')']
    [hashtags, word, iterate, i] = [[], '', False, 0]
    while i < (len(message)):
        if iterate and ((message[i] in endChars) or (i == (len(message))-1)):
            if (i == (len(message))-1) and (message[i] not in endChars):
                word += message[i]
            hashtags += [word]
            [word, iterate] = [word, iterate]
        elif (not iterate) and (message[i] == '#'):
            iterate = True
            word += message[i]
        elif iterate: word += message[i]
        i += 1
    return hashtags

# parses through 'embed' segment of the df to find the date if it exists
def findDate(fromString):
    for i in range(len(fromString)-1, 0, -1):
        if (fromString[i-2] == '-') and (fromString[i-5] == '-'):
            date = fromString[i-9:i+1]
            numeric = (int(date[0:4]))+(int(date[5:7])/12)
            return [date, numeric]
    return [0, 0]

# adds 5 new columns to the dataframe
def addColumns(data, stateDf):
    [names, positions, states, regions, hashtags, dates, numericDates] = \
        [[], [], [], [], [], [], []]  # empty list variable definition
    for index, row in data.iterrows():
        names.append(parse(row['label'], 'From: ', ' (', 5, 2))
        positions.append(parse(row['label'], '(', ' from', 0, 5))
        state = parse(row['label'], 'from ', ')', 4, 1)
        states.append(state)
        rRow = stateDf.loc[stateDf['state'] == state, 'region']
        regions.append(rRow.values[0])
        hashtags.append(findHashtags(row['text']))
        date = findDate(row['embed'])
        dates.append(date[0])
        numericDates.append(date[1])
    data['name'] = names
    data['position'] = positions
    data['state'] = states
    data['region'] = regions
    data['hashtags'] = hashtags
    data['date'] = dates
    data['numeric_date'] = numericDates
    return None

# returns both sentiment category and exact sentiment value
def findSentiment(classifier, message):
    score = classifier.polarity_scores(message)['compound']
    if score < -0.1: return ['negative', score]
    elif score > 0.1: return ['positive', score]
    return ['neutral', score]

# adds both sentiment category and exact score columns to the dataframe
def addSentimentColumn(data):
    classifier = SentimentIntensityAnalyzer()
    [sentiments, exactSentiments] = [[], []]
    for index, row in data.iterrows():
        group = findSentiment(classifier, row['text'])
        sentiments.append(group[0])
        exactSentiments.append(group[1])
    data['sentiment'] = sentiments
    data['exact_sentiment'] = exactSentiments
    return None

# creates dict for mapping states and designated values
def getDataCountByState(data, colName, dataToCount):
    d = {}
    if colName == '' and dataToCount == '':
        for index, row in data.iterrows():
            if (row['state']) in d:
                number = d[(row['state'])]+0
                d[row['state']] = number+1
            else: d[row['state']] = 1
    else:
        for index, row in data.iterrows():
            if row[colName] == dataToCount:
                if (row['state']) in d:
                    number = d[(row['state'])]+0
                    d[row['state']] = number+1
                else: d[row['state']] = 1
    return d

# creates dict for mapping regions and designated values
def getDataForRegion(data, colName):
    d = {}
    for index, row in data.iterrows():
        if row['region'] not in d: d[row['region']] = {}
        if row[colName] not in d[row['region']]:
            d[row['region']][row[colName]] = 1
        else: d[row['region']][row[colName]] += 1
    return d

# creates dict to map hashtags and their total occurrences in the dataframe
def getHashtagRates(data):
    [d, hashtags] = [{'#h42ht4g2': 0}, []]  # '#h42ht4g2' is unique null value
    for index, row in data.iterrows(): hashtags += row['hashtags']
    for hashtag in hashtags:
        if hashtag in d: d[hashtag] += 1
        else: d[hashtag] = 1
    return d

def mostCommonHashtags(hashtags, count):
    d = {}
    while len(d) < count:
        [maxVal, maxKey] = [0, '']
        for hashtag in hashtags:
            if (hashtag not in d) and (hashtags[hashtag] > maxVal):
                [maxVal, maxKey] = [hashtags[hashtag], hashtag]
        d[maxKey] = maxVal
    return d

def getHashtagSentiment(data, hashtag):
    sentiments = []
    sentimentKey = {'positive': 1, 'negative': -1, 'neutral': 0}
    for index, row in data.iterrows():
        contained = False
        for item in findHashtags(row['text']):
            if item == hashtag: contained = True
        if contained: sentiments.append(float(sentimentKey[row['sentiment']]))
    return float(sum(sentiments)/len(sentiments))

def graphStateCounts(stateCounts, title):
    [labels, values] = [[], []]
    for pair in stateCounts:
        labels.append(pair)
        values.append(stateCounts[pair])
    fig, axis = plt.subplots()
    ind = range(len(values))
    rects1 = axis.bar(ind, values)
    axis.set_ylabel('count')
    axis.set_title(title)
    axis.set_xticks(ind)
    axis.set_xticklabels(labels)
    plt.xticks(rotation='vertical')
    plt.show()

def graphTopNStates(stateCounts, stateFeatureCounts, n, title):
    [stateRates, fullRates] = [{}, {}]
    for state in stateFeatureCounts:
        fullRates[state] = stateFeatureCounts[state]
    for state in stateCounts:
        if state in fullRates:
            tmp = fullRates[state]
            fullRates[state] = tmp/stateCounts[state]
    while len(stateRates) < n:
        [maxRate, maxState] = [0, 'NULL']
        for state in fullRates:
            if (state not in stateRates) and (fullRates[state] > maxRate):
                [maxRate, maxState] = [fullRates[state], state]
        stateRates[maxState] = maxRate
    graphStateCounts(stateRates, title)

def graphRegionComparison(regionDicts, title):
    [features, regions, listPairs] = [[], [], []]
    for region in regionDicts:
        regions.append(region)
        for feature in regionDicts[region]:
            if feature not in features: features.append(feature)
    for region in regionDicts:
        tmp = []
        for feature in features:
            if feature in regionDicts[region]:
                tmp.append(regionDicts[region][feature])
            else: tmp.append(0)
        listPairs.append(tmp)
    sideBySideBarPlots(features, regions, listPairs, title)

def graphHashtagSentimentByFrequency(df):
    [hashtags, frequencies, sentimentScores] = [[], [], []]
    hashtagRates = getHashtagRates(df)
    commonHashtags = mostCommonHashtags(hashtagRates, 50)
    for hashtag in commonHashtags:
        hashtags.append(hashtag)
        frequencies.append(commonHashtags[hashtag])
        sentimentScores.append(getHashtagSentiment(df, hashtag))
    title = '50 common hashtags by frequency and sentiment'
    scatterPlot(frequencies, sentimentScores, hashtags, title, frequencies,
                'frequency', 'sentiment score')

# non-destructively filters out designated 'filter' value from 2 lists
def filterValues(list1, list2, filter):
    [new1, new2] = [[], []]
    for i in range(len(list1)):
        if (list1[i] != filter) and (list2[i] != filter):
            new1 += [list1[i]]
            new2 += [list2[i]]
    return [new1, new2]

def scatterPlot(xValues, yValues, labels, title, xTicks, xLabel, yLabel):
    fig, ax = plt.subplots()
    plt.scatter(xValues, yValues)
    for i in range(len(labels)):
        displayLabels = ['obamacare', 'irs', 'benghazi', 'nsa', 'sotu',
                         'keystonexl', 'junteenth', 'gopbudget', 'isis', 'jobs']
        if labels[i][1:].lower() in displayLabels:
            plt.annotate(labels[i], (xValues[i], yValues[i]),
                     textcoords="offset points", xytext=(0, 10), ha='center')
    ax.set_ylabel(yLabel)
    ax.set_xlabel(xLabel)
    ax.plot(color='black')
    plt.xticks(xTicks)
    plt.title(title)
    plt.show()

def sideBySideBarPlots(xLabels, labelList, valueLists, title):
    x = np.arange(len(xLabels))
    width = 0.8/len(labelList)
    fig, ax = plt.subplots()
    colors = ['red', 'blue', 'orange', 'green']
    for i in range(len(valueLists)):
        ax.bar(x-0.4+width*(i+0.5), valueLists[i], width, label=labelList[i],
               color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels(xLabels)
    plt.xticks(rotation="vertical")
    ax.legend()
    plt.title(title)
    fig.tight_layout()
    plt.show()

def createDf():
    df = pd.read_csv("political_data.csv")
    stateDf = pd.read_csv("state_mappings.csv")
    addColumns(df, stateDf)
    addSentimentColumn(df)
    print(df.head())
    print(df.describe())
    print(df.info())
    return df

def printOuts(df):
    stateCounts = getDataCountByState(df, "", "")
    print("\nTotal Messages Per State:")
    print(stateCounts)
    negSentiments = getDataCountByState(df, "sentiment", "negative")
    print("\nState Counts for Negative Sentiment:")
    print(negSentiments)
    attacks = getDataCountByState(df, "message", "attack")
    print("\nState Counts for Attacks:")
    print(attacks)
    partisanship = getDataCountByState(df, "bias", "partisan")
    print("\nState Counts for Partisanship:")
    print(partisanship)
    messages = getDataForRegion(df, "message")
    print("\nMessage Types for Region:")
    print(messages)
    audiences = getDataForRegion(df, "audience")
    print("\nAudience Types for Region:")
    print(audiences)
    hashtags = getHashtagRates(df)
    freqHashtags = mostCommonHashtags(hashtags, 6)
    for hashtag in freqHashtags:
        print(hashtag, "sentiment score:", getHashtagSentiment(df, hashtag))

def displayFigures(df):
    stateCounts = getDataCountByState(df, "", "")
    print("\nBasic bar charts:")
    twitterCounts = getDataCountByState(df, "source", "facebook")
    graphStateCounts(stateCounts, "Total Messages Per State")
    graphStateCounts(twitterCounts, "Total *Facebook* Messages Per State")
    print("\nFiltered bar charts:")
    attackCounts = getDataCountByState(df, "message", "attack")
    policyCounts = getDataCountByState(df, "message", "policy")
    nationalCounts = getDataCountByState(df, "audience", "national")
    graphTopNStates(stateCounts, attackCounts, 5, "Top Attack Message Rates")
    graphTopNStates(stateCounts, policyCounts, 5, "Top Policy Message Rates")
    graphTopNStates(stateCounts, nationalCounts, 5, "Top National Messages")
    print("\nSide-by-side bar charts:")
    messageTypes = getDataForRegion(df, "message")
    positionTypes = getDataForRegion(df, "position")
    graphRegionComparison(messageTypes, "Messages by Region")
    graphRegionComparison(positionTypes, "Position by Region")
    print('\nscatterplots:')
    [list1, list2] = filterValues(df['numeric_date'], df['exact_sentiment'], 0)
    scatterPlot(list1, list2, [], 'Sentiment Through Time', [2013, 2014, 2015],
                '', 'sentiment score')
    graphHashtagSentimentByFrequency(df)

if __name__ == "__main__":
    print('\ncreating dataframe')
    df = createDf()
    print('\nnow printing')
    printOuts(df)
    print('\ndisplaying figures')
    displayFigures(df)
    print('\nFinished')
