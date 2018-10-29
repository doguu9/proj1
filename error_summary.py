from PIL import Image
import numpy as np
#load error list in format [(ground, prediction, confidence), ...]

#SUMMARY
#total errors
#top missed images, top predictions
#average confidence of errors

#OPTIONS
#show all errors
#show side-by-side comparison of ground and prediction

#total errors
def convert(errors):
    e = np.ndarray.tolist(errors)
    for error in e:
        error[2] = float(error[2])
    return e

def summary(errors):
	
    # subroutine: make dictionary
    counts = {}
    for error in errors:
	if error[0] in counts:
	    counts[error[0]] += 1
	else:
	    counts[error[0]] = 1
    one = max(counts, key=counts.get)
    one_v = counts[one]
    counts.pop(one, None)
    two = max(counts, key=counts.get)
    two_v = counts[two]
    counts.pop(two, None)
    three = max(counts, key=counts.get)
    three_v = counts[three]

    pairs = {}
    for error in errors:
    	if (error[0], error[1]) in pairs:
    	    pairs[(error[0], error[1])] += 1
    	else:
    	    pairs[(error[0], error[1])] = 1
    four = max(pairs, key=pairs.get)
    four_v = pairs[four]
    pairs.pop(four, None)
    five = max(pairs, key=pairs.get)
    five_v = pairs[five]
    pairs.pop(five, None)
    six = max(pairs, key=pairs.get)
    six_v = pairs[six]

    summation = 0
    for error in errors:
    	summation += error[2]
    avg = summation / len(errors)

    print('Total # of Errors: {}'.format(len(errors)))
    print('Most commonly missed images: (1) {} - {}, (2) {} - {}, (3) {} - {}'.format(one, one_v, two, two_v, three, three_v))
    print('Most common erroneous predictions: (1) {} - {}, (2) {} - {}, (3) {} - {}'.format(four, four_v, five, five_v, six, six_v))
    print('Average erroneous prediction confidence: {}'.format(avg))

def show_errors(errors):
    print('Showing all errors...')
    print()
    for error in errors:
        print('Object: {}        Prediction: {}        Confidence: {}'.format(error[0], error[1], error[2]))

def visualize(error, directory):
    label = error[0]
    pred = error[1]
    ground = Image.open('{}/{}/{}_000000.png'.format(directory, label, label))
    prediction = Image.open('{}/{}/{}_000000.png'.format(directory, pred, pred))
    ground.show()
    prediction.show()
