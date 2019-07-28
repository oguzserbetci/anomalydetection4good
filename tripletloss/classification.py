import sys
import pickle
import numpy as np

"""
Example pickle file:
   [(array([ 0.07532644, -0.0781407 ]),
     '/notebooks/data/datasets/pipistrel/Hackathon/SingleFrame_ObjectProposalClassification/test/nature/14005.png'),
    (array([ 0.07638371, -0.07900357]),
     '/notebooks/data/datasets/pipistrel/Hackathon/SingleFrame_ObjectProposalClassification/test/nature/16590.png'),
    ...

"""

def print_usage():
  print("usage: classification.py <baseline.pickle> <embeddings.pickle>")

def sq_dist(a, b):
  return np.linalg.norm(a-b)
  # return (a[0]-b[0])**2 + (a[1]-b[1])**2

def parse_baseline(base):
  """Find the average location of boats and nature in the baseline data"""

  b = np.array([ x[0] for x in base if "boat" in x[1] ])
  n = np.array([ x[0] for x in base if "nature" in x[1] ])

  print("loaded {} boat and {} nature baseline samples".format(len(b), len(n)))

  avg_b = np.average(b, 0)
  avg_n = np.average(n, 0)
  dist = sq_dist(avg_b, avg_n)

  print("boats\n\t{} average\n\t{} variance".format(avg_b, np.var(b, 0)))
  print("nature\n\t{} average\n\t{} variance".format(avg_n, np.var(n, 0)))
  print("squared distance of cluster means:\n\t{}".format(dist))

  return avg_b, avg_n

def classify(sample, avg_boat, avg_nature):
  """Classify a sample as boat or nature"""

  d_boat = sq_dist(sample, avg_boat)
  d_nature = sq_dist(sample, avg_nature)
  return 1.0 if d_boat < d_nature else 0.0

if __name__ == "__main__":
  if len(sys.argv) != 3:
    print_usage()
    exit(1)
  basefile = sys.argv[1]
  testfile = sys.argv[2]

  with open(basefile, "rb") as f:
    base = pickle.load(f)
  with open(testfile, "rb") as f:
    test = pickle.load(f)

  avg_b, avg_n = parse_baseline(base) 

  for sample in test:
    result = classify(sample[0], avg_b, avg_n)
    name = sample[1].split("/")[-1]
    print("{},{}".format(name, result))
