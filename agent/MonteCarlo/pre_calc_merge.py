import pickle

def dicts(paths):
	for path in paths:
		with open(path, "rb") as fr:
			yield pickle.load(fr)

def merge(paths, out):

	merged = {}
	ds = tuple(dicts(paths))

	for key in ds[0].keys():
		pcs = [(d[key][0]*d[key][1], d[key][1]) for d in ds]
		wins, counts = zip(*pcs)
		wins = sum(wins)
		counts = sum(counts)
		merged[key] = (wins/counts, counts)
	
	for d in ds:
		assert len(merged) == len(d)

	with open(out, "wb") as fw:
		pickle.dump(merged, fw)

if __name__ == "__main__":
	paths = ["preflop-probabilities.run1.p", "preflop-probabilities.run2.p"]
	out = "preflop-probabilities.merged.p"

	merge(paths, out)
