"""A machine learning system that is a topic generator / labeler for given questions!


Input: M training examples, each consisting of a line for topic IDs followed by a line for the underlying question.

Eval: on N (unseen) questions.

Output: exactly 10 topic (integer id) suggestions (ordered by relevance) for each question in the eval dataset.
"""

def get_topics(q):
	pass

def get_guesses(q):
	pass

def score(q, G):
	topics = get_topics(q)
	numer = sum(math.sqrt(10-i) if G[i] in topics else 0 for i in range(10))
	numTopics = len(topics)
	denom = sum(math.sqrt(10-i) for i in range(min(numTopics, 10)))
	ratio = numer/denom
	return ratio

def raw_score(Q):
	return sum(score(q, get_guesses(q)) for q in Q)


def main():
	return raw_score([])
if __name__ == '__main__':
	main()
