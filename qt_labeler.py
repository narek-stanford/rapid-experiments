"""A machine learning system that is a topic generator / labeler for given questions!


Input: M training examples, each consisting of a line for topic IDs followed by a line for the underlying question.

Eval: on N (unseen) questions.

Output: exactly 10 topic (integer id) suggestions (ordered by relevance) for each question in the eval dataset.
"""
def score(question):
	pass
def raw_score(Questions):
	return sum(score(question) for question in Questions)
def main():
	return raw_score([])
if __name__ == '__main__':
	main()
