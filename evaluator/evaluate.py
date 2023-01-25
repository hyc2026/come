from rouge import Rouge
import sys
from nltk.translate.meteor_score import single_meteor_score
# import nltk
# nltk.download('wordnet')
# nltk.download('omw-1.4')

def evaluate(ref, hyp):
    with open(hyp, 'r') as r:
        hypothesis = r.readlines()
        hyps = [v.strip().lower() for v in hypothesis]
    with open(ref, 'r') as r:
        references = r.readlines()
        refs = [v.strip().lower() for v in references]

    sentence_meteor_lst = [single_meteor_score(ref_sentence, gen_sentence) for ref_sentence, gen_sentence in zip(refs, hyps)]
    stc_meteor = sum(sentence_meteor_lst) / len(sentence_meteor_lst)
    print("Meteor = " + str(stc_meteor * 100))

    rouge = Rouge()
    scores_Rouge = rouge.get_scores(hyps=hyps, refs=refs, avg=True)
    for k, v in scores_Rouge.items():
        print(k + ' = ' + str(v['f'] * 100))

if __name__ == '__main__':
    evaluate(sys.argv[1], sys.argv[2])
