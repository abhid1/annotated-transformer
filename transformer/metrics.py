import sacrebleu

def evaluate_bleu(predictions, labels):
    try:
        bleu_sacre = sacrebleu.raw_corpus_bleu(predictions, [labels], .01).score
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as e:
        print("\nWARNING: Could not compute BLEU-score. Error:", str(e))

    return bleu_sacre
