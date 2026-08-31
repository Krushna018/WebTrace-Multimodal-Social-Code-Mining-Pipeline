#!/usr/bin/env python3
import argparse
from webtrace.models.textcnn import load_textcnn, predict_textcnn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="results/textcnn")
    ap.add_argument("text", nargs="+")
    args = ap.parse_args()
    model, vocab, config = load_textcnn(args.model_dir)
    preds, probs = predict_textcnn(model, vocab, config, args.text)
    for text, pred, prob in zip(args.text, preds, probs):
        print({"text": text, "harmful": int(pred), "p_harmful": float(prob[1])})


if __name__ == "__main__":
    main()
