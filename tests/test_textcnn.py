import pandas as pd
from webtrace.models.textcnn import TextCNNConfig, train_textcnn, predict_textcnn


def test_textcnn_train_and_predict(tmp_path):
    train = pd.DataFrame({
        "text": [
            "thanks for sharing this code", "beautiful generative art", "nice github project", "helpful tutorial",
            "you are disgusting", "go away idiot", "i hate you", "stupid trash",
        ] * 3,
        "label_harmful": [0, 0, 0, 0, 1, 1, 1, 1] * 3,
    })
    valid = pd.DataFrame({
        "text": ["great code", "you idiot", "nice work", "stupid person"],
        "label_harmful": [0, 1, 0, 1],
    })
    cfg = TextCNNConfig(epochs=1, batch_size=4, embedding_dim=16, channels=8, max_length=24)
    model, vocab, _ = train_textcnn(train, valid, str(tmp_path), cfg)
    preds, probs = predict_textcnn(model, vocab, cfg, valid.text.tolist())
    assert len(preds) == 4
    assert probs.shape == (4, 2)
    assert (tmp_path / "model.pt").exists()
