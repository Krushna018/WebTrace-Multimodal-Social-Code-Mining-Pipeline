from webtrace.models.baseline import BaselineModel


def test_baseline_fit_predict():
    texts = ["nice project", "great work", "you are stupid", "this is garbage", "helpful code", "idiot"]
    labels = [0, 0, 1, 1, 0, 1]
    m = BaselineModel.create().fit(texts, labels)
    pred = m.predict(["great code", "you idiot"])
    assert len(pred) == 2
