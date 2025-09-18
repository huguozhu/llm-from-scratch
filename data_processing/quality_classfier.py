import fasttext


def train(
    input_file="data/quality_classifier/train.txt",
    epoch=25,
    lr=1.0,
    wordNgrams=2,
    verbose=2,
    minCount=1,
    loss="softmax",
    checkpoint_path="checkpoints/quality_classifier.bin",
):
    model = fasttext.train_supervised(
        input=input_file,
        epoch=epoch,
        lr=lr,
        wordNgrams=wordNgrams,
        verbose=verbose,
        minCount=minCount,
        loss=loss,
    )

    model.save_model(checkpoint_path)


class QualityClassifier:
    def __init__(self, model_path="checkpoints/quality_classifier.bin"):
        self.model = fasttext.load_model(model_path)
        self.label_prefix = "__label__"

    def identify(self, text: str, k=1):
        if not text.strip():
            return "low-quality", 0.0
        text = text.replace("\n", " ")
        labels, probs = self.model.predict(text, k=1)

        label = labels[0].replace("__label__", "")
        confidence = float(probs[0])

        if label not in ["high_quality", "low_quality"]:
            return "low-quality", 0.0

        label = label.replace("_", "-")

        return label, confidence


if __name__ == "__main__":
    train()
