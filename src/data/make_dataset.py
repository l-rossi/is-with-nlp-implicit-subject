from datasets import load_dataset


def main():
    dataset = load_dataset("conll2012_ontonotesv5", "english_v12", split="train")
    print(dataset)


if __name__ == '__main__':
    main()
