class SST2Template:
    verbalizer = {0: "terrible", 1: "great"}

    def encode(self, sample):
        text = sample.data["sentence"].strip()
        return f"{text} It was"

    def verbalize(self, sample, candidate):
        text = sample.data["sentence"].strip()
        return f"{text} It was {self.verbalizer[candidate]}"


class SQuADTemplate:
    def encode(self, sample):
        question = sample.data['question'].strip()
        title = sample.data['title']
        context = sample.data['context']
        return f"Title: {title}\nContext: {context}\nQuestion: {question}\nAnswer:"

    def verbalize(self, sample, candidate):
        question = sample.data['question'].strip()
        title = sample.data['title']
        context = sample.data['context']
        answer = sample.data['answers'][0]

        return f"Title: {title}\nContext: {context}\nQuestion: {question}\nAnswer: {answer}\n"
