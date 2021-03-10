from transformers import FSMTForConditionalGeneration, FSMTTokenizer
import tqdm

model_name = "facebook/wmt19-ru-en"
tokenizer = FSMTTokenizer.from_pretrained(model_name)
model = FSMTForConditionalGeneration.from_pretrained(model_name)

with open('test.ru.txt') as f:
    test = f.readlines()

with open('answer.txt', 'w+') as f:
    for i, val in tqdm.tqdm(enumerate(test), total=len(test)):
        inputs = tokenizer.encode(val, return_tensors="pt")
        outputs = model.generate(inputs, max_length=70, num_beams=7, early_stopping=True)
        sequence = (tokenizer.decode(outputs[0])).replace('</s>', '')
        f.write(sequence + "\n") if i != len(test) - 1 else f.write(sequence)
