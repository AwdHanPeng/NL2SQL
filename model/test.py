from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
print(tokenizer.tokenize("AppointmentID"))
print(tokenizer.tokenize("PrimaryAffiliation"))
