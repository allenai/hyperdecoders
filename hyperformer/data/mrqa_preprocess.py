'''
A little script to generate a chunked version of mrqa.
For this version, we chunk the dataset into 512-length
chunks, to simulate bert-style preprocessing. 
'''

def chunk_sample(tokenizer, sample, stride=128, max_length=512):
    initial_sample = f"question: {sample['question']} context:"
    init_input_ids = tokenizer(initial_sample, add_special_tokens=False)['input_ids']
    start_len = len(init_input_ids)
    context = sample['context']
    # context = context.replace('[PAR]', '</s>')
    # context = context.replace('[DOC]', '</s>')
    # context = context.replace('[TLE]', '</s>')
    tokenized_output = tokenizer(context, return_offsets_mapping=True)
    context_tokens = tokenized_output['input_ids']
    offsets = tokenized_output['offset_mapping']
    remaining_length = max_length - start_len
    while len(context_tokens) > 0:
        chunk = context_tokens[:remaining_length]
        context_tokens = context_tokens[remaining_length-stride:] # stride for some overlap
        offsets_chunk = offsets[:remaining_length]
        # answer may not be possible with this chunk.
        chunk_ans = 'None'
         # im not sure that answers and spans are the same order, but this seems fine.
        for i, span in enumerate(sample['detected_answers']['char_spans']):
            if span['start'][0] >= offsets_chunk[0][0] and span['end'][0] <= offsets_chunk[-1][-1]:
                chunk_ans = sample['answers'][i]
                break
        yield {
            'question': sample['question'],
            'context': sample['context'],
            'input_ids': init_input_ids + chunk,
            'answer': chunk_ans,
            'qid': sample['qid'],
            'task': 'mrqa'
        }
    
    
def chunk_dataset(tokenizer, dataset, stride=128, max_length=512):
    for sample in dataset:
        for chunked_sample in chunk_sample(tokenizer, sample, stride, max_length):
            yield chunked_sample

# testing
if __name__ == '__main__':
    from datasets import load_dataset
    from transformers import T5Tokenizer
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    mrqa = load_dataset('mrqa', split='train')
    print(f'MRQA has {len(mrqa)} samples')
    print(f'First sample: {mrqa[0]}')
    chunked_ds = list(chunk_dataset(tokenizer, mrqa, stride=128, max_length=512))
    print(f'Chunked MRQA has {len(chunked_ds)} samples')
    print(f'First sample: {chunked_ds[0]}')