'''
A little script to generate a chunked version of mrqa.
For this version, we chunk the dataset into 512-length
chunks, to simulate bert-style preprocessing. 
'''

def chunk_sample(tokenizer, sample, is_train, stride=128, max_length=512):
    initial_sample = f"question: {sample['question']} context: "
    init_input_ids = tokenizer(initial_sample, add_special_tokens=False)['input_ids']
    start_len = len(init_input_ids)
    context = sample['context']
    # context = context.replace('[PAR]', '</s>')
    # context = context.replace('[DOC]', '</s>')
    # context = context.replace('[TLE]', '</s>')
    tokenized_output = tokenizer(context, return_offsets_mapping=True)
    context_tokens = tokenized_output['input_ids'][:-1]
    offsets = tokenized_output['offset_mapping'][:-1] # ignore the last (0,0) for </s>
    remaining_length = max_length - start_len - 1 # for '</s>'
    while len(context_tokens) > 0:
        chunk = context_tokens[:remaining_length] + [1]
        offsets_chunk = offsets[:remaining_length]
        # edge case: when the chunk is entirely within, finish up.
        # Otherwise we might add more chunks for sake of stride.
        if len(context_tokens) <= remaining_length:
            context_tokens = []
            offsets = []
        else:
            context_tokens = context_tokens[remaining_length-stride:] # stride for some overlap
            offsets = offsets[remaining_length-stride:]
        # assuming answer strings in same order as char spans.
        def detect_answer(sample, offsets_chunk):
            for i, span in enumerate(sample['detected_answers']['char_spans']):
                for start, end in zip(span['start'], span['end']): # we can have multiple answer instances
                    if start >= offsets_chunk[0][0] and end <= offsets_chunk[-1][-1]:
                        return sample['answers'][i]
            return '' # if we find nothing.
        chunk_ans = detect_answer(sample, offsets_chunk)
        yield {
            'question': sample['question'],
            'context': sample['context'],
            'input_ids': init_input_ids + chunk,
            'answer': chunk_ans,
            'qid': sample['qid'],
            'subset': sample['subset'],
            'task': 'mrqa'
        }
    
def chunk_dataset(tokenizer, dataset, stride=128, max_length=512):
    for sample in dataset:
        for chunked_sample in chunk_sample(tokenizer, sample, stride, max_length):
            yield chunked_sample

# testing
if __name__ == '__main__':
    from datasets import load_dataset
    from transformers import T5TokenizerFast
    tokenizer = T5TokenizerFast.from_pretrained('t5-base')
    mrqa = load_dataset('mrqa', split='train')
    print(f'MRQA has {len(mrqa)} samples')
    print(f'First sample: {mrqa[0]}')
    chunked_ds = list(chunk_dataset(tokenizer, mrqa, stride=128, max_length=512))
    print(f'Chunked MRQA has {len(chunked_ds)} samples')
    print(f'First sample: {chunked_ds[0]}')