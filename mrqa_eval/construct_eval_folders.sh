mkdir in-domain
mkdir out-domain

echo "downloading in-domain data"

wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/SQuAD.jsonl.gz -O in-domain/SQuAD.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/NewsQA.jsonl.gz -O in-domain/NewsQA.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/TriviaQA-web.jsonl.gz -O in-domain/TriviaQA-web.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/SearchQA.jsonl.gz -O in-domain/SearchQA.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/HotpotQA.jsonl.gz -O in-domain/HotpotQA.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/NaturalQuestionsShort.jsonl.gz -O in-domain/NaturalQuestionsShort.jsonl.gz

echo "downloading out-domain data"

wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/BioASQ.jsonl.gz -O out-domain/BioASQ.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/DROP.jsonl.gz -O out-domain/DROP.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/DuoRC.ParaphraseRC.jsonl.gz -O out-domain/DuoRC.ParaphraseRC.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/RACE.jsonl.gz -O out-domain/RACE.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/RelationExtraction.jsonl.gz -O out-domain/RelationExtraction.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/TextbookQA.jsonl.gz -O out-domain/TextbookQA.jsonl.gz

echo "done!"