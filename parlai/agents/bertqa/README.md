# BertQA

Agent built from https://github.com/deepset-ai/FARM repo

python examples/eval_model.py -m drqa -t squad:Index -mf "models:drqa/squad/model"

## Single task
python examples/eval_model.py -m bertqa -t fcom:DefaultWithId --init-model models:bertqa/bert-english-qa-large --report report_bertqa_large.json

## Multi task
python examples/eval_model.py -m bertqa -t fcom:DefaultWithId --init-model models:bertqa/bert-english-qa-large --report report_bertqa_large.json
python examples/eval_model.py -m bertqa -t squad:Index --init-model models:bertqa/bert-english-qa-large --report report_bertqa_large.json
