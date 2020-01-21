from parlai.core.agents import Agent

import copy
import uuid
import requests
import json
import queue
import os
import logging

from farm.infer import Inferencer
from farm.data_handler.utils import write_squad_predictions, write_nq_predictions, get_candidates


class BertqaAgent(Agent):

    def __init__(self, opt, userid=None):
        # initialize defaults first
        super().__init__(opt, userid)
        self.model_path = self.opt.get('init_model')
        self.proxies = {}
        self.episode_done = False
        self.solr_docs = None
        self.text = None
        self.qaid = None
        self.question = None
        self.reply = None
        self.report_log = {"pquad_explorer": {}, "logs": []}
        self.id = self.__class__.__name__
        self.top_k_doc = 3
        if self.opt.get('top_k_doc'):
            self.top_k_doc = self.opt.get('top_k_doc')
        self.batch_size = 4
        if self.opt.get('batch_size'):
            self.batch_size = self.opt.get('batch_size')
        self.top_k_candidates = 3
        if self.opt.get('top_k_candidates'):
            self.top_k_candidates = self.opt.get('top_k_candidates')

        self.model = Inferencer.load(self.model_path, batch_size=self.batch_size, gpu=False)
        logging.getLogger('farm.data_handler.processor').setLevel(logging.ERROR)
        logging.getLogger('farm.infer').setLevel(logging.ERROR)
        
        self.report_log = {"pquad_explorer": {}, "logs": []}
        # if user id is null, set a new one
        if userid is None:
            self.userid = str(uuid.uuid4())

    def reset(self):
        super().reset()
        self.userid = str(uuid.uuid4())

    def shutdown(self):
        # ending session
        super().shutdown()

    def observe(self, observation):
        observation = copy.deepcopy(observation)

        if self.episode_done:
            self.reset()
        
        if 'solr_docs' in observation:
            self.solr_docs = observation['solr_docs']
            self.question = observation['question']
        else:
            self.qaid = observation["qaid"]
            fields = observation['text'].strip().split('\n')
            # Data is expected to be text + '\n' + question
            if len(fields) < 2:
                raise RuntimeError('Invalid input. Is task a QA task?')
            self.text = '\n'.join(fields[:-1])
            self.question =fields[-1]
        return observation

    def report(self):
        return self.report_log

    def build_predictions_with_solr_docs(self, inferences, solr_docs):
        predictions = []
        top_k_span = 3
        candidates = []
        sorted_candidates = get_candidates(inferences, order=True)
        sorted_candidates = sorted_candidates[:(self.top_k_candidates+1)]

        solr_docs_set = {}
        for i, d in enumerate(solr_docs):
            solr_docs_set[d['id']] = {'doc': d, 'rank': i}

        sorted_candidates_with_docs = []
        for i, sc in enumerate(sorted_candidates):
            if sc.id in solr_docs_set:
                doc = solr_docs_set[sc.id]['doc']
                rank = solr_docs_set[sc.id]['rank']
                c = {
                    'doc_ident': doc['code'][0],
                    'doc_title': doc['title_text'][0],
                    'highlight': sc.context_string,
                    'answer': sc.span,
                    'doc_rank': rank,
                    'doc_score': doc['score'],
                    'answer_rank': i,
                    'answer_score': sc.answer_score,
                    'combined_score': doc['score']*sc.answer_score  # TODO: tune
                }
                sorted_candidates_with_docs.append(c)
        sorted_candidates_with_docs = sorted(sorted_candidates_with_docs, key=lambda c: c['combined_score'], reverse=True)
        return sorted_candidates_with_docs
    
    def report(self):
        return self.report_log

    def act(self):
        reply = {}
        reply['id'] = self.getID()

        if self.solr_docs:
            solr_docs = self.solr_docs[:self.top_k_doc]

            top_passages = []
            for d in solr_docs:
                top_passages.append(
                    {
                        "qas": [
                            {'question': self.question.lower(),
                            'id': d['id']
                            }
                        ],
                        "context": d['content_txt'][0].replace('<em>', '').replace('</em>', '').lower(),
                    }
                )

            inferences = self.model.inference_from_dicts(
                dicts=top_passages, rest_api_schema=True
            )

            predictions = self.build_predictions_with_solr_docs(inferences, solr_docs)
            self.reply = predictions

        else: 
            top_passages = [{
                "qas": [
                    {
                        'question': self.question,
                        'id': 'id'
                    }
                ],
                "context": self.text,
            }]

            inferences = self.model.inference_from_dicts(
                dicts=top_passages, rest_api_schema=True
            )

            sorted_candidates = get_candidates(inferences, order=True)
            sorted_candidates = sorted_candidates[:self.top_k_candidates]

            reply['text'] = sorted_candidates[0].span
            reply['text_candidates'] = []
            reply['candidate_scores'] = []
            self.report_log["logs"].append({"id": self.qaid, "question": self.question, "answer": reply["text"]})
            self.report_log["pquad_explorer"][self.qaid] = reply["text"]

            self.reply = reply

       
        return self.reply
