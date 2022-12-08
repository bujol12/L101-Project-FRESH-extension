from typing import Optional, Dict, Any

from Rationale_Analysis.models.thresholders.base_thresholder import Thresholder
from allennlp.models.model import Model
from allennlp.training.metrics import F1Measure
import math
import torch
import numpy as np

@Model.register("top_k")
class TopKThresholder(Thresholder) :
    def __init__(self, max_length_ratio: float) :
        self._max_length_ratio = max_length_ratio
        self.tok_f1 = F1Measure(1)
        super().__init__()

    def forward(self, attentions, document, metadata=None) :
        rationales = self.extract_rationale(attentions=attentions, document=document)
        output_dict = {'metadata' : metadata, 'rationale' : rationales}
        
        self._calculate_metrics(output_dict)

        return output_dict
 
    def extract_rationale(self, attentions, document, as_one_hot=False):
        attentions = attentions.cpu().data.numpy()
        document_tokens = [x["tokens"] for x in document] 

        assert len(attentions) == len(document)
        assert attentions.shape[1] == max([len(d['tokens']) for d in document])

        rationales = []
        for b in range(attentions.shape[0]):
            sentence = [x.text for x in document_tokens[b]]
            attn = attentions[b][:len(sentence)]
            max_length = math.ceil(len(sentence) * self._max_length_ratio)
            
            top_ind, top_vals = np.argsort(attn)[-max_length:], np.sort(attn)[-max_length:]
            if as_one_hot :
                rationales.append([1 if i in top_ind else 0 for i in range(attentions.shape[1])])
                continue
            
            rationales.append({
                'document' : " ".join([x for i, x in enumerate(sentence) if i in top_ind]),
                'spans' : [{'span' : (i, i+1), 'value' : float(v)} for i, v in zip(top_ind, top_vals)],
            })

        return rationales

    def _calculate_metrics(self, output_dict):
        
        for d_idx in range(len(output_dict['rationale'])):
            pred = output_dict['rationale'][d_idx]['spans']
            gold = output_dict['metadata'][d_idx]['human_rationale']
            doc = output_dict['rationale'][d_idx]['document']

            golds = torch.zeros(len(doc))
            preds = torch.zeros((len(doc), 2))
            preds[:, 0] = 1

            for start, finish in gold:
                golds[start:finish] = 1
            for curr_span in pred:
                start, finish = curr_span['span']
                preds[start:finish, 1] = 1
                preds[start:finish, 0] = 0
            
            self.tok_f1(preds, golds)

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = self.tok_f1.get_metric(reset)
        metrics = dict(zip(["p", "r", "f1"], metrics))
        return metrics

            