from Rationale_Analysis.models.saliency_scorer.base_saliency_scorer import SaliencyScorer
from Rationale_Analysis.data.dataset_readers.base_reader import to_token

from allennlp.models.model import Model
from lime import lime_text
from lime.lime_text import LimeTextExplainer
import numpy as np
import torch

from functools import partial
from copy import deepcopy
from typing import List

@Model.register("lime")
class LimeSaliency(SaliencyScorer):
    def __init__(self, model):
        super().__init__(model)

        self.lime_explainer = LimeTextExplainer(mask_string=model._vocabulary._oov_token, bow=False)

    def score(self, document, metadata, **kwargs):
        # metadata is the original input - modified that by LIME and re-tokenise
        # contains: annotation_id, human_rationale, document, label

        # use as labels for LIME
        output_dicts = self._model['model'](document=document, metadata=metadata, **kwargs)
        labels = output_dicts['predicted_labels']

        for idx, mt in enumerate(metadata):
            reader_obj = document[idx]['reader_object']

            pred_fn = partial(self.__prepare_data_and_call_model, reader_obj=reader_obj, 
                annotation_id=mt['annotation_id'], human_rationale=mt['human_rationale'], mt_label=mt['label'], **kwargs)
            label = labels[idx].cpu().item()
            exp = self.lime_explainer.explain_instance(mt['document'], pred_fn, num_features=4096, labels=(label, ), num_samples=500)    
            scores_map = exp.as_map()[label]

            scores = torch.zeros(len(scores_map))
            
            for i, score in scores_map:
                scores[i] = max(score, 0.0)
            
            output_dicts['attentions'][idx][:len(scores)] = scores
        
        return output_dicts
        
    def __prepare_data_and_call_model(self, mt_documents: List[str], annotation_id, human_rationale, mt_label, reader_obj, **kwargs):
        res = []
        for mt_document in mt_documents:

            instance = reader_obj.text_to_instance(
                annotation_id,
                mt_document,
                None,
                None,
                human_rationale
            )

            inp_dct = instance.as_tensor_dict()
            for key, val in inp_dct.items():
                inp_dct[key] = [val]

            res.append(self._model['model'](**inp_dct)['probs'].cpu().numpy())

        return np.vstack(res)
