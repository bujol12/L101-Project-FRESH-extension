from Rationale_Analysis.models.saliency_scorer.base_saliency_scorer import SaliencyScorer
from Rationale_Analysis.data.dataset_readers.base_reader import to_token

from allennlp.models.model import Model
from lime import lime_text
from lime.lime_text import LimeTextExplainer
import numpy as np

from functools import partial
from copy import deepcopy
from typing import List

@Model.register("lime")
class LimeSaliency(SaliencyScorer):
    def __init__(self, model):
        super().__init__(model)

        #self.lime_explainer = LimeTextExplainer(mask_string=)

    def score(self, document, metadata, **kwargs):
        # metadata is the original input - modified that by LIME and re-tokenise
        # contains: annotation_id, human_rationale, document, label

        # use as labels for LIME
        preds = self._model['model'](document=document, metadata=metadata, **kwargs)
        labels = preds['predicted_labels']
        print(preds['probs'])

        for idx, mt in enumerate(metadata):
            reader_obj = document[idx]['reader_object']

            pred_fn = partial(self.__prepare_data_and_call_model, reader_obj=reader_obj, 
                annotation_id=mt['annotation_id'], human_rationale=mt['human_rationale'], mt_label=mt['label'], **kwargs)
            print(pred_fn(mt_documents=[mt['document']]))
        
        # TODO:
        # 0) Correcntly initialise LimeTextExplainer with the UNK token
        # 1) Call Lime Text Explainer in the above loop
        # 2) Return Lime token scores for each token
        # 3) Handle make_output_human_readable() missing error -> check gradients version
        # print(self._model['model']._vocabulary._oov_token)

        # print(document)
        # print(query)
        # print(document[0]["tokens"][5])

        # document[0]["tokens"][5] = to_token(self._model['model']._vocabulary._oov_token)
        # print(document[0]['reader_object'].combine_document_query(document, query, self._model['model']._vocabulary))

    def __prepare_data_and_call_model(self, annotation_id, human_rationale, mt_label, reader_obj, mt_documents: List[str], **kwargs):
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

            res.append(self._model['model'](**inp_dct)['probs'])

        return np.array(res)
