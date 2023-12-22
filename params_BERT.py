# params_BERT.py Software AnatomyNLM
# Copyright 2023 b<>com. All rights reserved.
# This software is licensed under the Apache License, Version 2.0.
# You may not use this file except in compliance with the license. 
# You may obtain a copy of the license at: 
# http://www.apache.org/licenses/LICENSE-2.0 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and 
# limitations under the License.

"""
Created on Wed Sep 27 07:46:20 2023

@author: Majd SALEH
"""

import keras_nlp
from keras.utils.layer_utils import count_params
import Transformer_params_calculator as params_calc

def main(L,d_k,d_v,d_e,M,d_f,v,n):
    """ This function compares the number of trainable parameters computed 
    using the formula in num_params_GPT2 and the one reported by
    keras.utils.layer_utils.count_params"""
    
    # Inputs:
    # L   : number of transformer blocks
    # d_k : dimension of key vector
    # d_v : dimension of value vector
    # d_e : dimension of embedding vector
    # M   : number of attention heads
    # d_f : feedforward dimension 
    # v   : vocabulary size
    # n   : sequence lenth (max)
    
    # Output:
    # the function prints a report of the considered coparison   
    
    # -------------------------------------------------------------------------
    # Load pretrained BERT language model
    model = keras_nlp.models.BertMaskedLM.from_preset("bert_base_en_uncased")
    # model = keras_nlp.models.BertBackbone.from_preset("bert_base_en_uncased")
    
    # -------------------------------------------------------------------------
    # Print model summary
    model.summary(expand_nested=True)
    
    # -------------------------------------------------------------------------
    # count trainable parameters from the model and print their shapes
    print("Trainable parameters' shapes:")
    separator="-"*60
    for tw in model.trainable_weights:
        print(tw.shape)
    print(separator)
    print("Trainable parameters' count obtained from the model:")
    total_tp=count_params(model.trainable_weights)
    print(total_tp)
    print(separator)
        
    # -------------------------------------------------------------------------
    # count trainable parameters from the derived formula
    num_params=params_calc.num_params_BERT(L,d_k,d_v,d_e,M,d_f,v,n)
    
    print("Trainable parameters' count from the derived formula:")
    print(num_params)
    print(separator)
    

if __name__ == '__main__':
    # Hyper parameters of BERT (bert_base_en_uncased)
    v=30522
    n=512
    d_e=768 #(12*64)
    M=12
    L=12
    d_k=64
    d_v=64
    d_f=3072
    
    main(L,d_k,d_v,d_e,M,d_f,v,n)
