# params_Transformer.py – Software AnatomyNLM
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
Created on Tue Sep 26 19:53:49 2023

@author: Majd SALEH
"""

from keras.utils.layer_utils import count_params
from keras_nlp.layers import TransformerEncoder
from keras import Model,Input
import Transformer_params_calculator as params_calc


def main(L,d_k,d_v,d_e,M,d_f,encoder):
    """ This function compares the number of trainable parameters computed 
    using the formula in num_params_TransformerBlock and the one reported by
    keras.utils.layer_utils.count_params"""
    
    # Inputs:
    # L   : number of transformer blocks
    # d_k : dimension of key vector
    # d_v : dimension of value vector
    # d_e : dimension of embedding vector
    # M   : number of attention heads
    # d_f : feedforward dimension    
    # encoder : boolean variable; ture to use encoder, false to use decoder. This
    # is only to show that this will make no difference on the number of trainable
    # parameters
    
    # Output:
    # the function prints a report of the considered coparison   
    
    # -----------------------------------------------------------------------------
    # Create a transformer model (stacked transformer blocks)
    inputs= Input(shape=(10, d_e))
    trans_output = inputs

    for l in range(L):
        transformer_block = TransformerEncoder(intermediate_dim=d_f, num_heads=M)
        trans_output = transformer_block(trans_output)
        
    model = Model(inputs=inputs, outputs=trans_output)

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
    num_params=params_calc.num_params_transformer(L,d_k,d_v,d_e,M,d_f)
    
    print("Trainable parameters' count from the derived formula:")
    print(num_params)
    print(separator)

if __name__ == '__main__':
    # Hyper-parameters
    L=3
    M=8 # number of self attention heads
    d_k=16 # diminsion of key vector 
    d_v=d_k # diminsion of value vector 
    d_e = M*d_k  # embedding dimention
    d_f=2048 # feedforward dimension

    encoder=False # encoder mode or decoder mode otherwise
    
    main(L,d_k,d_v,d_e,M,d_f,encoder)


