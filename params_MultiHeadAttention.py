# params_MultiHeadAttention.py – Software AnatomyNLM
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
Created on Tue Sep 26 14:52:03 2023

@author: Majd SALEH
"""

from keras import Model, Input
from keras.layers import MultiHeadAttention
from keras.utils.layer_utils import count_params
import Transformer_params_calculator as params_calc


def main(d_k,d_v,d_e,M,zeta):
    """ This function compares the number of trainable parameters computed 
    using the formula in num_params_MultiHeadAttention and the one reported by
    keras.utils.layer_utils.count_params"""
    
    # Inputs:
    # d_k : dimension of key vector
    # d_v : dimension of value vector
    # d_e : dimension of embedding vector
    # M   : number of attention heads
    # zeta: 1 ==> bias vectors are considered, 0 ==> they aren't considered
    
    # Output:
    # the function prints a report of the considered coparison
    
    # -------------------------------------------------------------------------
    # Create a simple model
    layer = MultiHeadAttention(num_heads=M, key_dim=d_k,value_dim=d_v,use_bias=zeta)
    input_tensor = Input(shape=[5, 4, d_e]) 
    output_tensor = layer(input_tensor, input_tensor)
    
    model = Model(input_tensor, output_tensor)
    
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
    
    # -------------------------------------------------------------------------
    # count trainable parameters from the derived formula
    num_params=params_calc.num_params_multiHeadAttention(d_k,d_v,d_e,M,zeta)
        
    print(separator)
    print("Trainable parameters' count from the derived formula:")
    print(num_params)
    print(separator)


if __name__ == '__main__':
    # Hyper-parameters   
    M=8 # number of self attention heads
    d_k=64 # diminsion of key vector 
    d_v=d_k # diminsion of value vector 
    d_e = M*d_k  # embedding dimention
        
    zeta=1 # use_bias (0 or 1)
    
    main(d_k,d_v,d_e,M,zeta)
