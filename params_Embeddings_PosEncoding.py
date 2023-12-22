# params_Embeddings_PosEncoding.py – Software AnatomyNLM
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
Created on Tue Sep 26 21:12:26 2023

@author: Majd SALEH
"""

from keras.utils.layer_utils import count_params
from keras.layers import Embedding
from keras_nlp.layers import PositionEmbedding
from keras import Model,Input
import Transformer_params_calculator as params_calc

def main(n,v,d_e):
    """ This function compares the number of trainable parameters computed 
    using the formula in num_params_Embedding_PositionalEncoding and the one
    reported by keras.utils.layer_utils.count_params"""
    
    # Inputs:
    # n   : sequence length
    # v   : vocabulary size
    # d_e : dimension of embedding vector
    
    # Output:
    # the function prints a report of the considered coparison 
    
    # -------------------------------------------------------------------------
    # Create a simple model
    inputs = Input(shape=(n,))
    
    token_embeddings = Embedding(input_dim=v, output_dim=d_e)(inputs)
    position_embeddings = PositionEmbedding(sequence_length=n)(token_embeddings)
    
    embeddings = token_embeddings + position_embeddings
    
    model = Model(inputs=inputs, outputs=embeddings)
    
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
    params_Embedding=params_calc.num_params_embedding(v, d_e)
    params_PosEncoding= params_calc.num_params_positionalEncoding(n, d_e)
    num_params=params_Embedding+params_PosEncoding
    
    print("Trainable parameters' count from the derived formula:")
    print(num_params)
    print(separator)

if __name__ == '__main__':
    # Hyper-parameters
    n = 50  # sequence length
    v = 5000  # vocabulary size
    d_e = 128 # embedding dimension
    
    main(n,v,d_e)



