# Transformer_params_calculator.py – Software AnatomyNLM
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
Created on Wed Sep 27 16:28:28 2023

@author: Majd SALEH
"""

def num_params_embedding(v,d_e):
    """ num_params_Embedding computes the number of trainable parameters in 
    embedding layer"""
    
    # Inputs:
    # v   : vocabulary size
    # d_e : dimension of embedding vector
    
    # Output:
    # number of trainable parameters in embedding layer
    
    return v*d_e

def num_params_positionalEncoding(n,d_e):
    """ num_params_PositionalEncoding computes the number of trainable
    parameters in positional encoding layer"""
    
    # Inputs:
    # n   : sequence length   
    # d_e : dimension of embedding vector
    
    # Output:
    # number of trainable parameters in positional encoding layer   
    
    return n*d_e

def num_params_multiHeadAttention(d_k,d_v,d_e,M,zeta):
    """ num_params_MultiHeadAttention computes the number of trainable 
    parameters in a  MultiHeadAttention layer"""
    
    # Inputs:
    # d_k : dimension of key vector
    # d_v : dimension of value vector
    # d_e : dimension of embedding vector
    # M   : number of attention heads
    # zeta: 1 ==> bias vectors are considered, 0 ==> they aren't considered
    
    # Output:
    # number of trainable parameters in MultiHeadAttention layer   
    
    return 2*M*d_e*(d_k+d_v)+zeta*(M*(2*d_k+d_v)+d_e)

def num_params_ffnn(d_f,d_e):
    """ num_params_ffnn computes the number of trainable parameters in the
    ffnn part of a transformer block"""
    
    # Inputs:
    # d_f : feedforward dimension (internal dimension)
    # d_e : dimension of embedding vector    
    
    # Output:
    # number of trainable parameters in the ffnn of a transformer block  
    
    return 2*d_f*d_e+d_e+d_f

def num_params_layerNorm(d_e):
    """ num_params_layerNorm computes the number of trainable parameters in
    layerNorm layer"""
    
    # Inputs:
    # d_e : dimension of embedding vector    
    
    # Output:
    # number of trainable parameters in layerNorm layer
    
    return 2*d_e

def num_params_transformerBlock(d_k,d_v,d_e,M,d_f):
    """ num_params_TransformerBlock computes the number of trainable 
    parameters in a  transformer block"""
    
    # Inputs:
    # d_k : dimension of key vector
    # d_v : dimension of value vector
    # d_e : dimension of embedding vector
    # M   : number of attention heads
    # d_f : feedforward dimension    
    
    # Output:
    # number of trainable parameters in transformer block   
    
    zeta=1 # use_bias in attention layer (zeta=1 fixed)
           # despite that TransformerEncoder and TransformerDecoder use 
           # MultiHeadAttention, thier interfaces don't enable accessing the 
           # argument use_bias. The latter defaults to true, thus zeta should be
           # be set to 1
           
    params_multiHeadAttention=num_params_multiHeadAttention(d_k,d_v,d_e,M,zeta)
        
    params_ffnn=num_params_ffnn(d_f,d_e)

    params_layerNorm_1=num_params_layerNorm(d_e)

    params_layerNorm_2=num_params_layerNorm(d_e)

    params_total=params_multiHeadAttention+\
                 params_ffnn+\
                 params_layerNorm_1+\
                 params_layerNorm_2
               
    return params_total

def num_params_transformer(L,d_k,d_v,d_e,M,d_f):
    """ num_params_transformer computes the number of trainable parameters in 
    a  transformer (stack of transformer blocks)"""
    
    # Inputs:
    # L   : number of transformer blocks
    # d_k : dimension of key vector
    # d_v : dimension of value vector
    # d_e : dimension of embedding vector
    # M   : number of attention heads
    # d_f : feedforward dimension    
    
    # Output:
    # number of trainable parameters in transformer   
    
    return L*num_params_transformerBlock(d_k,d_v,d_e,M,d_f)

def num_params_GPT2(L,d_k,d_v,d_e,M,d_f,v,n):
    """ num_params_GPT2 computes the number of trainable 
    parameters in the GPT2 AR LM """
    
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
    # number of trainable parameters in GPT2 AR LM        
    
    params_transformer=num_params_transformer(L,d_k,d_v,d_e,M,d_f)
    
    params_embedding=num_params_embedding(v,d_e)
    
    params_posEncoding=num_params_positionalEncoding(n,d_e)
    
    params_output_layerNorm=num_params_layerNorm(d_e)
    
    params_total=params_transformer+params_embedding+params_posEncoding+params_output_layerNorm
    
    # equivalent code
    # zeta=1 # use_bias in attention layer (zeta=1 fixed)
    #        # despite that TransformerEncoder and TransformerDecoder use 
    #        # MultiHeadAttention, thier interfaces don't enable accessing the 
    #        # argument use_bias. The latter defaults to true, thus zeta should be
    #        # be set to 1
           
    # params_total=d_e*(v+n+2)+L*(2*M*d_e*(d_k+d_v)+zeta*(M*(2*d_k+d_v)+d_e)+2*d_e*d_f+5*d_e+d_f)
    
    return params_total

def num_params_segmentEncoding(d_e):
    """ num_params_segmentEncoding computes the number of trainable 
    parameters in the segment embedding layer of BERT"""
    
    # Inputs:
    # d_e : dimension of embedding vector  
    
    # Output:
    # number of trainable parameters of the segment embedding layer of BERT
    
    return 2*d_e

def num_params_pooled_dense(d_e):
    """ num_params_pooled_dense computes the number of trainable parameters in 
    the pooled dense layers of BERT"""
    
    # Inputs:
    # d_e : dimension of embedding vector  
    
    # Output:
    # number of trainable parameters of the pooled dense layer of BERT
    
    return d_e*d_e+d_e # a simple dense layer with a kernel W (d_e x d_e) and 
                     # a bias b (d_e)

def num_params_BERT_Backbone(L,d_k,d_v,d_e,M,d_f,v,n):
    """ num_params_BERT_Backbone computes the number of trainable 
    parameters in BERT Backbone """
    
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
    # number of trainable parameters in BERT Backbone
    
    params_embedding=num_params_embedding(v,d_e)
    
    params_posEncoding=num_params_positionalEncoding(n,d_e)
    
    params_segEncoding=num_params_segmentEncoding(d_e)
    
    params_embedding_layerNorm=num_params_layerNorm(d_e)    
      
    params_transformer=num_params_transformer(L,d_k,d_v,d_e,M,d_f)
    
    params_pooled_dense_CLS=num_params_pooled_dense(d_e) # a dense layer on top of the [CLS] token
    
    # total number of parameters
    params_total=params_embedding+\
                 params_posEncoding+\
                 params_segEncoding+\
                 params_embedding_layerNorm+\
                 params_transformer+\
                 params_pooled_dense_CLS
                 
    return params_total

def num_params_MLM_Head(L,d_k,d_v,d_e,M,d_f,v,n):
    """ num_params_MLM_Head computes the number of trainable 
    parameters in an MLM Head """
    
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
    # number of trainable parameters in MLM head
    
    params_pooled_dense_MLM=num_params_pooled_dense(d_e) # a dense layer of the MLM head
    
    params_MLMHead_layerNorm=num_params_layerNorm(d_e)
    
    params_bias_Embedding_transpose=v
    
    # total number of parameters
    params_total=params_pooled_dense_MLM+\
                 params_MLMHead_layerNorm+\
                 params_bias_Embedding_transpose
                 
    return params_total

def num_params_BERT(L,d_k,d_v,d_e,M,d_f,v,n):
    """ num_params_BERT computes the number of trainable 
    parameters in BERT model """
    
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
    # number of trainable parameters of BERT (BERT backbone + MLM head)
                 
    params_backbone=num_params_BERT_Backbone(L,d_k,d_v,d_e,M,d_f,v,n)
    params_head=num_params_MLM_Head(L,d_k,d_v,d_e,M,d_f,v,n)
    
    params_total=params_backbone+params_head
    
    return params_total
