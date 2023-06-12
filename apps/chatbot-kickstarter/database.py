import pandas as pd 
import numpy as np
import openai
from redis import Redis
from redis.commands.search.field import VectorField
from redis.commands.search.field import TextField, NumericField
from redis.commands.search.query import Query

from config import EMBEDDINGS_MODEL, PREFIX, VECTOR_FIELD_NAME

# Get a Redis connection
def get_redis_connection(host='localhost',port='6379',db=0):
    
    r = Redis(host=host, port=port, db=db,decode_responses=False)
    return r

# Create a Redis index to hold our data
def create_hnsw_index (redis_conn,vector_field_name,vector_dimensions=1536, distance_metric='COSINE'):
    redis_conn.ft().create_index([
        VectorField(vector_field_name, "HNSW", {"TYPE": "FLOAT32", "DIM": vector_dimensions, "DISTANCE_METRIC": distance_metric}),
        TextField("filename"),
        TextField("text_chunk"),        
        NumericField("file_chunk_index")
    ])

# Create a Redis pipeline to load all the vectors and their metadata
def load_vectors(client:Redis, input_list, vector_field_name):
    p = client.pipeline(transaction=False)
    for text in input_list:    
        #hash key
        key=f"{PREFIX}:{text['id']}"
        
        #hash values
        item_metadata = text['metadata']
        #
        item_keywords_vector = np.array(text['vector'],dtype= 'float32').tobytes()
        item_metadata[vector_field_name]=item_keywords_vector
        
        # HSET
        p.hset(key,mapping=item_metadata)
            
    p.execute()

# Make query to Redis
# Redis에 쿼리하기
def query_redis(redis_conn,query,index_name, top_k=2):
    
    

    ## Creates embedding vector from user query
    ## 사용자 쿼리에서 임베딩 벡터 생성
    embedded_query = np.array(openai.Embedding.create(
                                                input=query,
                                                model=EMBEDDINGS_MODEL,
                                            )["data"][0]['embedding'], dtype=np.float32).tobytes()

    #prepare the query
    #쿼리 준비
    q = Query(f'*=>[KNN {top_k} @{VECTOR_FIELD_NAME} $vec_param AS vector_score]').sort_by('vector_score').paging(0,top_k).return_fields('vector_score','filename','text_chunk','text_chunk_index').dialect(2) 
    params_dict = {"vec_param": embedded_query}

    
    #Execute the query
    #쿼리 실행
    results = redis_conn.ft(index_name).search(q, query_params = params_dict)
    
    return results

# Get mapped documents from Weaviate results
# Weaviate 결과에서 매핑된 문서 가져오기
def get_redis_results(redis_conn,query,index_name):
    
    # Get most relevant documents from Redis
    # Redis에서 가장 관련 있는 문서 가져오기
    query_result = query_redis(redis_conn,query,index_name)
    
    # Extract info into a list
    # 정보를 목록으로 추출
    query_result_list = []
    for i, result in enumerate(query_result.docs):
        result_order = i
        text = result.text_chunk
        score = result.vector_score
        query_result_list.append((result_order,text,score))
        
    # Display result as a DataFrame for ease of us
    # 우리의 편의를 위해 결과를 DataFrame으로 표시
    result_df = pd.DataFrame(query_result_list)
    result_df.columns = ['id','result','certainty']
    return result_df