import openai
from termcolor import colored
import streamlit as st

from database import get_redis_connection, get_redis_results

from config import CHAT_MODEL, COMPLETIONS_MODEL, INDEX_NAME

redis_client = get_redis_connection()

# A basic class to create a message as a dict for chat
# 채팅을 위한 딕셔너리로 메시지를 생성하는 기본 클래스
class Message:
    
    def __init__(self, role,content):
        self.role = role
        self.content = content
        
    def message(self):
        return {
            "role": self.role,
            "content": self.content
        }


# New Assistant class to add a vector database call to its responses
# 응답에 벡터 데이터베이스 호출을 추가하는 새로운 Assistant 클래스
class RetrievalAssistant:
    
    def __init__(self):
        self.conversation_history = []  

    def _get_assistant_response(self, prompt):
        try:
            completion = openai.ChatCompletion.create(
              model=CHAT_MODEL,
              messages=prompt,
              temperature=0.1
            )
            
            response_message = Message(
                completion['choices'][0]['message']['role'],
                completion['choices'][0]['message']['content']
            )
            return response_message.message()
            
        except Exception as e:

            return f'Request failed with exception {e}'
    
    # The function to retrieve Redis search results
    # Redis 검색 결과 조회 기능

    def _get_search_results(self,prompt):
        latest_question = prompt
        search_content = get_redis_results(
            redis_client,latest_question, 
            INDEX_NAME
        )['result'][0]

        return search_content
        
    def ask_assistant(self, next_user_prompt):
        [self.conversation_history.append(x) for x in next_user_prompt]
        assistant_response = self._get_assistant_response(self.conversation_history)
        
        # Answer normally unless the trigger sequence is used "searching_for_answers"
        # 트리거 시퀀스가 "searching_for_answers"로 사용되지 않는 한 정상적으로 응답합니다.
        if 'searching for answers' in assistant_response['content'].lower():
            # [prompt] 이 대화에서 사용자의 최근 질문과 해당 질문의 연도를 추출합니다:
            # {self.conversation_history}. 질문과 연도를 나타내는 문장으로 추출합니다.
            question_extract = openai.Completion.create(
                model = COMPLETIONS_MODEL, 
                prompt=f'''
                Extract the user's latest question and the year for that question from this 
                conversation: {self.conversation_history}. Extract it as a sentence stating the Question and Year"
            '''
            )
            search_result = self._get_search_results(question_extract['choices'][0]['text'])
            
            # We insert an extra system prompt here to give fresh context to the Chatbot on how to use the Redis results
            # Redis 결과를 사용하는 방법에 대해 Chatbot에 새로운 컨텍스트를 제공하기 위해 여기에 추가 시스템 프롬프트를 삽입합니다.
            # In this instance we add it to the conversation history, but in production it may be better to hide
            # 이 경우 대화 기록에 추가하지만 프로덕션에서는 숨기는 것이 더 나을 수 있습니다.
            # [prompt] 이 콘텐츠를 사용하여 사용자의 질문에 답변하세요: {search_result}.
            # 질문에 답할 수 없는 경우 '죄송합니다, 이 질문에 대한 답을 모릅니다'라고 말합니다.
            self.conversation_history.insert(
                -1,{
                "role": 'system',
                "content": f'''
                Answer the user's question using this content: {search_result}. 
                If you cannot answer the question, say 'Sorry, I don't know the answer to this one'
                '''
                }
            )
            
            assistant_response = self._get_assistant_response(
                self.conversation_history
                )
            
            self.conversation_history.append(assistant_response)
            return assistant_response
        else:
            self.conversation_history.append(assistant_response)
            return assistant_response
            
    def pretty_print_conversation_history(
            self, 
            colorize_assistant_replies=True):
        
        for entry in self.conversation_history:
            if entry['role']=='system':
                pass
            else:
                prefix = entry['role']
                content = entry['content']
                if colorize_assistant_replies and entry['role'] == 'assistant':
                    output = colored(f"{prefix}:\n{content}, green")
                else:
                    output = colored(f"{prefix}:\n{content}")
                print(output)
